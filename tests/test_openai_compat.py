"""OpenAI-compatible client against a mocked HTTP transport (no network)."""
from __future__ import annotations

import json

import httpx
import pytest

from app.agent.planner import PLAN_SCHEMA
from app.agent.models import PlannerStep
from app.config import Settings
from app.container import build_llm
from app.llm.base import LLMError, call_structured
from app.llm.openai_compat import OpenAICompatClient, _simplify_schema

PLAN = {"sub_questions": [{"id": "q1", "text": "t", "status": "OPEN", "evidence_ids": []}],
        "action": {"type": "finalize", "args": {}, "reason": "r"}}


def client(handler, mode="auto"):
    sent = []

    def wrapped(request: httpx.Request):
        body = json.loads(request.content)
        sent.append(body)
        return handler(body, len(sent))

    c = OpenAICompatClient("https://llm.test/v1", "key", "test-model", mode=mode,
                           transport=httpx.MockTransport(wrapped), sleep=lambda s: None)
    return c, sent


def tool_reply(args):
    return httpx.Response(200, json={"choices": [{"message": {"tool_calls": [
        {"type": "function", "function": {"name": "plan_step", "arguments": json.dumps(args)}}]}}]})


def test_tools_mode_forces_function_and_parses_arguments():
    c, sent = client(lambda body, n: tool_reply(PLAN))
    out = c.complete_json(purpose="plan_step", system="s", user="u", schema=PLAN_SCHEMA)
    assert out == PLAN
    assert sent[0]["tool_choice"] == {"type": "function", "function": {"name": "plan_step"}}
    assert sent[0]["messages"][0] == {"role": "system", "content": "s"}


def test_falls_back_to_json_mode_when_tools_rejected():
    def handler(body, n):
        if "tools" in body:
            return httpx.Response(400, json={"error": {"message": "tool_choice not supported"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(PLAN)}}]})

    c, sent = client(handler)
    assert json.loads(c.complete_json(purpose="plan_step", system="s", user="u", schema=PLAN_SCHEMA)) == PLAN
    assert c.mode == "json" and sent[1]["response_format"] == {"type": "json_object"}
    assert "JSON schema" in sent[1]["messages"][0]["content"]
    c.complete_json(purpose="plan_step", system="s", user="u", schema=PLAN_SCHEMA)
    assert "tools" not in sent[2]                      # stays in json mode


def test_rate_limit_is_retried():
    def handler(body, n):
        if n == 1:
            return httpx.Response(429, headers={"retry-after": "2"}, json={"error": {"message": "slow down"}})
        return tool_reply(PLAN)

    c, sent = client(handler)
    assert c.complete_json(purpose="plan_step", system="s", user="u", schema=PLAN_SCHEMA) == PLAN
    assert len(sent) == 2


def test_auth_error_becomes_llm_error():
    c, _ = client(lambda body, n: httpx.Response(401, json={"error": {"message": "bad key"}}), mode="json")
    with pytest.raises(LLMError):
        c.complete_json(purpose="p", system="s", user="u", schema={})


def test_unknown_model_is_a_config_error_not_a_mode_switch():
    body = {"error": {"message": "The model `old` does not exist", "code": "model_not_found"}}
    c, sent = client(lambda b, n: httpx.Response(404, json=body))
    with pytest.raises(LLMError, match="configuration error"):
        c.complete_json(purpose="p", system="s", user="u", schema={})
    assert len(sent) == 1 and c.mode == "tools"


def test_failed_tool_generation_goes_to_repair():
    def handler(body, n):
        if n == 1:
            return httpx.Response(400, json={"error": {"code": "tool_use_failed",
                                                       "failed_generation": "{\"sub_questions\": oops"}})
        return tool_reply(PLAN)

    c, sent = client(handler)
    res = call_structured(c, purpose="plan_step", system="s", user="u", schema=PLAN_SCHEMA, model_cls=PlannerStep)
    assert res.value is not None and res.attempts == 2
    assert "INVALID" in sent[1]["messages"][1]["content"]


def test_schema_simplified_for_strict_providers():
    s = _simplify_schema({"type": "object", "properties": {"v": {"type": ["string", "null"]}}})
    assert s["properties"]["v"] == {"type": "string", "nullable": True}


def test_provider_selection(tmp_path):
    s = Settings(data_dir=tmp_path, llm_provider="groq", llm_base_url="https://api.groq.com/openai/v1",
                 llm_api_key="k", llm_model="openai/gpt-oss-120b", _env_file=None)
    assert isinstance(build_llm(s), OpenAICompatClient)
    with pytest.raises(ValueError):
        build_llm(Settings(data_dir=tmp_path, llm_provider="nope", _env_file=None))


def test_parse_duration():
    from app.llm.openai_compat import _parse_duration

    assert _parse_duration("7.66s") == 7.66
    assert abs(_parse_duration("2m59.5s") - 179.5) < 1e-9
    assert _parse_duration("120ms") == 0.12
    assert _parse_duration("3") == 3.0 and _parse_duration(None) is None


def test_paces_calls_using_rate_limit_headers():
    waits = []
    now = [100.0]

    def handler(request):
        return httpx.Response(200, headers={"x-ratelimit-remaining-tokens": "50",
                                            "x-ratelimit-reset-tokens": "12s"},
                              json={"choices": [{"message": {"tool_calls": [{"function": {
                                  "arguments": json.dumps(PLAN)}}]}}]})

    c = OpenAICompatClient("https://llm.test/v1", "k", "m", transport=httpx.MockTransport(handler),
                           sleep=waits.append, clock=lambda: now[0])
    c.complete_json(purpose="plan_step", system="s", user="u", schema=PLAN_SCHEMA)
    assert waits == []                                   # first call: nothing known yet
    c.complete_json(purpose="plan_step", system="s", user="u" * 2000, schema=PLAN_SCHEMA)
    assert len(waits) == 1 and 12 <= waits[0] <= 13      # waited for the token window to reset


def test_reasoning_effort_auto_for_gpt_oss_and_dropped_if_rejected():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body.get("reasoning_effort"))
        if "reasoning_effort" in body:
            return httpx.Response(400, json={"error": {"message": "unknown parameter reasoning_effort"}})
        return httpx.Response(200, json={"choices": [{"message": {"tool_calls": [{"function": {
            "arguments": json.dumps(PLAN)}}]}}]})

    c = OpenAICompatClient("https://llm.test/v1", "k", "openai/gpt-oss-120b",
                           transport=httpx.MockTransport(handler), sleep=lambda s: None)
    assert c.reasoning_effort == "low"
    assert c.complete_json(purpose="plan_step", system="s", user="u", schema=PLAN_SCHEMA) == PLAN
    assert seen == ["low", None] and c.reasoning_effort is None and c.mode == "tools"

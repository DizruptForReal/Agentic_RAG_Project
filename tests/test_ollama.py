"""Local model via Ollama: provider defaults, strict JSON output, fallback when strict output is ignored,
and the setup checks in scripts/check_llm.py (against a fake Ollama server - no Ollama needed)."""
from __future__ import annotations

import json

import httpx

from app.config import Settings
from app.container import apply_provider_defaults, build_llm
from app.llm.openai_compat import OpenAICompatClient, is_ollama, ollama_root, supports_strict


def test_provider_ollama_needs_no_other_llm_settings():
    s = Settings(llm_provider="ollama", _env_file=None)
    apply_provider_defaults(s)
    assert s.llm_base_url == "http://localhost:11434/v1"
    assert s.llm_model == "gpt-oss:20b"
    assert s.llm_timeout_s >= 300
    llm = build_llm(s)
    from app.llm.ollama import OllamaClient

    assert isinstance(llm, OllamaClient) and llm.models == ["gpt-oss:20b"]   # native API, no key, no fallback
    assert llm.num_ctx == 16384 and llm.keep_alive == "30m"


def test_explicit_model_and_url_are_kept():
    s = Settings(llm_provider="ollama", llm_model="qwen3.5:9b", llm_base_url="http://gpu-box:11434/v1", _env_file=None)
    apply_provider_defaults(s)
    assert (s.llm_model, s.llm_base_url) == ("qwen3.5:9b", "http://gpu-box:11434/v1")


def test_ollama_detection_and_root():
    assert is_ollama("http://localhost:11434/v1") and not is_ollama("https://api.groq.com/openai/v1")
    assert ollama_root("http://localhost:11434/v1") == "http://localhost:11434"
    assert supports_strict("http://localhost:11434/v1", "any-model")


def _client(handler):
    return OpenAICompatClient("http://localhost:11434/v1", "", "gpt-oss:20b", transport=httpx.MockTransport(handler),
                              sleep=lambda s: None)


def test_strict_request_sends_the_schema_to_ollama():
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"answer": "Paris", "number": 5}'}}]})

    c = _client(handler)
    out = c.complete_json(purpose="ping", system="s", user="u",
                          schema={"type": "object", "properties": {"answer": {"type": "string"},
                                                                   "number": {"type": "integer"}}})
    assert out == {"answer": "Paris", "number": 5}
    assert sent[0]["response_format"]["type"] == "json_schema"
    assert sent[0]["reasoning_effort"] == "low"                             # gpt-oss: short hidden reasoning


def test_ignored_strict_output_switches_to_tool_calling():
    def handler(request):
        body = json.loads(request.content)
        if "tools" in body:
            return httpx.Response(200, json={"choices": [{"message": {"tool_calls": [
                {"type": "function", "function": {"name": "ping", "arguments": '{"answer": "ok"}'}}]}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Sure! The answer is ok."}}]})

    c = _client(handler)
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    c.complete_json(purpose="ping", system="s", user="u", schema=schema)
    c.complete_json(purpose="ping", system="s", user="u", schema=schema)
    assert c.mode == "tools"
    assert c.complete_json(purpose="ping", system="s", user="u", schema=schema) == {"answer": "ok"}


# ------------------------------------------------------------------ setup checks (fake Ollama server)
def _fake_ollama(monkeypatch, *, models, ps):
    import scripts.check_llm  # noqa: F401  (module under test imports httpx lazily)

    def fake_get(url, timeout=None):
        if url.endswith("/api/version"):
            return httpx.Response(200, json={"version": "0.12.0"})
        if url.endswith("/api/tags"):
            return httpx.Response(200, json={"models": [{"name": m} for m in models]})
        if url.endswith("/api/ps"):
            return httpx.Response(200, json={"models": ps})
        raise AssertionError(url)

    monkeypatch.setattr(httpx, "get", fake_get)


def _settings():
    s = Settings(llm_provider="ollama", _env_file=None)
    apply_provider_defaults(s)
    return s


def test_check_reports_missing_model(monkeypatch, capsys):
    from scripts.check_llm import check_ollama_server

    _fake_ollama(monkeypatch, models=["qwen3.5:9b"], ps=[])
    assert not check_ollama_server(_settings())
    assert "ollama pull gpt-oss:20b" in capsys.readouterr().out


def test_check_flags_short_context_and_cpu_spill(monkeypatch, capsys):
    from scripts.check_llm import check_ollama_loaded

    _fake_ollama(monkeypatch, models=["gpt-oss:20b"],
                 ps=[{"name": "gpt-oss:20b", "size": 14_000_000_000, "size_vram": 11_200_000_000, "context_length": 4096}])
    assert not check_ollama_loaded(_settings())
    out = capsys.readouterr().out
    assert "80% on GPU" in out and "scripts.setup_ollama" in out


def test_check_passes_when_fully_on_gpu_with_enough_context(monkeypatch, capsys):
    from scripts.check_llm import check_ollama_loaded, check_ollama_server

    ps = [{"name": "gpt-oss:20b", "size": 14_000_000_000, "size_vram": 14_000_000_000, "context_length": 16384}]
    _fake_ollama(monkeypatch, models=["gpt-oss:20b"], ps=ps)
    assert check_ollama_server(_settings()) and check_ollama_loaded(_settings())
    assert "100% on GPU, context 16384" in capsys.readouterr().out


def test_setup_creates_a_long_context_variant():
    from scripts.setup_ollama import base_model, create_variant, ensure_pulled, load_and_report, variant_name

    assert variant_name("gpt-oss:20b", 16384) == "gpt-oss:20b-16k"
    assert base_model("gpt-oss:20b-16k") == "gpt-oss:20b" and base_model("claude-sonnet-5-5") == "gpt-oss:20b"
    calls = []

    def handler(request):
        body = json.loads(request.content) if request.content else {}
        calls.append((request.url.path, body))
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "gpt-oss:20b"}]})
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "gpt-oss:20b-16k", "size": 13e9, "size_vram": 13e9,
                                                         "context_length": 16384}]})
        return httpx.Response(200, json={})

    with httpx.Client(base_url="http://localhost:11434", transport=httpx.MockTransport(handler)) as http:
        ensure_pulled(http, "gpt-oss:20b")                         # already downloaded: no pull
        create_variant(http, "gpt-oss:20b", "gpt-oss:20b-16k", 16384)
        m = load_and_report(http, "gpt-oss:20b-16k", "30m")
    create = next(b for p, b in calls if p == "/api/create")
    assert create == {"model": "gpt-oss:20b-16k", "from": "gpt-oss:20b", "parameters": {"num_ctx": 16384},
                      "stream": False}
    assert not any(p == "/api/pull" for p, _ in calls)
    assert m["context_length"] == 16384


# ------------------------------------------------------------------ native Ollama client
def _native(handler, model="gpt-oss:20b"):
    from app.llm.ollama import OllamaClient

    return OllamaClient("http://localhost:11434/v1", model, transport=httpx.MockTransport(handler), sleep=lambda s: None)


def test_native_request_sets_schema_reasoning_context_and_keep_alive():
    from app.agent.planner import PLAN_SCHEMA

    sent = []

    def handler(request):
        sent.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"message": {"content": '{"sub_questions": [], "action": {"type": "finalize", '
                                                                '"args": {}, "reason": "r"}}', "thinking": "short"},
                                         "done_reason": "stop", "eval_count": 40, "eval_duration": 5e8})

    c = _native(handler)
    out = c.complete_json(purpose="plan_step", system="s", user="u", schema=PLAN_SCHEMA)
    path, body = sent[0]
    assert path == "/api/chat" and body["stream"] is False
    assert body["format"] == PLAN_SCHEMA                      # grammar-constrained to the typed schema
    assert body["think"] == "low"                             # gpt-oss: short hidden reasoning (the speed lever)
    assert body["options"]["num_ctx"] == 16384 and body["keep_alive"] == "30m"
    assert out["action"]["type"] == "finalize" and c.last_stats["eval_count"] == 40


def test_other_thinking_models_have_thinking_switched_off():
    from app.llm.ollama import think_setting

    assert think_setting("qwen3.5:9b", None) is False and think_setting("gpt-oss:20b-16k", None) == "low"


def test_reply_in_thinking_field_is_still_used():
    c = _native(lambda r: httpx.Response(200, json={"message": {"content": "", "thinking": '{"answer": "ok"}'}}))
    assert c.complete_json(purpose="p", system="s", user="u", schema={"type": "object"}) == {"answer": "ok"}


def test_missing_model_and_server_down_are_clear_errors():
    import pytest

    from app.llm.base import LLMError

    c = _native(lambda r: httpx.Response(404, json={"error": "model 'gpt-oss:20b' not found"}))
    with pytest.raises(LLMError, match="ollama pull gpt-oss:20b"):
        c.complete_json(purpose="p", system="s", user="u", schema={"type": "object"})

    def down(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError, match="not reachable"):
        _native(down).complete_json(purpose="p", system="s", user="u", schema={"type": "object"})


def test_think_option_dropped_if_the_model_rejects_it():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "think" in body:
            return httpx.Response(400, json={"error": "model does not support thinking"})
        return httpx.Response(200, json={"message": {"content": "{}"}})

    c = _native(handler, model="gemma4:e4b")
    c.reasoning_effort = "on"
    assert c.complete_json(purpose="p", system="s", user="u", schema={"type": "object"}) == {}
    assert "think" in bodies[0] and "think" not in bodies[1]


def test_empty_constrained_reply_is_rescued_by_a_prompt_only_retry():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if isinstance(body["format"], dict):          # the grammar-constrained attempt comes back empty
            return httpx.Response(200, json={"message": {"content": "", "thinking": "Let me think"},
                                             "done_reason": "stop", "prompt_eval_count": 900, "eval_count": 30})
        return httpx.Response(200, json={"message": {"content": 'Here: {"answer": "ok", "n": 2}'},
                                         "prompt_eval_count": 950, "eval_count": 12})

    c = _native(handler)
    out = c.complete_json(purpose="p", system="s", user="u",
                          schema={"type": "object", "properties": {"answer": {"type": "string"}}})
    assert out == {"answer": "ok", "n": 2} and c.fallbacks == 1
    assert bodies[1]["format"] == "json" and "JSON schema" in bodies[1]["messages"][0]["content"]
    assert (c.tokens_in, c.tokens_out) == (1850, 42)                   # both attempts counted
    assert bodies[0]["options"]["temperature"] == 0 and bodies[0]["options"]["seed"] == 7


def test_planner_steps_record_local_token_counts(container, llm, seeded):
    llm.tokens_in = llm.tokens_out = 0

    def plan(user):
        llm.tokens_in += 1200
        llm.tokens_out += 150
        return {"sub_questions": [{"id": "q1", "text": "t", "status": "OPEN", "evidence_ids": []}],
                "action": {"type": "finalize", "args": {}, "reason": "r"}}

    llm.push("plan_step", plan)
    state = container.investigator.run(container.investigator.create("How long does the Acme lease run?").id)
    step = next(s for s in state.steps if s.label == "Concluding the investigation")
    assert (step.tokens_in, step.tokens_out) == (1200, 150)

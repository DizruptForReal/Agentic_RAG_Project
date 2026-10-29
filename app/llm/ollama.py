"""Client for a local Ollama server using its native API (/api/chat), not the OpenAI-compatible one.

Why native: per request it can set what the OpenAI-compatible endpoint cannot -
  * `format`   - the JSON schema; Ollama compiles it into a grammar, so the reply always has the right shape
  * `think`    - reasoning effort ("low" for gpt-oss; off for other thinking models), the main speed lever
  * `options.num_ctx` - the context window (no OLLAMA_CONTEXT_LENGTH / model variant needed)
  * `keep_alive`      - how long the model stays loaded between questions
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable

import httpx

from app.llm.base import LLMError

log = logging.getLogger(__name__)


def _root(base_url: str) -> str:
    return re.sub(r"/v1/?$", "", base_url.rstrip("/"))


def think_setting(model: str, effort: str | None) -> str | bool | None:
    """gpt-oss takes a level (it cannot switch reasoning off); other thinking models take on/off, and off is
    much faster for this app's short structured steps. None = leave the model's default."""
    if "gpt-oss" in model:
        return effort or "low"
    if effort in (None, "", "none", "off", "false"):
        return False
    return True


def extract_json(text: str) -> Any:
    """The first complete JSON object in free text (strips code fences / leading prose). Raises ValueError."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in the reply")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
            elif ch != "\\":
                esc = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unterminated JSON object in the reply")


class OllamaClient:
    mode = "ollama-schema"

    def __init__(self, base_url: str, model: str, *, max_tokens: int = 8192, timeout_s: float = 300.0,
                 num_ctx: int = 16384, keep_alive: str = "30m", reasoning_effort: str | None = None,
                 transport: httpx.BaseTransport | None = None, sleep: Callable[[float], None] = time.sleep):
        self.root = _root(base_url)
        self.base_url = base_url
        self.model, self.max_tokens, self.num_ctx, self.keep_alive = model, max_tokens, num_ctx, keep_alive
        self.reasoning_effort = reasoning_effort
        self.http = httpx.Client(base_url=self.root, timeout=timeout_s, transport=transport)
        self.sleep = sleep
        self.models = [model]
        self.last_model: str | None = None
        self.last_stats: dict[str, Any] = {}
        self.tokens_in = 0           # cumulative prompt / output tokens (the loop records them per step)
        self.tokens_out = 0
        self.fallbacks = 0           # replies rescued by the prompt-only retry

    def list_models(self) -> list[str]:
        r = self.http.get("/api/tags")
        if r.status_code != 200:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
        return sorted(m.get("name", "") for m in r.json().get("models", []))

    def complete_json(self, *, purpose: str, system: str, user: str, schema: dict[str, Any],
                      model: str | None = None) -> Any:
        model = model or self.model
        msg, data = self._chat(model, system, user, fmt=schema)
        content = (msg.get("content") or "").strip() or (msg.get("thinking") or "").strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        # Constrained decoding occasionally comes back empty or cut off with reasoning models (gpt-oss through
        # Ollama). Retry the same request once with the schema in the prompt instead of as a grammar, and take
        # the JSON object out of the reply. Different mechanism, so a different failure mode.
        log.warning("Ollama %s reply for %s was not JSON (done_reason=%s, %s output tokens, %d chars content, "
                    "%d chars thinking); retrying with the schema in the prompt", model, purpose,
                    data.get("done_reason"), data.get("eval_count"), len(msg.get("content") or ""),
                    len(msg.get("thinking") or ""))
        instructions = (f"\n\nReply with ONLY one JSON object (no prose, no code fences) that matches this JSON "
                        f"schema:\n{json.dumps(schema, separators=(',', ':'))}")
        msg2, _ = self._chat(model, system + instructions, user, fmt="json")
        text = (msg2.get("content") or "").strip() or (msg2.get("thinking") or "").strip()
        try:
            out = extract_json(text)
            self.fallbacks += 1
            return out
        except (ValueError, json.JSONDecodeError):
            return text or content      # still malformed: call_structured repairs it

    def _chat(self, model: str, system: str, user: str, *, fmt: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": fmt,
            "stream": False,
            "keep_alive": self.keep_alive,
            # temperature 0 + fixed seed: the same state gives the same plan (reproducible runs and evaluations)
            "options": {"num_ctx": self.num_ctx, "temperature": 0, "seed": 7, "num_predict": self.max_tokens},
        }
        think = think_setting(model, self.reasoning_effort)
        if think is not None:
            body["think"] = think
        data = self._post(body)
        self.last_model = model
        self.last_stats = {k: data.get(k) for k in ("prompt_eval_count", "eval_count", "total_duration",
                                                    "prompt_eval_duration", "eval_duration", "done_reason")}
        self.tokens_in += data.get("prompt_eval_count") or 0
        self.tokens_out += data.get("eval_count") or 0
        if data.get("done_reason") == "length":
            log.warning("Ollama output hit num_predict (%s) and was truncated", self.max_tokens)
        return data.get("message") or {}, data

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(3):
            try:
                r = self.http.post("/api/chat", json=body)
            except httpx.ConnectError as exc:
                raise LLMError(f"Ollama is not reachable at {self.root} - is it running? ({exc})") from exc
            except httpx.TimeoutException as exc:
                raise LLMError(f"Ollama did not answer within {self.http.timeout.read:.0f}s ({exc})") from exc
            except httpx.HTTPError as exc:
                if attempt < 2:
                    self.sleep(2)
                    continue
                raise LLMError(f"network error talking to Ollama: {exc}") from exc
            if r.status_code == 200:
                return r.json()
            text = r.text[:500]
            if r.status_code == 400 and "think" in body and "think" in text.lower():
                log.warning("model %s does not support the think option; sending without it", body["model"])
                body.pop("think")
                continue
            if r.status_code == 404 or "not found" in text.lower():
                raise LLMError(f"LLM configuration error: model '{body['model']}' is not downloaded in Ollama - "
                               f"run: ollama pull {body['model']}  ({text})")
            if r.status_code >= 500 and attempt < 2:
                self.sleep(2)
                continue
            raise LLMError(f"Ollama HTTP {r.status_code}: {text}")
        raise LLMError("Ollama request failed after retries")

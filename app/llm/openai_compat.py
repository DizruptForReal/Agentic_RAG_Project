"""LLMClient for any OpenAI-compatible Chat Completions endpoint.

Works with free / local options as well as paid ones, e.g.
    Groq        https://api.groq.com/openai/v1                           (free tier)
    Gemini      https://generativelanguage.googleapis.com/v1beta/openai  (free tier)
    OpenRouter  https://openrouter.ai/api/v1                             (":free" models)
    Cerebras    https://api.cerebras.ai/v1                               (free tier)
    Ollama      http://localhost:11434/v1                                (local, no key)

Structured output strategy:
  1. "tools" mode: force a single function call whose parameters are the JSON schema.
  2. If the provider rejects tools / tool_choice (HTTP 400), switch permanently to "json" mode:
     response_format=json_object with the schema embedded in the prompt.
Either way the result is validated (and repaired once) by `call_structured` in app/llm/base.py.
429 / 5xx are retried with backoff (honouring Retry-After); anything else becomes LLMError.
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


def _simplify_schema(schema: Any) -> Any:
    """Some providers (e.g. Gemini's compat layer) reject union types like ["string", "null"]."""
    if isinstance(schema, dict):
        out = {}
        for k, v in schema.items():
            if k == "type" and isinstance(v, list):
                non_null = [t for t in v if t != "null"]
                out[k] = non_null[0] if len(non_null) == 1 else non_null
                if "null" in v:
                    out["nullable"] = True
            else:
                out[k] = _simplify_schema(v)
        return out
    if isinstance(schema, list):
        return [_simplify_schema(x) for x in schema]
    return schema


class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str, model: str, *, max_tokens: int = 2048,
                 timeout_s: float = 60.0, mode: str = "auto", max_retries: int = 6,
                 reasoning_effort: str | None = None, fallback_models: list[str] | None = None,
                 switch_after_s: float = 2.0, max_wait_s: float = 65.0,
                 transport: httpx.BaseTransport | None = None, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        if not base_url:
            raise LLMError("LLM_BASE_URL is not set")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_s,
                                 transport=transport)
        self.model, self.max_tokens, self.max_retries, self.sleep = model, max_tokens, max_retries, sleep
        # "strict" = constrained decoding against the JSON schema (always-valid JSON); "tools" = forced
        # function call; "json" = json_object + schema in the prompt.
        self._auto = mode == "auto"
        self.mode = (("strict" if supports_strict(base_url, model) else "tools") if self._auto else mode)
        # Reasoning models (e.g. gpt-oss) spend completion tokens on hidden reasoning; "low" keeps the
        # answer from being truncated and keeps token-per-minute usage down on free tiers.
        if reasoning_effort is None and "gpt-oss" in model:
            reasoning_effort = "low"
        self.reasoning_effort = reasoning_effort or None
        self.clock = clock
        # Waits longer than this are not worth sitting through (e.g. a per-day quota that frees up in
        # minutes): the call fails at once with a clear message instead of blocking the investigation.
        self.max_wait_s = max_wait_s
        # Rate limits on Groq (and most providers) are per model, so a second model has its own budget:
        # when the primary is throttled the client switches instead of waiting.
        self.models = [model] + [m for m in (fallback_models or []) if m and m != model]
        self.switch_after_s = switch_after_s
        self._rate: dict[str, dict[str, float | None]] = {}
        self.last_model: str | None = None
        self.base_url = base_url

    # ------------------------------------------------------------------ public
    def complete_json(self, *, purpose: str, system: str, user: str, schema: dict[str, Any],
                      model: str | None = None) -> Any:
        if self.mode == "strict":
            try:
                return self._strict_call(purpose, system, user, schema, model)
            except _Unsupported as exc:
                if not self._auto:
                    raise LLMError(f"provider rejected strict structured output: {exc}") from exc
                log.warning("provider rejected strict structured output (%s); switching to tool calling", exc)
                self.mode = "tools"
        if self.mode == "tools":
            try:
                return self._tools_call(purpose, system, user, schema, model)
            except _Unsupported as exc:
                if not self._auto:
                    raise LLMError(f"provider rejected tool calling: {exc}") from exc
                log.warning("provider rejected tool calling (%s); switching to JSON mode", exc)
                self.mode = "json"
        return self._json_call(purpose, system, user, schema, model)

    def list_models(self) -> list[str]:
        r = self.http.get("/models")
        if r.status_code != 200:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
        return sorted(m.get("id", "") for m in r.json().get("data", []))

    # ------------------------------------------------------------------ modes
    def _tools_call(self, purpose, system, user, schema, model) -> Any:
        body = {
            "model": model or self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "tools": [{"type": "function", "function": {
                "name": purpose, "description": f"Submit the {purpose.replace('_', ' ')} result.",
                "parameters": _simplify_schema(schema)}}],
            "tool_choice": {"type": "function", "function": {"name": purpose}},
        }
        data = self._post(body, allow_unsupported=True)
        msg = data["choices"][0]["message"]
        for call in msg.get("tool_calls") or []:
            args = call.get("function", {}).get("arguments")
            if isinstance(args, str):
                try:
                    return json.loads(args)
                except json.JSONDecodeError:
                    return args          # malformed: let the validator repair it
            return args
        return msg.get("content") or ""

    def _strict_call(self, purpose, system, user, schema, model) -> Any:
        body = {
            "model": model or self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": purpose, "strict": True, "schema": to_strict_schema(schema)}},
        }
        data = self._post(body, allow_unsupported=True)
        content = data["choices"][0]["message"].get("content") or ""
        try:
            out = restore_from_strict(json.loads(content), schema)
            self._strict_bad = 0
            return out
        except json.JSONDecodeError:
            # Some local servers/models accept the schema but ignore it. In auto mode, two non-JSON replies in a
            # row mean strict mode is not really supported here: use tool calling from now on.
            self._strict_bad = getattr(self, "_strict_bad", 0) + 1
            if self._auto and self._strict_bad >= 2:
                log.warning("strict structured output is being ignored by %s; switching to tool calling", self.model)
                self.mode = "tools"
            return content            # malformed (e.g. truncated): let the validator repair it

    def _json_call(self, purpose, system, user, schema, model) -> Any:
        instructions = (f"\n\nRespond with ONLY a JSON object (no prose, no code fences) that matches this "
                        f"JSON schema for '{purpose}':\n{json.dumps(schema)}")
        body = {
            "model": model or self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "messages": [{"role": "system", "content": system + instructions}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
        }
        data = self._post(body, allow_unsupported=False)
        return data["choices"][0]["message"].get("content") or ""

    # ------------------------------------------------------------------ http
    def _post(self, body: dict[str, Any], *, allow_unsupported: bool) -> dict[str, Any]:
        from app.agent import progress

        primary = body.get("model") or self.model
        strict = body.get("response_format", {}).get("type") == "json_schema"
        for attempt in range(self.max_retries + 1):
            candidates = [primary] + [m for m in self.models if m != primary]
            if strict:
                candidates = [m for m in candidates if m == primary or supports_strict(self.base_url, m)]
            model, delay = self._pick(candidates, body)
            if delay > self.max_wait_s:
                raise LLMError(self._quota_message(candidates, delay))
            if delay > 0:
                self._wait(delay)
            if model != primary:
                progress.report(f"{primary} is rate-limited; continuing with {model}", "wait")
                log.info("rate limit on %s; using fallback model %s", primary, model)
            body["model"] = model
            if self.reasoning_effort and "gpt-oss" in model:
                body["reasoning_effort"] = self.reasoning_effort
            else:
                body.pop("reasoning_effort", None)
            try:
                r = self.http.post("/chat/completions", json=body)
            except httpx.HTTPError as exc:
                if attempt < self.max_retries:
                    self.sleep(min(2 ** attempt, 20))
                    continue
                raise LLMError(f"network error: {exc}") from exc
            self._read_rate_headers(model, r)
            if r.status_code == 200:
                self.last_model = model
                data = r.json()
                if (data.get("choices") or [{}])[0].get("finish_reason") == "length":
                    log.warning("LLM output hit max_tokens (%s) and was truncated", body.get("max_tokens"))
                return data
            text = r.text[:800]
            if (r.status_code == 400 and "reasoning_effort" in body and "reasoning_effort" in text
                    and not _is_tool_generation_failure(text)):
                log.warning("provider rejected reasoning_effort; dropping it")
                self.reasoning_effort = None
                continue
            if r.status_code in (401, 403) or "model_not_found" in text or "does not exist" in text:
                if model != primary and r.status_code not in (401, 403):
                    log.warning("fallback model %s is not available; removing it", model)
                    self.models = [m for m in self.models if m != model]
                    continue
                # Configuration problems: never retried, never treated as "tools unsupported".
                raise LLMError(f"LLM configuration error (check LLM_API_KEY / LLM_MODEL / LLM_BASE_URL): "
                               f"HTTP {r.status_code}: {text}")
            if r.status_code == 400 and _is_tool_generation_failure(text):
                # e.g. Groq "tool_use_failed": the model produced invalid arguments. Treat it as
                # malformed output so the repair prompt can fix it.
                self.last_model = model
                return {"choices": [{"message": {"content": _failed_generation(text)}}]}
            if r.status_code in (400, 404, 422) and allow_unsupported:
                raise _Unsupported(f"HTTP {r.status_code}: {text}")
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                wait = _retry_after(r) or _retry_in_text(text) or min(2 ** attempt * 2, 30)
                if r.status_code == 429:
                    st = self._state(model)
                    st["blocked_until"] = self.clock() + wait
                    st["why"] = _limit_kind(text)
                    log.warning("LLM 429 on %s (retry after %.1fs)", model, wait)
                else:
                    log.warning("LLM HTTP %s, retrying in %.1fs", r.status_code, wait)
                    self._wait(wait)
                continue
            if r.status_code == 429:
                raise LLMError(f"rate limit reached on every configured model ({', '.join(self.models)}); "
                               f"the provider's free-tier quota is used up for now - try again later. {text}")
            raise LLMError(f"HTTP {r.status_code}: {text}")
        raise LLMError("retries exhausted")

    # ------------------------------------------------------------------ rate limiting
    def _state(self, model: str) -> dict[str, Any]:
        return self._rate.setdefault(model, {"remaining": None, "reset_at": 0.0, "blocked_until": 0.0, "why": ""})

    def _quota_message(self, candidates: list[str], delay: float) -> str:
        whys = sorted({self._state(m).get("why") or "requests/tokens per minute" for m in candidates})
        mins = f"about {delay / 60:.0f} min" if delay >= 90 else f"about {delay:.0f} s"
        return (f"rate limit reached on every configured model ({', '.join(candidates)}; limit: {', '.join(whys)}). "
                f"The next model is available again in {mins}; not waiting that long. Try again later, or "
                f"use another provider/key.")

    def _delay(self, model: str, estimate: float) -> float:
        st, now = self._state(model), self.clock()
        delay = max(0.0, (st["blocked_until"] or 0.0) - now)
        if st["remaining"] is not None and estimate > st["remaining"]:
            delay = max(delay, (st["reset_at"] or 0.0) - now + 0.5)
        return round(delay, 1)

    def _pick(self, candidates: list[str], body: dict[str, Any]) -> tuple[str, float]:
        """First model that can run (almost) now; otherwise the one that frees up soonest."""
        estimate = len(json.dumps(body)) / 4 + (body.get("max_tokens") or 0) * 0.5
        delays = [(m, self._delay(m, estimate)) for m in candidates]
        for m, d in delays:
            if d <= self.switch_after_s:
                return m, d
        return min(delays, key=lambda x: x[1])

    def _wait(self, seconds: float) -> None:
        from app.agent import progress

        progress.report(f"Waiting for the model's rate limit ({seconds:.0f}s)", "wait")
        progress.add_wait(seconds)
        self.sleep(seconds)
        log.info("pacing LLM call: waited %.1fs for the rate limit", seconds)

    def _read_rate_headers(self, model: str, r: httpx.Response) -> None:
        remaining = r.headers.get("x-ratelimit-remaining-tokens")
        reset = _parse_duration(r.headers.get("x-ratelimit-reset-tokens"))
        if remaining is not None:
            try:
                st = self._state(model)
                st["remaining"] = float(remaining)
                st["reset_at"] = self.clock() + (reset or 0.0)
            except ValueError:
                return


def _parse_duration(value: str | None) -> float | None:
    """Parse '7.66s', '2m59.56s', '120ms', '1h2m' or a bare number of seconds."""
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    total, found = 0.0, False
    for num, unit in re.findall(r"([\d.]+)(ms|h|m|s)", value):
        found = True
        total += float(num) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total if found else None


def is_ollama(base_url: str) -> bool:
    """A local Ollama server (default port 11434, or a URL that names it)."""
    return ":11434" in base_url or "ollama" in base_url.lower()


def ollama_root(base_url: str) -> str:
    """http://localhost:11434/v1 -> http://localhost:11434 (Ollama's native API lives at the root)."""
    return re.sub(r"/v1/?$", "", base_url.rstrip("/"))


def supports_strict(base_url: str, model: str) -> bool:
    """Strict (constrained) JSON-schema decoding: Groq for these model families; Ollama for every model
    (it compiles the schema into a grammar, so output always parses)."""
    if is_ollama(base_url):
        return True
    return "api.groq.com" in base_url and any(m in model for m in ("gpt-oss", "qwen3"))


def to_strict_schema(schema: Any) -> Any:
    """Make a schema acceptable for strict mode: closed objects, every property required (optional ones
    become nullable), and free-form objects (no properties) encoded as JSON strings."""
    if isinstance(schema, list):
        return [to_strict_schema(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    t = schema.get("type")
    if t == "object" and "properties" not in schema:
        return {"type": "string", "description": "A JSON object encoded as a string, e.g. {\"key\": \"value\"}"}
    out = {k: to_strict_schema(v) for k, v in schema.items() if k not in ("minItems", "maxItems")}
    if t == "object":
        required = set(schema.get("required", []))
        props = {}
        for name, sub in schema["properties"].items():
            sub = to_strict_schema(sub)
            if name not in required:
                sub = _nullable(sub)
            props[name] = sub
        out["properties"] = props
        out["required"] = list(props)
        out["additionalProperties"] = False
    return out


def _nullable(schema: dict) -> dict:
    t = schema.get("type")
    if isinstance(t, list):
        return schema if "null" in t else {**schema, "type": t + ["null"]}
    if t is None:
        return {"anyOf": [schema, {"type": "null"}]}
    out = {**schema, "type": [t, "null"]}
    if "enum" in out:
        out["enum"] = list(out["enum"]) + [None]
    return out


def restore_from_strict(value: Any, schema: Any) -> Any:
    """Undo to_strict_schema: drop nulls for optional fields and decode JSON-string objects."""
    if not isinstance(schema, dict):
        return value
    t = schema.get("type")
    if t == "object" and "properties" not in schema:
        if isinstance(value, str):
            try:
                return json.loads(value) if value.strip() else {}
            except json.JSONDecodeError:
                return value
        return value
    if t == "object" and isinstance(value, dict):
        required = set(schema.get("required", []))
        out = {}
        for k, v in value.items():
            sub = schema["properties"].get(k)
            if v is None and k not in required:
                continue
            out[k] = restore_from_strict(v, sub) if sub else v
        return out
    if t == "array" and isinstance(value, list):
        return [restore_from_strict(v, schema.get("items")) for v in value]
    return value


class _Unsupported(Exception):
    pass


def _retry_after(r: httpx.Response) -> float | None:
    try:
        return float(r.headers.get("retry-after", ""))
    except ValueError:
        return None


def _retry_in_text(text: str) -> float | None:
    """Groq puts the exact wait in the message: 'Please try again in 4m7.536s'."""
    import re

    m = re.search(r"try again in ([0-9hms.]+)", text)
    return _parse_duration(m.group(1).rstrip(".")) if m else None


def _limit_kind(text: str) -> str:
    t = text.lower()
    if "per day" in t or "(tpd)" in t or "(rpd)" in t:
        return "daily quota"
    if "per minute" in t or "(tpm)" in t or "(rpm)" in t:
        return "per-minute quota"
    return "rate limit"


def _is_tool_generation_failure(text: str) -> bool:
    return any(k in text for k in ("tool_use_failed", "failed_generation", "json_validate_failed"))


def _failed_generation(text: str) -> str:
    try:
        err = json.loads(text).get("error", {})
        return err.get("failed_generation") or err.get("message") or text
    except json.JSONDecodeError:
        return text

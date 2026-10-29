"""Check the configured LLM before running investigations.

    python -m scripts.check_llm

Lists the models the provider offers (OpenAI-compatible providers), checks the quota of every configured
model (primary and fallbacks) with a tiny request, then makes one small structured call with the same
mechanism the planner uses, and reports which mode (strict / tools / json) works.

With Ollama it also checks that the server is running, the model is downloaded, the model runs fully on the
GPU and the context window is long enough (OLLAMA_CONTEXT_LENGTH).

Note for hosted providers: limits (including daily token quotas) usually belong to the account or organization,
not to the API key, so a new key in the same account shares the same quota.
"""
from __future__ import annotations

import sys

from pydantic import BaseModel

from app.config import Settings
from app.container import build_llm
from app.llm.base import call_structured


class Ping(BaseModel):
    answer: str
    number: int


SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}, "number": {"type": "integer"}},
          "required": ["answer", "number"]}


def check_quota(llm) -> bool:
    """One tiny request per configured model; print what the provider says about its limits."""
    import re

    from app.llm.openai_compat import _limit_kind, _retry_in_text

    ok_all = True
    print("\nquota per model:")
    for model in llm.models:
        r = llm.http.post("/chat/completions", json={"model": model, "max_tokens": 1,
                                                     "messages": [{"role": "user", "content": "hi"}]})
        h = r.headers
        if r.status_code == 200:
            print(f"  OK   {model}: requests left today {h.get('x-ratelimit-remaining-requests', '?')}"
                  f"/{h.get('x-ratelimit-limit-requests', '?')}, tokens left this minute "
                  f"{h.get('x-ratelimit-remaining-tokens', '?')}/{h.get('x-ratelimit-limit-tokens', '?')}")
        elif r.status_code == 429:
            ok_all = False
            text = r.text
            wait = _retry_in_text(text)
            org = re.search(r"organization `([^`]+)`", text)
            print(f"  BUSY {model}: {_limit_kind(text)}"
                  + (f", free again in about {wait / 60:.0f} min" if wait and wait >= 90 else
                     f", free again in {wait:.0f}s" if wait else "")
                  + (f" (organization {org.group(1)})" if org else ""))
        else:
            ok_all = False
            print(f"  FAIL {model}: HTTP {r.status_code}: {r.text[:200]}")
    return ok_all


MIN_CONTEXT = 12288     # the planner prompt is 4-6k tokens plus the reply; shorter contexts silently drop evidence


def check_ollama_server(s) -> bool:
    """Before the first call: is Ollama running and is the model downloaded?"""
    import httpx

    from app.llm.openai_compat import ollama_root

    root = ollama_root(s.llm_base_url)
    try:
        version = httpx.get(f"{root}/api/version", timeout=5).json().get("version", "?")
    except Exception as exc:  # noqa: BLE001
        print(f"\n!! Ollama is not reachable at {root} ({exc}). Start the Ollama app (or `ollama serve`) and retry.")
        return False
    names = [m.get("name", "") for m in httpx.get(f"{root}/api/tags", timeout=10).json().get("models", [])]
    print(f"\nOllama {version} at {root}; downloaded models: {', '.join(names) or 'none'}")
    wanted = s.llm_model if ":" in s.llm_model else f"{s.llm_model}:latest"
    if wanted not in names:
        print(f"!! model '{s.llm_model}' is not downloaded. Run:  ollama pull {s.llm_model}")
        return False
    return True


def check_ollama_loaded(s) -> bool:
    """After a call: is the model fully on the GPU, and is the context long enough?"""
    import httpx

    from app.llm.openai_compat import ollama_root

    ok = True
    running = httpx.get(f"{ollama_root(s.llm_base_url)}/api/ps", timeout=10).json().get("models", [])
    m = next((x for x in running if x.get("name") == s.llm_model or x.get("model") == s.llm_model), None)
    if not m:
        print("could not see the loaded model in `ollama ps`")
        return False
    size, vram = m.get("size") or 0, m.get("size_vram") or 0
    gpu = round(100 * vram / size) if size else 0
    print(f"loaded: {s.llm_model}, {size / 1e9:.1f} GB, {gpu}% on GPU", end="")
    ctx = m.get("context_length")
    print(f", context {ctx} tokens" if ctx else "")
    if gpu < 100:
        ok = False
        print(f"!! {100 - gpu}% of the model runs on the CPU, which is much slower. Close other GPU-heavy apps, or "
              f"lower the context (OLLAMA_CONTEXT_LENGTH) and restart Ollama.")
    if ctx is None:
        print(f"   (this Ollama version does not report the context; run python -m scripts.setup_ollama to be sure)")
    elif ctx < MIN_CONTEXT:
        ok = False
        print(f"!! context is {ctx} tokens; the app needs at least {MIN_CONTEXT}. Fix:\n"
              f"     python -m scripts.setup_ollama      (creates {s.llm_model}-16k; then set LLM_MODEL to it in .env)")
    return ok


def main() -> None:
    s = Settings()
    print(f"provider={s.llm_provider} model={s.llm_model} base_url={s.llm_base_url or '(anthropic)'}")
    llm = build_llm(s)
    from app.llm.openai_compat import is_ollama

    local = is_ollama(s.llm_base_url)
    if local and not check_ollama_server(s):
        sys.exit(1)
    if hasattr(llm, "list_models"):
        try:
            models = llm.list_models()
            print(f"\n{len(models)} models available:")
            for m in models:
                print(("  * " if m == s.llm_model else "    ") + m)
            if s.llm_model not in models:
                print(f"\n!! LLM_MODEL '{s.llm_model}' is NOT in this list - pick one above and update .env")
        except Exception as exc:  # noqa: BLE001
            print(f"could not list models: {exc}")
    if hasattr(llm, "http") and hasattr(llm, "models") and not local:
        if not check_quota(llm):
            print("\n!! at least one model is rate-limited or failing. Hosted quotas are usually per account/organization:"
                  " a new key in the same organization does not reset it.")
    if local:
        print("\nloading the model and sending a test request (the first load can take a minute)...")
    import time

    t0 = time.monotonic()
    res = call_structured(llm, purpose="ping", system="Reply using the requested structure.",
                          user="What is the capital of France, and what is 2+3?", schema=SCHEMA, model_cls=Ping)
    if res.value is None:
        print(f"\nFAILED structured call: {res.errors}")
        sys.exit(1)
    mode = getattr(llm, "mode", "tools")
    print(f"\nOK structured call ({mode} mode, {res.attempts} attempt(s), {time.monotonic() - t0:.1f}s): "
          f"{res.value.model_dump()}")
    stats = getattr(llm, "last_stats", None)
    if stats and stats.get("eval_count"):
        gen_s = (stats.get("eval_duration") or 0) / 1e9
        pre_s = (stats.get("prompt_eval_duration") or 0) / 1e9
        print(f"   prompt {stats.get('prompt_eval_count')} tokens in {pre_s:.1f}s, output {stats['eval_count']} tokens "
              f"in {gen_s:.1f}s ({stats['eval_count'] / gen_s:.0f} tokens/s)" if gen_s else "")
    if local and not check_ollama_loaded(s):
        sys.exit(1)


if __name__ == "__main__":
    main()

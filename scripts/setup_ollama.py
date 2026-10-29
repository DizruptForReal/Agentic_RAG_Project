"""One-command Ollama setup for this app: make sure the model is downloaded, then create a copy of it with a
long enough context window, so the app no longer depends on the OLLAMA_CONTEXT_LENGTH environment variable.

    python -m scripts.setup_ollama                       # gpt-oss:20b -> gpt-oss:20b-16k
    python -m scripts.setup_ollama --model qwen3.5:9b --ctx 16384

Why: Ollama's OpenAI-compatible endpoint (which the app uses) cannot set the context size per request; it uses
the server default, which is often 4096 tokens and silently cuts the planner's prompt. A model variant with
`num_ctx` baked in always loads with the right context. It shares the weights with the original, so it costs
no extra disk space.

Afterwards set LLM_MODEL=<printed name> in .env and run: python -m scripts.check_llm
"""
from __future__ import annotations

import argparse
import json
import sys

import httpx

from app.config import Settings
from app.container import OLLAMA_DEFAULT_MODEL, OLLAMA_DEFAULT_URL
from app.llm.openai_compat import ollama_root

MIN_CTX = 12288


def base_model(configured: str) -> str:
    """The model to start from: the configured Ollama model without a '-16k' style suffix, else gpt-oss:20b."""
    import re

    if ":" not in configured:            # e.g. the Anthropic default - not an Ollama model name
        return OLLAMA_DEFAULT_MODEL
    return re.sub(r"-\d+k$", "", configured)


def variant_name(model: str, ctx: int) -> str:
    """gpt-oss:20b -> gpt-oss:20b-16k ; llama3 -> llama3:latest-16k"""
    name, _, tag = model.partition(":")
    return f"{name}:{tag or 'latest'}-{ctx // 1024}k"


def ensure_pulled(http: httpx.Client, model: str) -> None:
    names = {m.get("name") for m in http.get("/api/tags").json().get("models", [])}
    if model in names or f"{model}:latest" in names:
        print(f"model {model} is downloaded")
        return
    print(f"downloading {model} (this can take a while) ...")
    with http.stream("POST", "/api/pull", json={"model": model}, timeout=None) as r:
        last = ""
        for line in r.iter_lines():
            if not line:
                continue
            status = json.loads(line).get("status", "")
            if status != last:
                print(f"  {status}")
                last = status
    print(f"model {model} downloaded")


def create_variant(http: httpx.Client, base: str, name: str, ctx: int) -> None:
    r = http.post("/api/create", json={"model": name, "from": base, "parameters": {"num_ctx": ctx}, "stream": False},
                  timeout=600)
    if r.status_code >= 400:   # older Ollama versions take a Modelfile instead
        r = http.post("/api/create", json={"name": name, "modelfile": f"FROM {base}\nPARAMETER num_ctx {ctx}",
                                           "stream": False}, timeout=600)
    r.raise_for_status()
    print(f"created {name} (context {ctx} tokens, same weights as {base})")


def load_and_report(http: httpx.Client, name: str, keep_alive: str) -> dict | None:
    http.post("/api/generate", json={"model": name, "keep_alive": keep_alive}, timeout=600).raise_for_status()
    running = http.get("/api/ps").json().get("models", [])
    m = next((x for x in running if x.get("name") == name or x.get("model") == name), None)
    if m:
        size, vram = m.get("size") or 0, m.get("size_vram") or 0
        gpu = round(100 * vram / size) if size else 0
        print(f"loaded {name}: {size / 1e9:.1f} GB, {gpu}% on GPU, context {m.get('context_length', '?')} tokens")
    return m


def main() -> None:
    s = Settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=base_model(s.llm_model))
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--url", default=s.llm_base_url if ":11434" in s.llm_base_url else OLLAMA_DEFAULT_URL)
    args = ap.parse_args()
    if args.ctx < MIN_CTX:
        sys.exit(f"--ctx must be at least {MIN_CTX} for this app")
    root = ollama_root(args.url)
    with httpx.Client(base_url=root, timeout=30) as http:
        try:
            version = http.get("/api/version").json().get("version", "?")
        except httpx.HTTPError as exc:
            sys.exit(f"Ollama is not reachable at {root} ({exc}). Start Ollama from the Start menu and retry.")
        print(f"Ollama {version} at {root}")
        ensure_pulled(http, args.model)
        name = variant_name(args.model, args.ctx)
        create_variant(http, args.model, name, args.ctx)
        m = load_and_report(http, name, s.ollama_keep_alive)
    ok = m is not None and (m.get("context_length") in (None, args.ctx) or m.get("context_length", 0) >= MIN_CTX)
    print("\nnext: in .env set")
    print("    LLM_PROVIDER=ollama")
    print(f"    LLM_MODEL={name}")
    print("then run:  python -m scripts.check_llm")
    if not ok:
        sys.exit("!! the loaded context is still short - see the output above")


if __name__ == "__main__":
    main()

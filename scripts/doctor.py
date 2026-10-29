"""Setup check: is everything this project needs installed, running and configured?

    python -m scripts.doctor

Read-only - it installs and changes nothing. Every check prints OK, WARN or FAIL, and every problem comes with
the exact fix. Run it after each setup step (see docs/SETUP.md); when it reports no FAIL, the app will work.
Exit code 0 = no failures.
"""
from __future__ import annotations

import importlib.util
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IS_WIN = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
PY_OK = ((3, 11), (3, 13))           # tested range
OLLAMA_MIN = (0, 11)                 # first version with gpt-oss (thinking levels, structured output)
PACKAGES = {"fastapi": "fastapi", "uvicorn": "uvicorn", "pydantic_settings": "pydantic-settings",
            "elasticsearch": "elasticsearch", "sentence_transformers": "sentence-transformers",
            "pymupdf": "pymupdf", "pytesseract": "pytesseract", "PIL": "pillow", "httpx": "httpx",
            "reportlab": "reportlab", "pytest": "pytest"}


@dataclass
class Result:
    status: str                      # OK | WARN | FAIL
    name: str
    detail: str = ""
    fix: list[str] = field(default_factory=list)


def _cmd(*lines: str) -> list[str]:
    return list(lines)


# ------------------------------------------------------------------ checks
def check_python() -> Result:
    v = sys.version_info[:2]
    text = f"Python {platform.python_version()} ({sys.executable})"
    if v < PY_OK[0]:
        return Result("FAIL", "Python", text, ["Install Python 3.12 or 3.13 from https://www.python.org/downloads/ "
                                               "and recreate the virtual environment."])
    if v > PY_OK[1]:
        return Result("WARN", "Python", text + " - newer than tested (3.11-3.13); PyTorch wheels may be missing",
                      ["If pip install fails, use Python 3.12 or 3.13."])
    return Result("OK", "Python", text)


def check_venv() -> Result:
    if sys.prefix != sys.base_prefix:
        return Result("OK", "Virtual environment", sys.prefix)
    act = r".\.venv\Scripts\Activate.ps1" if IS_WIN else "source .venv/bin/activate"
    return Result("WARN", "Virtual environment", "not running inside a virtual environment",
                  _cmd("python -m venv .venv", act, "pip install -r requirements.txt"))


def check_packages() -> Result:
    missing = [pip for mod, pip in PACKAGES.items() if importlib.util.find_spec(mod) is None]
    if missing:
        return Result("FAIL", "Python packages", f"missing: {', '.join(missing)}",
                      _cmd("pip install -r requirements.txt"))
    return Result("OK", "Python packages", "all required packages installed")


def check_env_file() -> Result:
    if (ROOT / ".env").exists():
        return Result("OK", ".env file", str(ROOT / ".env"))
    return Result("FAIL", ".env file", "missing", _cmd("copy .env.example .env" if IS_WIN else "cp .env.example .env"))


def check_docker() -> Result:
    exe = shutil.which("docker")
    if not exe:
        return Result("FAIL", "Docker", "docker command not found",
                      ["Install Docker Desktop: https://www.docker.com/products/docker-desktop/ (start it once, "
                       "accept the terms, wait until it says 'Engine running')."])
    try:
        r = subprocess.run([exe, "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Result("FAIL", "Docker", f"docker did not answer ({exc})", ["Start Docker Desktop and wait ~1 minute."])
    if r.returncode != 0 or not r.stdout.strip():
        return Result("FAIL", "Docker", "Docker is installed but the engine is not running",
                      ["Start Docker Desktop and wait until it shows 'Engine running', then retry."])
    return Result("OK", "Docker", f"engine {r.stdout.strip()} running")


def check_elasticsearch(es_url: str, es_index: str) -> tuple[Result, Result]:
    import httpx

    try:
        info = httpx.get(es_url, timeout=5).json()
    except Exception as exc:  # noqa: BLE001
        return (Result("FAIL", "Elasticsearch", f"not reachable at {es_url} ({type(exc).__name__})",
                       _cmd("docker compose up -d elasticsearch",
                            "wait ~30 s, then: curl.exe http://localhost:9200" if IS_WIN else
                            "wait ~30 s, then: curl http://localhost:9200")),
                Result("WARN", "Corpus", "cannot check until Elasticsearch runs"))
    version = str(info.get("version", {}).get("number", "?"))
    es = (Result("OK", "Elasticsearch", f"{version} at {es_url}") if version.startswith("8.") else
          Result("WARN", "Elasticsearch", f"version {version} at {es_url}; this project is tested with 8.15",
                 ["Use the version in docker-compose.yml: docker compose up -d elasticsearch"]))
    try:
        r = httpx.get(f"{es_url}/{es_index}/_count", timeout=5)
        n = r.json().get("count", 0) if r.status_code == 200 else 0
    except Exception:  # noqa: BLE001
        n = 0
    corpus = (Result("OK", "Corpus", f"{n} chunks indexed in '{es_index}'") if n else
              Result("WARN", "Corpus", "not seeded yet", _cmd("python -m scripts.seed_corpus")))
    return es, corpus


def check_tesseract(tesseract_cmd: str | None) -> Result:
    try:
        import pytesseract
    except ImportError:
        return Result("FAIL", "Tesseract OCR", "pytesseract not installed", _cmd("pip install -r requirements.txt"))
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    try:
        version = str(pytesseract.get_tesseract_version())
    except Exception:  # noqa: BLE001
        if IS_WIN:
            default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
            hint = (f"It is installed at {default}: add this line to .env:  TESSERACT_CMD={default}"
                    if default.exists() and not tesseract_cmd else
                    "Install the 64-bit build from https://github.com/UB-Mannheim/tesseract/wiki, then add to "
                    r".env:  TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe")
        elif IS_MAC:
            hint = "brew install tesseract"
        else:
            hint = "sudo apt install tesseract-ocr"
        where = f" (TESSERACT_CMD={tesseract_cmd})" if tesseract_cmd else ""
        return Result("FAIL", "Tesseract OCR", f"tesseract not found{where}", [hint])
    if not version.startswith(("5", "6")):
        return Result("WARN", "Tesseract OCR", f"version {version}; tested with 5.x", ["Install Tesseract 5."])
    return Result("OK", "Tesseract OCR", f"version {version}")


def _version_tuple(text: str) -> tuple[int, ...]:
    out = []
    for part in text.split(".")[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def check_ollama(base_url: str, model: str) -> tuple[Result, Result]:
    import httpx

    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        version = httpx.get(f"{root}/api/version", timeout=5).json().get("version", "0")
    except Exception as exc:  # noqa: BLE001
        return (Result("FAIL", "Ollama", f"not reachable at {root} ({type(exc).__name__})",
                       ["Install Ollama from https://ollama.com/download and start it "
                        "(Windows/macOS: from the Start menu / Applications; Linux: ollama serve)."]),
                Result("FAIL", "LLM model", "cannot check until Ollama runs", _cmd(f"ollama pull {model}")))
    ok_version = _version_tuple(version) >= OLLAMA_MIN
    olla = (Result("OK", "Ollama", f"version {version} at {root}") if ok_version else
            Result("FAIL", "Ollama", f"version {version} is too old (need {'.'.join(map(str, OLLAMA_MIN))}+)",
                   ["Update Ollama from https://ollama.com/download"]))
    try:
        names = [m.get("name", "") for m in httpx.get(f"{root}/api/tags", timeout=10).json().get("models", [])]
    except Exception:  # noqa: BLE001
        names = []
    wanted = model if ":" in model else f"{model}:latest"
    if wanted in names:
        mdl = Result("OK", "LLM model", f"{model} downloaded (GPU check: python -m scripts.check_llm)")
    else:
        mdl = Result("FAIL", "LLM model", f"{model} not downloaded (have: {', '.join(names) or 'none'})",
                     _cmd(f"ollama pull {model}"))
    return olla, mdl


def check_port(port: int, purpose: str) -> Result:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        in_use = s.connect_ex(("127.0.0.1", port)) == 0
    if not in_use:
        return Result("OK", f"Port {port}", f"free for {purpose}")
    try:
        import httpx

        if httpx.get(f"http://127.0.0.1:{port}/health", timeout=3).status_code == 200:
            return Result("OK", f"Port {port}", "the API is already running here: http://localhost:8000/ui")
    except Exception:  # noqa: BLE001
        pass
    return Result("WARN", f"Port {port}", f"in use by another program (needed for {purpose})",
                  [f"Stop that program, or start the API on another port: uvicorn app.main:app --port 8001"])


def check_disk(path: Path, need_gb: float = 5.0) -> Result:
    free = shutil.disk_usage(path).free / 1e9
    if free < need_gb:
        return Result("WARN", "Disk space", f"{free:.1f} GB free on {path.anchor or path}",
                      [f"Free at least {need_gb:.0f} GB (models and Elasticsearch data)."])
    return Result("OK", "Disk space", f"{free:.0f} GB free")


# ------------------------------------------------------------------ main
def run_checks() -> list[Result]:
    results = [check_python(), check_venv(), check_packages(), check_env_file()]
    if importlib.util.find_spec("pydantic_settings") is None or importlib.util.find_spec("httpx") is None:
        return results                     # settings cannot be read without these packages
    sys.path.insert(0, str(ROOT))
    from app.config import Settings
    from app.container import apply_provider_defaults

    s = Settings()
    apply_provider_defaults(s)
    results.append(check_docker())
    es, corpus = check_elasticsearch(s.es_url, s.es_index)
    if es.status == "OK" and results[-1].status == "FAIL":
        results[-1] = Result("OK", "Docker", "not needed: Elasticsearch is already reachable")
    results += [es, corpus, check_tesseract(s.tesseract_cmd)]
    if s.llm_provider.lower() == "ollama":
        results += list(check_ollama(s.llm_base_url, s.llm_model))
    else:
        results.append(Result("WARN", "LLM", f"LLM_PROVIDER={s.llm_provider} (not the local setup); "
                                            "check it with: python -m scripts.check_llm"))
    results += [check_port(8000, "the API"), check_disk(ROOT)]
    return results


def main() -> None:
    print(f"Setup check for {ROOT}\n")
    results = run_checks()
    for r in results:
        print(f"[{r.status:4}] {r.name:20} {r.detail}")
        for line in r.fix:
            print(f"       fix: {line}")
    fails = [r for r in results if r.status == "FAIL"]
    warns = [r for r in results if r.status == "WARN"]
    print()
    if fails:
        print(f"{len(fails)} problem(s) to fix (see 'fix:' lines above), then run this check again.")
        sys.exit(1)
    nxt = "python -m scripts.seed_corpus, then " if any(r.name == "Corpus" and r.status != "OK" for r in results) else ""
    print(f"Everything required is in place{f' ({len(warns)} warning(s))' if warns else ''}. "
          f"Next: {nxt}uvicorn app.main:app --port 8000  ->  http://localhost:8000/ui")


if __name__ == "__main__":
    main()

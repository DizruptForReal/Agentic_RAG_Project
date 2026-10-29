# Setup guide

Everything needed to run the project on your own machine, step by step. Each step ends with a **Verify** command
and the output you should see; do not continue until it matches. At any point, `python -m scripts.doctor` checks
everything at once and prints the fix for anything missing.

- **Time:** about 30-60 minutes the first time, mostly downloads (~20 GB in total).
- **Cost / accounts:** none. Everything is free; no sign-up or API key is needed. After setup the system runs offline.
- **Commands** are for **Windows PowerShell**. macOS / Linux differences are given under each step.

## 1. Requirements

### Software

| # | Software | Version | Why it is needed | Download |
|---|---|---|---|---|
| 1 | **Git** | 2.x | Clone the repository | https://git-scm.com/downloads |
| 2 | **Python** | **3.11 - 3.13** (3.13 tested) | The application | https://www.python.org/downloads/ |
| 3 | **Docker Desktop** | 4.x | Runs Elasticsearch 8.15 in a container | https://www.docker.com/products/docker-desktop/ |
| 4 | **Tesseract OCR** | 5.x | Reads scanned pages and receipt images | Windows: https://github.com/UB-Mannheim/tesseract/wiki · macOS: `brew install tesseract` · Linux: `sudo apt install tesseract-ocr` |
| 5 | **Ollama** | 0.11 or newer (latest recommended) | Runs the language model locally | https://ollama.com/download |
| 6 | **gpt-oss:20b** model | - | The language model (~14 GB) | `ollama pull gpt-oss:20b` (step 6) |

Python packages (FastAPI, the Elasticsearch client, sentence-transformers / PyTorch, PyMuPDF, pytesseract, ...)
are installed by `pip install -r requirements.txt` in step 7. The embedding model and reranker (~1.5 GB, from
Hugging Face) download automatically the first time the corpus is seeded.

### Hardware

| | Minimum | Recommended |
|---|---|---|
| **GPU** (for the language model) | **16 GB of GPU memory**: NVIDIA RTX 4060 Ti 16 GB / 4070 Ti Super / 4080 / 4090 / 5070 Ti / 5080 / 5090 / 3090, or AMD Radeon RX 6800 / 6900 / 7800 XT / 7900 series, or an Apple Silicon Mac (M1-M4) with 32 GB unified memory | Same; close other GPU-heavy programs while testing |
| RAM | 16 GB | 32 GB |
| Free disk | 25 GB | 40 GB |
| OS | Windows 10 (22H2) or 11, 64-bit · macOS 13+ · Ubuntu 24.04+ (its `apt` provides Tesseract 5; 22.04 only has 4.x) | - |

With less than 16 GB of GPU memory, part of the model runs on the CPU and every step becomes very slow (minutes
instead of seconds). The embedding model and reranker run on the CPU and need no GPU.

### Other

| | |
|---|---|
| Internet | Only during setup (packages, Docker image, models) |
| Free ports | 9200 (Elasticsearch), 8000 (the app), 11434 (Ollama) |
| Admin rights | Needed to install Docker Desktop and Tesseract (normal for installers) |
| Docker Desktop licence | Free for personal use, education and small companies (under 250 employees and under $10M revenue); larger organisations need a paid Docker subscription - see [Running Elasticsearch without Docker Desktop](#running-elasticsearch-without-docker-desktop) |

## 2. Step by step

### Step 1 - Git

Install from https://git-scm.com/downloads (defaults are fine).

**Verify** (in a new PowerShell window):
```powershell
git --version
```
Expected: `git version 2.x.x`

### Step 2 - Python

Install Python **3.12 or 3.13** from https://www.python.org/downloads/. On Windows, tick
**"Add python.exe to PATH"** on the first installer screen.

**Verify:**
```powershell
python --version
```
Expected: `Python 3.12.x` or `Python 3.13.x`

> If `python` opens the Microsoft Store instead: Settings → Apps → Advanced app settings → App execution aliases →
> turn off both "python" aliases, then open a new PowerShell window.

### Step 3 - Docker Desktop

1. Install from https://www.docker.com/products/docker-desktop/ (Windows: keep **"Use WSL 2"** selected).
2. Restart if asked, start **Docker Desktop**, accept the terms (no account needed - "Skip" sign-in).
3. Wait until the bottom-left of the Docker Desktop window shows **Engine running**.

**Verify:**
```powershell
docker info --format "{{.ServerVersion}}"
docker compose version
```
Expected: a version number such as `27.x.x`, then `Docker Compose version v2.x.x`

> Windows: if Docker says WSL needs updating, run `wsl --update` and restart Docker Desktop. If it says
> virtualization is disabled, enable "Intel VT-x" / "AMD-V" (SVM) in the BIOS/UEFI settings.
>
> Linux: install Docker Engine and the compose plugin (https://docs.docker.com/engine/install/), then
> `sudo usermod -aG docker $USER` and log out/in.

### Step 4 - Tesseract OCR

- **Windows:** download the 64-bit installer from https://github.com/UB-Mannheim/tesseract/wiki and install with
  the defaults (location `C:\Program Files\Tesseract-OCR`).
- **macOS:** `brew install tesseract` (Homebrew: https://brew.sh)
- **Linux:** `sudo apt install tesseract-ocr`

**Verify:**
```powershell
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --version     # macOS/Linux: tesseract --version
```
Expected: first line `tesseract v5.x.x`

### Step 5 - Ollama

Install from https://ollama.com/download and start it (Windows/macOS: it runs in the background - look for the
llama icon in the system tray / menu bar; Linux: the installer starts a service).

**Verify:**
```powershell
ollama --version
```
Expected: `ollama version is 0.x.x` - 0.11 or newer

> Optional, before step 6: to store the ~14 GB model on another drive, run
> `setx OLLAMA_MODELS "D:\Ollama\models"`, then quit Ollama from the tray and start it again.

### Step 6 - Download the language model

```powershell
ollama pull gpt-oss:20b
```
This downloads ~14 GB once.

**Verify** that it runs on the GPU:
```powershell
ollama run gpt-oss:20b "Say hello in five words"
ollama ps
```
Expected: a short reply, then a line like
`gpt-oss:20b   ...   13 GB   100% GPU   ...`. **PROCESSOR must say `100% GPU`**; if it shows a CPU share, see
[Troubleshooting](#4-troubleshooting).

### Step 7 - Get the project and install its packages

```powershell
git clone https://github.com/DizruptForReal/Agentic_RAG_Project.git
cd Agentic_RAG_Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
- macOS/Linux: `source .venv/bin/activate` instead of the `Activate.ps1` line.
- Linux only: run `pip install torch --index-url https://download.pytorch.org/whl/cpu` before
  `pip install -r requirements.txt` (avoids a ~3 GB GPU build of PyTorch that is not needed).
- If PowerShell refuses to run `Activate.ps1`: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry.

The prompt now starts with `(.venv)`. **Every later command runs in this activated window**, from the project folder.

**Verify:**
```powershell
python -c "import fastapi, elasticsearch, sentence_transformers, pytesseract, pymupdf; print('packages ok')"
```
Expected: `packages ok`

### Step 8 - Configure

```powershell
copy .env.example .env          # macOS/Linux: cp .env.example .env
notepad .env                    # macOS/Linux: any text editor
```
`.env.example` is already set up for the local model (`LLM_PROVIDER=ollama`, `LLM_MODEL=gpt-oss:20b`).
On **Windows**, remove the `#` in front of this line so the app finds Tesseract:
```
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```
(macOS/Linux: nothing to change - Tesseract is on the PATH.)

### Step 9 - Start Elasticsearch

```powershell
docker compose up -d elasticsearch
```
The first time this downloads the image (~1 GB). Elasticsearch needs ~30 seconds to start.

**Verify:**
```powershell
curl.exe http://localhost:9200          # macOS/Linux: curl http://localhost:9200
```
Expected: a JSON reply containing `"number" : "8.15.3"`. (`curl.exe`, not `curl`, in Windows PowerShell.)

### Step 10 - Check the whole setup

```powershell
python -m scripts.doctor
```
Expected: `[OK  ]` for Python, packages, .env, Docker, Elasticsearch, Tesseract OCR, Ollama, LLM model, port 8000
and disk space; `Corpus` shows `[WARN] not seeded yet` (done in the next step); last line:
`Everything required is in place ...`. Any `[FAIL]` line is followed by `fix:` lines with the exact command.

### Step 11 - Load the document corpus

```powershell
python -m scripts.seed_corpus
```
The first run downloads the embedding model and reranker (~1.5 GB) and takes a few minutes. Expected: one line per
document, each with status `INDEXED`, e.g.
```
warranty_policy_v1.pdf           d_...                INDEXED   4(-)
warranty_amendment_2025.pdf      d_...                INDEXED   2(-)
service_manual_x200.pdf          d_...                INDEXED   3(-)
customer_contract_acme.pdf       d_...                INDEXED   3([3])
service_manual_x300.pdf          d_...                INDEXED   1(-)
chunks in index: ...
```
(`3([3])` = page 3 of the contract was OCR'd, because it is a scanned page.)

### Step 12 - Check the language model end to end

```powershell
python -m scripts.check_llm
```
Expected (after up to a minute while the model loads):
```
Ollama 0.x.x at http://localhost:11434; downloaded models: gpt-oss:20b
...
OK structured call (ollama-schema mode, 1 attempt(s), ...s): {'answer': 'Paris', 'number': 5}   <- wording may differ
...
loaded: gpt-oss:20b, 13.x GB, 100% on GPU, context 16384 tokens
```

### Step 13 - Start the app

```powershell
uvicorn app.main:app --port 8000
```
Expected: `Uvicorn running on http://127.0.0.1:8000`. Leave this window open. Open **http://localhost:8000/ui** in
a browser (give it ~30 s after start while the models load) and ask *"How long does the Acme lease run?"* - the
answer is **36 months**, cited to the contract.

Health check: http://localhost:8000/health shows `"elasticsearch": {"ok": true, ...}`. API docs:
http://localhost:8000/docs.

### Step 14 - Optional: demo and tests

In a **second** PowerShell window (project folder, `.\.venv\Scripts\Activate.ps1` first):
```powershell
python -m scripts.demo                       # main question with the smudged receipt -> explicit refusal
python -m scripts.demo --receipt nonhinge    # main question with a clear receipt     -> cited answer
python -m pytest                             # 156 automated tests (no services needed)
```
Expected for pytest: `155 passed, 1 skipped` while Elasticsearch is stopped - the skipped test needs Elasticsearch
and passes too when it is running.

Manual test cases for every behaviour: [AgenticRAG_test_run.xlsx](AgenticRAG_test_run.xlsx) and
[behaviour_test_pack.md](behaviour_test_pack.md).

## 3. Everyday use

| To | Do |
|---|---|
| Start | Start Docker Desktop and Ollama (if not already running) → `docker compose up -d elasticsearch` → activate the venv → `uvicorn app.main:app --port 8000` |
| Stop | `Ctrl+C` in the uvicorn window → `docker compose stop` |
| Clear past investigations and uploads (keeps the corpus) | Stop the app → `python -m scripts.reset_data --history --yes` |
| Start completely fresh | Stop the app → `python -m scripts.reset_data --yes` → `python -m scripts.seed_corpus` |
| Remove everything | `docker compose down` · delete the project folder · `ollama rm gpt-oss:20b` |

## 4. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `python` not found / opens the Microsoft Store | Reinstall Python with "Add python.exe to PATH", or turn off the App execution aliases (step 2) |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `pip install` fails on PyTorch | Use Python 3.12 or 3.13 (newer versions may not have PyTorch builds yet) |
| `docker: command not found` / `Cannot connect to the Docker daemon` | Start Docker Desktop and wait for *Engine running* |
| `curl http://localhost:9200` fails right after `docker compose up` | Wait 30-60 s; check `docker compose logs elasticsearch` |
| Port 9200 or 8000 already in use | Stop the other program, or use another app port: `uvicorn app.main:app --port 8001` |
| `tesseract is not installed or it's not in your PATH` | Windows: uncomment `TESSERACT_CMD=...` in `.env` (step 8) and check the path exists |
| `ollama ps` shows a CPU share (e.g. `30%/70% CPU/GPU`) | Not enough free GPU memory: close games/video tools, update the GPU driver (NVIDIA / AMD Adrenalin), restart Ollama. A GPU under 16 GB cannot hold the whole model |
| Each step takes minutes | Same as above - the model is running (partly) on the CPU |
| Answer says `TOOL_FAILURE` - "Language model: ... not reachable" | Ollama is not running: start it |
| Answer says `TOOL_FAILURE` - "Search: Elasticsearch could not be reached" | `docker compose up -d elasticsearch` (Docker Desktop must be running) |
| Upload shows `DUPLICATE` | That exact file was uploaded before - expected; `python -m scripts.reset_data --history --yes` for a clean run |
| Old answers after changing sample documents | `python -m scripts.reset_data --yes`, then `python -m scripts.seed_corpus` |

Still stuck: run `python -m scripts.doctor` and follow the `fix:` lines.

## Running Elasticsearch without Docker Desktop

The app only needs an Elasticsearch 8.x server at `ES_URL` (default `http://localhost:9200`). Alternatives to
Docker Desktop, all working with the same `docker-compose.yml` or none at all:

- **Docker Engine** on Linux, or inside WSL 2 on Windows: `docker compose up -d elasticsearch` as above.
- **Podman Desktop** or **Rancher Desktop**: `podman compose up -d elasticsearch` / `nerdctl compose up -d elasticsearch`.
- **No containers:** download Elasticsearch 8.15 from https://www.elastic.co/downloads/past-releases (it includes
  Java), unzip, add these two lines to `config/elasticsearch.yml` **before the first start**:
  ```
  discovery.type: single-node
  xpack.security.enabled: false
  ```
  then run `bin\elasticsearch.bat` (macOS/Linux: `bin/elasticsearch`) and keep that window open.

# Agentic RAG Project - Evidence Investigator

An AI agent that **investigates** a question instead of just answering it. It searches an Elasticsearch knowledge
base, reads document metadata from SQLite, OCRs documents the user uploads mid-investigation, and decides at every
step what evidence it still needs. It ends with a **cited answer** or an **explicit reason** why a reliable answer is
not possible yet: missing information, insufficient evidence, an unresolved conflict, or a tool failure.

The core principle: **the LLM proposes, code decides.** The model plans and labels evidence; deterministic code
validates every claim against an evidence ledger, checks OCR confidence word by word, and decides the outcome.

| Document | What it covers |
|---|---|
| [docs/SETUP.md](docs/SETUP.md) | **Setup guide**: requirements, step-by-step installation with a check after each step, troubleshooting |
| [docs/requirements_mapping.md](docs/requirements_mapping.md) | Every requirement of the assignment → how it is implemented → how to verify it |
| [docs/corpus.md](docs/corpus.md) | The domain, every sample document, the built-in difficulties and the ground truth |
| [docs/system_design.md](docs/system_design.md) | Architecture diagram, investigation loop, reliability, scaling |
| [docs/api_examples.md](docs/api_examples.md) | curl examples for every endpoint |
| [docs/behaviour_test_pack.md](docs/behaviour_test_pack.md) · [docs/AgenticRAG_test_run.xlsx](docs/AgenticRAG_test_run.xlsx) | Manual test cases for every answer behaviour, with upload files |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Decision log: what failed during testing and how each failure class was fixed |

---

## Contents

1. [What it does](#what-it-does) · 2. [Technical requirements → implementation](#technical-requirements--implementation)
· 3. [Domain and corpus](#domain-and-corpus) · 4. [What you need](#what-you-need) · 5. [Setup](#setup)
· 6. [Using it](#using-it) · 7. [Tests and evaluation](#tests-and-evaluation) · 8. [Project layout](#project-layout)
· 9. [Configuration](#configuration) · 10. [Troubleshooting](#troubleshooting) · 11. [Known limitations](#known-limitations)

## What it does

For a question such as *"Is the motor seizure of Acme Facilities' Aquaflow X200 pump (serial AQ-X200-55871) on 14
March 2026 covered under warranty?"* the agent:

1. splits it into sub-questions (warranty term, usage class, failure type, purchase date) and searches;
2. finds the policy (24 months) and its amendment (12 months for commercial units), reads their metadata and
   applies the amendment;
3. verifies the unit's commercial classification on a **scanned** contract page (OCR only on that page);
4. finds that **no document contains the purchase date** and asks for the receipt;
5. OCRs the uploaded receipt and checks the date's confidence word by word;
6. answers with citations (*covered* / *not covered*), or refuses with the reason if the decisive digit of the date
   is smudged.

Outcomes (decided in code): `ANSWERED`, `ANSWERED_WITH_CAVEATS`, `MISSING_INFO` (asks the user), `INSUFFICIENT_EVIDENCE`,
`CONFLICT`, `TOOL_FAILURE`.

## Technical requirements → implementation

| Requirement | Implementation |
|---|---|
| **FastAPI** backend | FastAPI + Uvicorn (`app/main.py`, `app/api/`); Swagger at `/docs`; a single-page web UI at `/ui` to watch investigations and verify citations |
| **Elasticsearch** for chunk indexing and retrieval | Elasticsearch 8.15 in Docker; BM25 over text/section/title + dense kNN; reciprocal-rank fusion with the reranker (`app/retrieval/`) |
| **SQLite** for metadata and processing state | Documents, pages (with per-word OCR confidence), chunks, and the full investigation state saved after every step (`app/db/`) |
| **Embedding model** | `BAAI/bge-small-en-v1.5` (384-d, CPU) via sentence-transformers |
| Reranker (optional in the brief) | `BAAI/bge-reranker-base` |
| **LLM** | **`gpt-oss:20b` running locally in Ollama** - no API key, nothing leaves the machine. Called through Ollama's native API with schema-constrained JSON output (`app/llm/ollama.py`); the client layer is pluggable (`app/llm/`) |
| **OCR** | Tesseract 5 (`pytesseract`, per-word confidence); PyMuPDF for PDF text layers and rendering |
| **Automated tests** for workflow and failure cases | 156 pytest tests with fakes for every external service: end-to-end workflow over HTTP, malformed model output, tool failures, repeated calls, duplicate uploads, crash-and-resume, OCR uncertainty (`tests/`) |

Not built, as the brief allows: no model training. A small optional frontend is included only for visibility.
The full requirement-by-requirement breakdown, with the tests that prove each point, is in
[docs/requirements_mapping.md](docs/requirements_mapping.md).

## Domain and corpus

**Domain: warranty claims for a (fictional) industrial pump manufacturer, Aquaflow Pumps Ltd.** A warranty decision
naturally needs several documents, has versioned rules, depends on a scanned usage declaration and on a date that
only the customer's receipt contains - exactly the situations the assignment asks the agent to handle.

| Document (`samples/`) | Pages | Role in the investigation |
|---|---|---|
| `warranty_policy_v1.pdf` | 4 | The rules: covered products, **24-month** period, what is covered, exclusions, conditions, claims procedure, key time limits |
| `warranty_amendment_2025.pdf` | 2 | Supersedes the period for units bought from 2025: **12 months** for commercial use, labour for 6 months → a **contradiction** resolved by versioning |
| `service_manual_x200.pdf` | 3 | Specifications, installation, fluids, maintenance, **motor seizure is a failure**, error codes; purchase date must come from the receipt |
| `customer_contract_acme.pdf` | 3 | Acme's lease of serial AQ-X200-55871; page 3 is a **scanned** usage declaration (**COMMERCIAL**) with a smudged signature; the purchase date is **absent** |
| `service_manual_x300.pdf` | 1 | A different product's 36-month Extended Care - a **decoy** for X200 questions |
| `receipt_nonhinge.png`, `receipt_hinge.png` | 1 each | Uploaded during the demo: a clear date (05-03-2025) and a date whose decisive digit is smudged |

Plus 11 upload files for the manual tests (`samples/test_uploads/`) and two question sets (42 held-out, 23
regression). Every document, difficulty and expected verdict is described in [docs/corpus.md](docs/corpus.md).
All files are generated by `python -m samples.make_samples` / `python -m samples.make_test_uploads`.

## What you need

The complete system runs on your own machine: free, no account or API key, and offline once set up.
**Full step-by-step instructions, with a check after every step: [docs/SETUP.md](docs/SETUP.md).**

| Requirement | Version / size | Why |
|---|---|---|
| **Git** | 2.x | Clone the repository |
| **Python** | 3.11 - 3.13 (3.13 tested) | The application |
| **Docker Desktop** | 4.x (Windows needs WSL 2 + virtualization enabled) | Runs Elasticsearch 8.15 (~1 GB image) |
| **Tesseract OCR** | 5.x | OCR for scanned pages and receipts |
| **Ollama** | 0.11+ (latest recommended) | Runs the language model locally |
| **gpt-oss:20b** model | ~14 GB download | The language model (`ollama pull gpt-oss:20b`) |
| **GPU** | **16 GB of GPU memory** (e.g. RTX 4060 Ti 16 GB / 4080 / 4090 / 5080, RX 7800 XT / 7900 series) or Apple Silicon with 32 GB | The model must fit entirely in GPU memory; otherwise every step takes minutes |
| RAM / disk | 16 GB RAM · 25 GB free disk | Models, Elasticsearch, Python packages |
| Internet | During setup only | Packages, Docker image, models |
| Free ports | 9200, 8000, 11434 | Elasticsearch, the app, Ollama |

Python packages, the embedding model and the reranker (~1.5 GB) are installed by the steps below; nothing else
is needed. Docker Desktop is free for personal use and small companies; [docs/SETUP.md](docs/SETUP.md#running-elasticsearch-without-docker-desktop)
lists alternatives (Docker Engine, Podman, or Elasticsearch without containers).

## Setup

Quick version for people who already have the requirements installed (Windows PowerShell; macOS/Linux
differences and expected outputs are in [docs/SETUP.md](docs/SETUP.md)):

```powershell
ollama pull gpt-oss:20b                                   # once, ~14 GB

git clone https://github.com/DizruptForReal/Agentic_RAG_Project.git
cd Agentic_RAG_Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1                              # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt                           # Linux: install CPU PyTorch first, see SETUP.md
copy .env.example .env                                    # Windows: uncomment TESSERACT_CMD in .env

docker compose up -d elasticsearch                        # Docker Desktop must be running; ~30 s to start
python -m scripts.doctor                                  # checks everything; prints a fix for any problem
python -m scripts.seed_corpus                             # indexes the corpus (first run downloads ~1.5 GB of models)
python -m scripts.check_llm                               # model answers, 100% on GPU, 16k context
uvicorn app.main:app --port 8000                          # then open http://localhost:8000/ui
```

Run the demonstration in a second terminal (venv activated):

```powershell
python -m scripts.demo                       # smudged receipt -> explicit refusal (INSUFFICIENT_EVIDENCE)
python -m scripts.demo --receipt nonhinge    # clear receipt   -> cited answer (not covered)
```

Or ask the main question in the UI and upload a receipt from `samples/` when the agent asks for it.

## Using it

**Web UI** (`/ui`): ask in the centre box (optionally attach a document). A live status line shows what the agent
is doing ("Searching ...", "Scanning receipt.png with OCR", "Thinking"); each step appears in a timeline with its
timing. When the agent needs something, a reply box appears (type and/or upload). Every `e#` citation opens a side
panel with the page text - cited span highlighted, low-confidence OCR words marked - and the original page.
*How the agent investigated* shows steps, model calls and timings; tabs show sub-question statuses and the full
evidence ledger. The sidebar has the history and the knowledge base.

**API**: Swagger at `/docs`; curl examples in [docs/api_examples.md](docs/api_examples.md).
`POST /investigations` (question, optional file) → `GET /investigations/{id}` → `POST /investigations/{id}/continue`
(reply and/or file).

**Questions to try** (full list with expected outcomes: [docs/AgenticRAG_test_run.xlsx](docs/AgenticRAG_test_run.xlsx)):

| Question | Expected |
|---|---|
| How long does the Acme lease run? | 36 months, cited to the contract |
| Is damage from a lightning strike covered? | No - an exclusion (policy §4) |
| Compare the specifications of the X100, X200 and X300 in a table | X200 and X300 table; caveat: no X100 data |
| Does a pump bought in 2024 fall under the amendment? | No - only units bought from 1 Jan 2025 |
| The main X200 question (above) | Asks for the receipt, then answers or refuses depending on the receipt |
| What is the warranty on the Aquaflow X900? | Refused: no document covers the X900 |
| What will the weather be in Leeds tomorrow? | Refused: outside the documents (never asks the user) |

**Reset between test runs** (keeps the corpus): stop the API, then
`python -m scripts.reset_data --history --yes`. Wipe everything: `python -m scripts.reset_data --yes`, then re-seed.

## Tests and evaluation

```powershell
python -m pytest
```

156 tests, no Elasticsearch, models or API key needed - external services are replaced by fakes
(`tests/fakes.py`). Optional real-system tests run automatically when available: Tesseract tests when `tesseract`
is installed, `tests/test_es_integration.py` when Elasticsearch answers at `ES_URL`. The GitHub Actions workflow
(`.github/workflows/tests.yml`) runs the suite, including the Elasticsearch and Tesseract tests, on every push.

Evaluation against the real stack (`samples/heldout_questions.json`, never used for tuning):

```powershell
python -m scripts.eval_retrieval              # retrieval only (no LLM): recall@k and MRR per variant and category
python -m scripts.eval_answers --limit 15     # whole agent: right outcome and required facts per category
python -m scripts.compare_models              # side by side, when run with more than one model
```

Manual behaviour tests - one per outcome and per document-handling case, with upload files:
[docs/behaviour_test_pack.md](docs/behaviour_test_pack.md) and the run sheet
[docs/AgenticRAG_test_run.xlsx](docs/AgenticRAG_test_run.xlsx).

## Project layout

```
app/
  main.py              FastAPI app, /ui, /health, startup recovery
  container.py         wiring of real (or fake) components; provider defaults
  config.py            all settings (overridable in .env)
  api/                 endpoints: investigations, documents
  agent/               loop, planner prompt, guards (outcome rules, answer review), ledger, answer composer
  tools/               search, get_metadata, process_document, ask_user + registry (validation, retries)
  retrieval/           Elasticsearch index, BGE embedder and reranker, hybrid retriever
  ingestion/           triage, Tesseract OCR, chunker, field extractor, word-level OCR verification, state machine
  llm/                 LLM interface + Ollama client (default); OpenAI-compatible and Anthropic clients (optional)
  db/                  SQLite schema and repository
  static/index.html    web UI
samples/               corpus, test uploads, question sets, generators
scripts/               seed, demo, doctor (setup check), check_llm, evaluation, reset
tests/                 fakes, scripted scenario, workflow and failure tests
docs/                  requirements mapping, corpus, system design, API examples, test pack, decisions
```

## Configuration

All settings are in `app/config.py` and can be overridden in `.env`. Most relevant:

| Setting | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER`, `LLM_MODEL` | `ollama`, `gpt-oss:20b` | The LLM (another pulled Ollama model works the same way, e.g. `qwen3.5:9b`) |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | Where Ollama listens (change if it runs on another machine) |
| `OLLAMA_NUM_CTX`, `OLLAMA_KEEP_ALIVE` | 16384, 30m | Ollama context window and keep-alive (sent per request) |
| `LLM_TIMEOUT_S` | 300 | Per model call (the first call also loads the model) |
| `OCR_CONF_THRESHOLD` | 80 | OCR words below this confidence are uncertain |
| `RELEVANCE_FLOOR` | 0.03 | Reranker score for a passage to count as relevant (calibrated with `scripts/eval_retrieval`) |
| `MAX_STEPS` | 12 | Planner step budget per run |
| `ES_URL`, `ES_INDEX` | `http://localhost:9200`, `chunks` | Elasticsearch |
| `DATA_DIR` | `data` | SQLite database and uploaded files |
| `TESSERACT_CMD` | - | Path to the Tesseract binary if not on PATH |
| `HF_HOME` (environment variable) | user cache | Where the BGE models are downloaded |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/health` shows `elasticsearch.ok: false`; answers are TOOL_FAILURE | Start Docker Desktop, `docker compose up -d elasticsearch`, wait ~30 s |
| `tesseract is not installed or it's not in your PATH` | Set `TESSERACT_CMD` in `.env` to the full path of the Tesseract binary |
| `check_llm`: Ollama not reachable | Start Ollama (Start menu / `ollama serve`) and retry |
| `check_llm`: model not downloaded | `ollama pull gpt-oss:20b` (or the model named in `LLM_MODEL`) |
| Answers are very slow (minutes) | `ollama ps` should show `100% GPU`; anything less means part of the model runs on the CPU - close GPU-heavy apps, update the GPU driver |
| Answer says "Language model: ... not reachable" (TOOL_FAILURE) | Ollama stopped; start it again |
| First question is slow | The BGE models and the Ollama model load on first use; the API preloads them at startup (~30 s) |
| Old answers cite outdated text after changing samples | `python -m scripts.reset_data --yes`, then `python -m scripts.seed_corpus` |
| Uploads show DUPLICATE in a new test run | Expected for the same file; `python -m scripts.reset_data --history --yes` starts clean |

## Known limitations

- Thresholds (OCR confidence 80, relevance floor, off-topic score) are calibrated on this corpus; re-check them with
  `python -m scripts.eval_retrieval` on a new one.
- The unknown-model check recognises code-like identifiers (X900, AQ-...), not unknown plain names.
- Answer quality depends on the LLM. Code-level guards stop unsupported, invented or uncertain answers, but a
  smaller model asks or refuses more often than a larger one would; `gpt-oss:20b` is the tested setup. Expect a few
  seconds per model call and roughly 20 s to 2 minutes per investigation on a 16 GB GPU.
- OCR confidences can differ slightly between Tesseract versions; the sample generators self-check the behaviour
  the tests rely on (`python -m samples.make_samples`).
- Single-node demo setup: Elasticsearch security is disabled, and there is no authentication. See
  `docs/system_design.md` for how this scales to concurrent users and larger collections.

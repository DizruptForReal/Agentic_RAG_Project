# How each requirement is implemented

Every requirement from the assignment, where it lives in the code, and how to check it. "Test" names are in
`tests/` (run `python -m pytest`); "Manual" refers to the cases in `docs/AgenticRAG_test_run.xlsx` /
`docs/behaviour_test_pack.md`.

The design principle throughout: **the LLM proposes, code decides.** The model plans and labels evidence;
deterministic code validates every claim against an evidence ledger, checks OCR confidence word by word, and
decides the final outcome.

## 1. Adaptive query investigation

| Requirement | Implementation | Verify |
|---|---|---|
| Work out what information the question needs | The planner (`app/agent/planner.py`) keeps a list of sub-questions, each with a status (OPEN, SUPPORTED, MISSING, CONFLICTING, UNVERIFIED, INSUFFICIENT) and the evidence ids behind it | UI: *Sub-questions* tab |
| Choose actions; no hardcoded workflow | One generic loop (`app/agent/loop.py`): each step the planner picks one tool (`search`, `get_metadata`, `process_document`, `ask_user`, `finalize`); the same loop handles every question | UI timeline differs per question; Test `test_fast_path_answers_in_one_planner_call` |
| Search Elasticsearch | `app/tools/search_kb.py` → hybrid retriever | Test `test_index_search_scope_roundtrip` (real ES) |
| Retrieve document metadata from SQLite | `app/tools/get_metadata.py` (effective dates, `supersedes`, OCR pages) | Manual 7 |
| Process an uploaded document through OCR | `app/tools/process_document.py` → ingestion pipeline | Manual 13-16, 19-20 |
| Invoke multiple tools in sequence | Loop runs until finalize / pause / budget; state saved after every step | Test `test_pauses_for_missing_purchase_date`; `python -m scripts.demo` |
| Evidence from multiple sources | Evidence ledger (`app/agent/ledger.py`) collects passages, fields, metadata and user replies with ids `e1, e2 ...`; answers cite them | Manual 5, 13 |
| Distinguish **missing information** | MISSING sub-question → `MISSING_INFO`; the agent may ask once (`ask_user`), only after investigating (≥ 3 actions), never for an off-topic question | Test `test_pauses_for_missing_purchase_date`; Manual 10a/10b |
| Distinguish **failed tool execution** | Typed `Observation` (OK/EMPTY/ERROR/BLOCKED/PAUSED) per tool; failed tools tracked → `TOOL_FAILURE`, never "no information"; names the failed component | Tests `test_persistent_tool_failure_is_reported_as_tool_failure`, `test_search_outage_stops_early_with_a_clear_reason`, `test_llm_outage_is_a_tool_failure`; Manual 23-24 |
| Distinguish **insufficient evidence** | Code-level checks: relevance of cited passages, unknown models (`extract_identifiers` + `text_mentions`), off-topic questions (best reranker score), unverified OCR | Tests `test_unknown_model_is_refused_even_if_the_planner_answers`, `test_off_topic_question_is_not_asked_about_and_is_insufficient`, `test_low_relevance_chunks_cannot_support`; Manual 8, 9 |
| Don't treat every problem as a reason to answer | Outcome decided in code (`decide_outcome` in `app/agent/guards.py`) from validated statuses: ANSWERED, ANSWERED_WITH_CAVEATS, MISSING_INFO, INSUFFICIENT_EVIDENCE, CONFLICT, TOOL_FAILURE; refusals are written by code, not the model | Test `test_outcome_priority`, `test_refusal_is_composed_without_llm` |

## 2. Evidence-aware retrieval

| Requirement | Implementation | Verify |
|---|---|---|
| Elasticsearch + embeddings + LLM | ES 8 index with BM25 (`text`, `section^2`, `doc_name`) and dense kNN (`BAAI/bge-small-en-v1.5`, 384-d); the LLM plans and writes the answer | `python -m scripts.eval_retrieval` |
| Reranker (BGE) | `BAAI/bge-reranker-base` scores "document - section + text"; fused with BM25 and kNN by reciprocal-rank fusion (reranker is a voter, not the judge) | Test `test_reranker_is_a_voter_not_the_judge` |
| Preserve source and page | Section-aware chunking with deterministic ids `{doc}:{page}:{idx}`; every hit and citation carries document, page and section | Test `test_hits_keep_source_and_page`; UI citation drawer opens the page |
| Combine evidence from several chunks | Sub-questions cite several ledger entries; the citation verifier keeps only real ids and requires each answered sub-question to be cited | Test `test_verify_citations_drops_unknown_and_duplicates` |
| Identify when context does not support an answer | Relevance rule (reranker ≥ floor or top-3 of the fused ranking); off-topic detection; answer review: every date must come from evidence (or follow from it by a stated period) | Tests `test_answer_without_valid_citations_becomes_refusal`, `test_date_from_outside_the_evidence_is_flagged` |
| Information **spread across documents** | Corpus catalogue shown to the planner; version-aware retrieval also searches an amendment when the document it amends is retrieved | Test `test_amendment_of_a_retrieved_document_is_fetched_before_planning`; Manual 5, 13 |
| **Misleading** evidence | X300 decoy filtered by product; unknown models refused in code; a document's effective date may not be used to compute deadlines | Tests `test_product_filter_excludes_decoy`, `test_deadline_computed_from_a_documents_effective_date_is_sent_back`; Manual 17 |
| **Contradictory** evidence | Policy (24 months) vs amendment (12 months) resolved through `supersedes` metadata; unresolved disagreement → `CONFLICT` with both sides cited | Test `test_unresolved_conflict_outcome`; Manual 7, 18 |
| Measured, not assumed | 42 held-out questions per section and question type: recall@k / MRR per retrieval variant and category; end-to-end outcome + required facts | `scripts/eval_retrieval.py`, `scripts/eval_answers.py`, `scripts/compare_models.py` |

## 3. On-demand document intelligence

| Requirement | Implementation | Verify |
|---|---|---|
| Introduce documents during an investigation | Upload with the question or when the agent asks (`POST /investigations/{id}/continue`); an attached document is read and searched before the next planning step | Test `test_attachment_is_read_before_the_first_planner_call`; Manual 11b, 13 |
| Decide whether OCR is needed | Per-page triage (`app/ingestion/triage.py`): pages with a text layer use it; only pages without one are OCR'd (Tesseract 5) | Test `test_triage_uses_text_layer_and_only_ocrs_scanned_pages`; contract: OCR on page 3 only |
| Extract relevant information | `process_document` `extract_only`: the LLM reads named fields; code grounds each value in the page | Test `test_clean_receipt_gives_cited_answer` |
| Answer directly from the upload, or index it | `extract_only` (one-off evidence) or `index` (searchable for this investigation) | Manual 19 (answer from card), 20 (indexed log) |
| Avoid unnecessary processing | Upload registration is cheap (hash + triage); OCR happens only when needed; SHA-256 dedupe; results cached in SQLite | Tests `test_upload_registration_is_cheap`, `test_duplicate_upload_returns_existing_document` |
| Uncertain OCR is never a verified fact | Per-word Tesseract confidence; words < 80 are marked `[[?word]]`; a value is VERIFIED only if every word that spells it is ≥ 80 (`app/ingestion/ocr_verify.py`); claims resting on uncertain words stay UNVERIFIED and block a verdict they decide | Tests `test_hinge_receipt_gives_explicit_refusal`, `test_genuinely_uncertain_claims_stay_unverified`, `test_real_tesseract_flags_the_smudged_date`; Manual 15-16 |
| ...but a clear value next to a smudge is usable | Word-level verification: the smudged signature on the contract scan does not make "COMMERCIAL" uncertain | Tests `test_passage_flagged_by_an_unrelated_smudge_is_verified_at_field_level`, `test_quote_with_copied_markers_and_page_reference_still_verifies` |
| Uploads stay private | Session scope: an upload is visible only to the investigations it is attached to; sharing is the explicit `promote` endpoint | Tests `test_session_documents_are_invisible_to_other_investigations`, `test_agent_cannot_publish_an_upload_to_other_investigations` |

## 4. Reliability and system design

| Requirement | Implementation | Verify |
|---|---|---|
| Independently testable parts | Interfaces + a wiring container (`app/container.py`); tests swap in fakes for the LLM, index, embedder, reranker and OCR (`tests/fakes.py`) - no network, ES, models or key needed | `python -m pytest` (156 tests; 1 needs a running Elasticsearch) |
| **Malformed model output** | Schema-constrained output (Ollama compiles the JSON schema into a grammar), typed action arguments, one repair round with the validation error, prompt-only retry for empty replies, safe fallback | Tests `test_malformed_output_is_repaired_once`, `test_malformed_output_twice_falls_back_safely`, `test_empty_constrained_reply_is_rescued_by_a_prompt_only_retry` |
| **Tool failures** | Registry validates arguments (bad arguments are not retried), retries transient errors with backoff, consecutive-failure budget; Elasticsearch circuit breaker; fallback to BM25 if embeddings fail and to fused order if the reranker fails; a stopped LLM is reported as TOOL_FAILURE naming the component | Tests `test_transient_search_failure_is_retried_and_recovers`, `test_embedder_failure_degrades_to_bm25`, `test_elasticsearch_circuit_breaker_fails_fast`, `test_missing_model_and_server_down_are_clear_errors` |
| **Repeated tool calls** | Call hashing (normalised arguments); identical successful calls blocked; repeated failing calls blocked; step budget | Tests `test_repeated_successful_call_is_blocked`, `test_step_budget_forces_finalize` |
| **Duplicate uploads** | SHA-256 of the bytes; the same content under any file name returns the existing document, not reprocessed | Tests `test_duplicate_receipt_upload_not_reprocessed`; Manual 21 |
| **Interrupted processing** | Ingestion state machine RECEIVED → TRIAGED → EXTRACTED → CHUNKED → INDEXED with per-page OCR progress and an atomic lock; on startup stale locks are cleared, documents resume, running investigations become INTERRUPTED (resumable) | Tests `test_crash_mid_ocr_resumes_without_redoing_pages`, `test_stale_lock_cleared_and_pending_document_resumed_on_startup`, `test_interrupted_investigation_resumes`; Manual 20 |
| Architecture diagram | Mermaid diagram | `docs/system_design.md` §2 |
| Concurrent users, larger collections | Stateless API, state in SQLite/ES, per-investigation claim lock, scaling notes | `docs/system_design.md` §7 |

## 5. Required demonstration

`python -m scripts.demo` (or the UI) with the main question - *Is the motor seizure of Acme Facilities' Aquaflow
X200 pump (serial AQ-X200-55871) on 14 March 2026 covered under warranty?*

| Requirement | Where it happens |
|---|---|
| Needs more than one document | Policy + amendment + X200 manual + Acme contract (+ the receipt) |
| Incomplete, contradictory or hard-to-extract information | Purchase date missing; 24 vs 12 months; commercial use only on a scanned page; smudged receipt |
| Uses an uploaded document | The agent asks for the receipt; the upload is OCR'd and its date verified word by word |
| Cited answer or explicit explanation | Clean receipt → cited "not covered" / "covered"; smudged receipt → INSUFFICIENT_EVIDENCE with the reason |

## 6. Technical requirements

| Requirement | Choice | Where |
|---|---|---|
| FastAPI backend | FastAPI + Uvicorn; Swagger at `/docs`; optional web UI at `/ui` (single static page) | `app/main.py`, `app/api/` |
| Elasticsearch for chunk indexing and retrieval | Elasticsearch 8.15 (Docker), BM25 + kNN `dense_vector` | `app/retrieval/es_client.py`, `docker-compose.yml` |
| SQLite for metadata and processing state | SQLite (WAL): documents, pages (per-word OCR confidence), chunks, investigations (full state after every step) | `app/db/schema.sql`, `app/db/repo.py` |
| Embedding model | `BAAI/bge-small-en-v1.5` (sentence-transformers, CPU) | `app/retrieval/embedder.py` |
| LLM | `gpt-oss:20b` in Ollama, local (native API: schema-constrained output, low reasoning effort, 16k context per request); the client interface is pluggable | `app/llm/ollama.py`, `app/llm/base.py` |
| OCR | Tesseract 5 via `pytesseract` (per-word confidence); PyMuPDF for text layers and page rendering | `app/ingestion/ocr.py`, `app/ingestion/triage.py` |
| Automated tests for workflow and failure cases | pytest, 156 tests: end-to-end workflow over HTTP, every failure case above, ingestion, OCR confidence, retrieval, LLM clients | `tests/` |

## 7. Deliverables

| Asked for | In the repository |
|---|---|
| Source code | `app/`, `scripts/` |
| README | `README.md` |
| Sample documents | `samples/` (+ `samples/test_uploads/`), described in `docs/corpus.md` |
| API examples | `docs/api_examples.md` (curl), Swagger at `/docs` |
| Tests | `tests/` |
| Short system design document | `docs/system_design.md` |

Also included: a web UI to watch an investigation and verify every citation against the original page, an
evaluation harness with a held-out question set, a manual behaviour test pack, and a decision log
(`docs/DECISIONS.md`) recording what failed during testing and how each failure class was fixed.

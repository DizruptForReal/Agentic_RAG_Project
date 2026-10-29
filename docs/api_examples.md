# API examples

Base URL `http://localhost:8000`. Interactive docs at `/docs`.
Examples use `curl` (on Windows PowerShell call `curl.exe`, not the `curl` alias).

## Health

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "elasticsearch": {"ok": true, "chunks": 40}, "llm_model": "openai/gpt-oss-120b",
 "recovery": {"stale_locks": [], "interrupted": [], "resumed_documents": []}}
```

## Add a document to the corpus

```bash
curl -F "file=@samples/warranty_policy_v1.pdf" -F "doc_type=policy" -F "product=X-series" \
     -F "effective_date=2023-01-01" -F "title=Aquaflow Limited Warranty Policy v1.0 (AQF-WP-1.0)" \
     http://localhost:8000/documents
```

Uploading the same bytes again returns the existing record with `"duplicate": true`. Indexing runs in the background; check it with:

```bash
curl http://localhost:8000/documents/d_8738d11c3cd5ffd4
# {"status": "INDEXED", "error": null, "error_stage": null, ...}
```

If processing was interrupted or failed (`error_stage` set), resume from the last completed stage:

```bash
curl -X POST http://localhost:8000/documents/d_8738d11c3cd5ffd4/resume
```

## Run in the background and watch progress

Add `-F "wait=false"` to `POST /investigations` or `/continue`: the call returns at once with `status: RUNNING`, and
`GET /investigations/{id}` then shows `activity` (what it is doing right now) and the steps so far, each with a
plain-language `label` and `llm_ms` / `tool_ms` / `wait_ms` timings. The web UI uses this.

```json
{"status": "RUNNING", "activity": {"text": "Waiting for the model's rate limit (21s)", "kind": "wait", "since": 1793136948.2},
 "steps": [{"idx": 1, "auto": true, "label": "Searching the knowledge base for \u201cWhat all is covered under the warranty?\u201d", "tool_ms": 1240}]}
```

## Start an investigation

```bash
curl -F "question=Is the motor seizure of Acme Facilities' Aquaflow X200 pump (serial AQ-X200-55871) on 14 March 2026 covered under warranty?" \
     http://localhost:8000/investigations
```

Typical response (abridged):

```json
{
  "id": "inv_3f2a9c1d7e4b",
  "status": "NEEDS_INPUT",
  "outcome": null,
  "pending_question": "Please upload the original sales receipt showing the purchase date.",
  "sub_questions": [
    {"id": "q1", "text": "Which warranty period applies?", "status": "SUPPORTED", "evidence_ids": ["e2", "e7"]},
    {"id": "q2", "text": "Is the unit used commercially?", "status": "SUPPORTED", "evidence_ids": ["e9"]},
    {"id": "q3", "text": "Is motor seizure a failure?", "status": "SUPPORTED", "evidence_ids": ["e11"]},
    {"id": "q4", "text": "What is the purchase date?", "status": "MISSING", "evidence_ids": []}
  ],
  "steps": [
    {"idx": 1, "action": {"type": "search", "args": {"query": "warranty period commercial"}},
     "observation": {"status": "OK", "summary": "6 chunks from 5 documents; 4 above relevance floor 0.2"}},
    {"idx": 2, "action": {"type": "get_metadata", "args": {"doc_type": "amendment"}}, "...": "..."}
  ],
  "ledger": [{"id": "e1", "kind": "chunk", "doc_name": "...", "page": 1, "verification": "VERIFIED",
              "rerank_score": 0.93, "text": "..."}]
}
```

You can attach a file when starting (`-F "file=@samples/receipt_hinge.png"`).

## Continue with an upload

```bash
curl -F "message=Here is the receipt" -F "file=@samples/receipt_hinge.png" \
     http://localhost:8000/investigations/inv_3f2a9c1d7e4b/continue
```

Hinge receipt (smudged date) — explicit refusal:

```json
{
  "status": "COMPLETED",
  "outcome": "INSUFFICIENT_EVIDENCE",
  "upload": {"doc_id": "d_5e0c...", "duplicate": false, "status": "TRIAGED", "ocr_pages": [1]},
  "answer": {
    "outcome": "INSUFFICIENT_EVIDENCE",
    "answer": "A reliable answer cannot be given: the available evidence does not adequately support one. ...",
    "reasons": ["q4 depends on unverified OCR evidence that could change the verdict: ..."],
    "caveats": ["[e14] from receipt_hinge.png (page 1) is UNVERIFIED, OCR confidence 71; it was not treated as a verified fact."],
    "citations": [{"evidence_id": "e7", "doc_name": "Warranty Policy Amendment No. 1 (AQF-WP-1.0-A1)", "page": 1, "verification": "VERIFIED", "excerpt": "..."}]
  }
}
```

Clean receipt — cited answer:

```json
{
  "outcome": "ANSWERED",
  "answer": {
    "answer": "Not covered. The unit is commercial [e9], so the amendment's 12-month term applies [e7] ... purchased 05-03-2025 [e14], cover ended 05-03-2026, before the 14 March 2026 failure.",
    "citations": ["... policy, amendment, metadata, contract p.3, X200 manual p.1, receipt p.1 ..."]
  }
}
```

Other `/continue` uses:
- answer a question without a file: `-F "message=The unit was bought on 17 March 2025"` (stored as a user statement; the agent decides how much weight it carries);
- resume an `INTERRUPTED` or `FAILED` investigation: call `/continue` with or without a message.

## Inspect an investigation

```bash
curl http://localhost:8000/investigations/inv_3f2a9c1d7e4b
```

## Promote a session document to the global corpus

```bash
curl -X POST http://localhost:8000/documents/d_5e0c.../promote
```

## Error responses

| Code | When |
|---|---|
| 400 | empty file; `/continue` on a `NEEDS_INPUT` investigation with no message or file |
| 404 | unknown investigation or document |
| 409 | investigation already running; document being processed |
| 415 | unsupported file type |
| 422 / 502 | ingestion failed (non-retryable / retryable) |

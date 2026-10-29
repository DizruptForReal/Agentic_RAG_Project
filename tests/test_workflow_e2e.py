"""End-to-end investigation through the HTTP API with scripted LLM output.

Covers the required demonstration: multiple documents, a conflict resolved by metadata, a decoy,
a scanned page, a pause for missing information, an uploaded scan with uncertain OCR, and a final
answer with sources or an explicit refusal."""
from __future__ import annotations

import re

from tests.conftest import SAMPLES
from tests.fakes import receipt_hinge, receipt_nonhinge
from tests.scenario import QUESTION, warranty_run1, warranty_run2

CONTRACT_EXTRACT = {"fields": [{"name": "usage_classification", "found": True, "value": "COMMERCIAL",
                                "page": 3, "source_text": "Usage classification: COMMERCIAL"}]}


def _cite_everything(user: str):
    ids = sorted(set(re.findall(r"^\[(e\d+)\]", user, flags=re.M)))
    return {"answer": "Not covered: the 12-month commercial warranty expired on 5 March 2026 [e1].",
            "citations": ids + ["e999"], "caveats": []}


def _start(client, llm):
    llm.push("plan_step", *warranty_run1())
    llm.push("extract_fields", CONTRACT_EXTRACT)
    r = client.post("/investigations", data={"question": QUESTION})
    assert r.status_code == 200, r.text
    return r.json()


def _doc_names(body):
    return {c["doc_name"] for c in body["answer"]["citations"]}


def test_pauses_for_missing_purchase_date(client, llm, seeded):
    body = _start(client, llm)
    assert body["status"] == "NEEDS_INPUT"
    assert "receipt" in body["pending_question"].lower()
    statuses = {q["id"]: q["status"] for q in body["sub_questions"]}
    assert statuses == {"q1": "SUPPORTED", "q2": "SUPPORTED", "q3": "SUPPORTED", "q4": "MISSING"}
    tools = [s["action"]["type"] for s in body["steps"]]
    assert tools == ["search", "get_metadata", "process_document", "search", "search", "ask_user"]
    # the decoy surfaced in retrieval ...
    assert any("X300" in e["doc_name"] for e in body["ledger"])
    # ... and the scanned contract page was used through field extraction, verified at field level
    usage = next(e for e in body["ledger"] if e["field_name"] == "usage_classification")
    assert usage["verification"] == "VERIFIED" and usage["page"] == 3


def test_hinge_receipt_gives_explicit_refusal(client, llm, ocr, seeded):
    inv = _start(client, llm)
    ocr.queue.append(receipt_hinge())
    llm.push("plan_step", *warranty_run2(robust=False))
    llm.push("extract_fields", {"fields": [
        {"name": "purchase_date", "found": True, "value": "1-03-2025", "page": 1, "source_text": "Date: 1-03-2025"},
        {"name": "receipt_number", "found": True, "value": "R-2025-04417", "page": 1,
         "source_text": "Receipt No: R-2025-04417"}]})
    with open(SAMPLES / "receipt_hinge.png", "rb") as f:
        r = client.post(f"/investigations/{inv['id']}/continue", files={"file": ("receipt.png", f, "image/png")})
    body = r.json()
    assert body["status"] == "COMPLETED"
    assert body["upload"]["ocr_pages"] == [1] and body["upload"]["duplicate"] is False
    assert body["outcome"] == "INSUFFICIENT_EVIDENCE"
    q4 = next(q for q in body["sub_questions"] if q["id"] == "q4")
    assert q4["status"] == "UNVERIFIED"            # guard downgraded the planner's SUPPORTED claim
    assert "unverified" in " ".join(body["answer"]["reasons"]).lower()
    assert any("UNVERIFIED" in c for c in body["answer"]["caveats"])
    assert llm.count("compose_answer") == 0        # refusals are composed in code, not by the LLM
    assert not any("X300" in n for n in _doc_names(body))


def test_clean_receipt_gives_cited_answer(client, llm, ocr, seeded):
    inv = _start(client, llm)
    ocr.queue.append(receipt_nonhinge())
    llm.push("plan_step", *warranty_run2(robust=False))
    llm.push("extract_fields", {"fields": [
        {"name": "purchase_date", "found": True, "value": "05-03-2025", "page": 1, "source_text": "Date: 05-03-2025"},
        {"name": "receipt_number", "found": True, "value": "R-2025-03892", "page": 1,
         "source_text": "Receipt No: R-2025-03892"}]})
    llm.push("compose_answer", _cite_everything)
    with open(SAMPLES / "receipt_nonhinge.png", "rb") as f:
        r = client.post(f"/investigations/{inv['id']}/continue", files={"file": ("receipt.png", f, "image/png")})
    body = r.json()
    assert body["outcome"] == "ANSWERED", body["answer"]
    ans = body["answer"]
    cited = {c["evidence_id"] for c in ans["citations"]}
    assert "e999" not in cited                                     # hallucinated id removed
    assert any("e999" in r for r in ans["reasons"])
    assert len({c["doc_id"] for c in ans["citations"]}) >= 4      # policy+amendment, contract, manual, receipt
    assert not any("X300" in n for n in _doc_names(body))          # decoy never cited
    kinds = {e["id"]: e["kind"] for e in body["ledger"]}
    assert all(c["page"] is not None for c in ans["citations"] if kinds[c["evidence_id"]] != "metadata")
    fields = {e["field_name"]: e["verification"] for e in body["ledger"] if e["kind"] == "field"}
    assert fields["purchase_date"] == "VERIFIED" and fields["receipt_number"] == "UNVERIFIED"


def test_duplicate_receipt_upload_not_reprocessed(client, llm, ocr, seeded):
    inv = _start(client, llm)
    ocr.queue.append(receipt_nonhinge())
    content = (SAMPLES / "receipt_nonhinge.png").read_bytes()
    r1 = client.post("/documents", files={"file": ("r1.png", content, "image/png")}, data={"index": "false"})
    calls = ocr.calls
    llm.push("plan_step", lambda u: {"sub_questions": [{"id": "q4", "text": "purchase date", "status": "MISSING",
                                                        "evidence_ids": []}],
                                     "action": {"type": "finalize", "args": {}, "reason": "stop"}})
    r2 = client.post(f"/investigations/{inv['id']}/continue", files={"file": ("again.png", content, "image/png")})
    assert r2.json()["upload"]["duplicate"] is True
    assert r2.json()["upload"]["doc_id"] == r1.json()["doc_id"]
    assert ocr.calls == calls
    assert r2.json()["outcome"] == "MISSING_INFO"


def test_unanswerable_question_is_refused(client, llm, seeded):
    llm.push("plan_step",
             lambda u: {"sub_questions": [{"id": "q1", "text": "Warranty period of the X500", "status": "OPEN",
                                           "evidence_ids": []}],
                        "action": {"type": "search", "args": {"query": "X500 warranty period"}, "reason": "look"}},
             lambda u: {"sub_questions": [{"id": "q1", "text": "Warranty period of the X500", "status": "INSUFFICIENT",
                                           "evidence_ids": [], "note": "only X200/X300 documents exist"}],
                        "action": {"type": "finalize", "args": {}, "reason": "nothing about X500"}})
    r = client.post("/investigations", data={"question": "What is the warranty period for the Aquaflow X500?"})
    body = r.json()
    assert body["outcome"] == "INSUFFICIENT_EVIDENCE"
    assert "do not cover X500" in body["answer"]["answer"]
    assert llm.count("compose_answer") == 0

"""Endpoints that give the user visibility: web UI, listings, original files, extracted pages, async runs."""
from __future__ import annotations

import time

from tests.conftest import SAMPLES


def test_ui_is_served(client):
    r = client.get("/ui")
    assert r.status_code == 200 and "Evidence Investigator" in r.text
    assert client.get("/", follow_redirects=False).status_code in (302, 307)


def test_document_listing_file_and_pages(client, seeded):
    docs = client.get("/documents").json()
    assert len(docs) == 5
    contract = seeded["customer_contract_acme.pdf"]
    f = client.get(f"/documents/{contract}/file")
    assert f.status_code == 200 and f.headers["content-type"] == "application/pdf"
    assert f.headers["content-disposition"].startswith("inline")
    pages = client.get(f"/documents/{contract}/pages").json()
    assert [p["method"] for p in pages["pages"]] == ["native", "native", "ocr"]
    assert pages["ocr_threshold"] == 80
    ocr_words = pages["pages"][2]["words"]
    assert all(c is not None for _, c in ocr_words) and any(c < 80 for _, c in ocr_words)
    assert client.get("/documents/d_nope/file").status_code == 404


def test_background_run_and_listing(client, llm, seeded):
    llm.push("plan_step", lambda u: {"sub_questions": [{"id": "q1", "text": "X500 term", "status": "INSUFFICIENT",
                                                        "evidence_ids": []}],
                                     "action": {"type": "finalize", "args": {}, "reason": "none"}})
    r = client.post("/investigations", data={"question": "What is the X500 warranty?", "wait": "false"})
    assert r.status_code == 200 and r.json()["status"] == "RUNNING"
    inv_id = r.json()["id"]
    for _ in range(50):
        body = client.get(f"/investigations/{inv_id}").json()
        if body["status"] != "RUNNING":
            break
        time.sleep(0.05)
    assert body["status"] == "COMPLETED" and body["outcome"] == "INSUFFICIENT_EVIDENCE"
    listed = client.get("/investigations").json()
    assert listed[0]["id"] == inv_id and listed[0]["outcome"] == "INSUFFICIENT_EVIDENCE"


def test_unsupported_upload_creates_no_investigation(client):
    before = len(client.get("/investigations").json())
    r = client.post("/investigations", data={"question": "What is in this file?"},
                    files={"file": ("notes.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 415 and "unsupported file type" in r.json()["detail"]
    assert len(client.get("/investigations").json()) == before

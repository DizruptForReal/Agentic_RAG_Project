"""Ingestion: triage avoids unnecessary OCR, duplicates are not reprocessed, crashes resume."""
from __future__ import annotations

import io

import pytest
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from app.ingestion.chunker import chunk_page, clean_text
from app.ingestion.pipeline import IngestionError
from tests.conftest import SAMPLES
from tests.fakes import ocr_crash, ocr_page


def _image_pdf(pages: int) -> bytes:
    """A PDF whose pages are images only (no text layer)."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for i in range(pages):
        img = Image.new("L", (400, 200), 255 - i)
        c.drawImage(ImageReader(img), 50, 400, 400, 200)
        c.showPage()
    c.save()
    return buf.getvalue()


def test_triage_uses_text_layer_and_only_ocrs_scanned_pages(container, ocr, seeded):
    calls_after_seed = ocr.calls
    assert calls_after_seed == 1                       # only the contract's scanned page 3 needed OCR
    contract = container.repo.get_document(seeded["customer_contract_acme.pdf"])
    pages = container.repo.get_pages(contract["doc_id"])
    assert [p["method"] for p in pages] == ["native", "native", "ocr"]
    policy_pages = container.repo.get_pages(seeded["warranty_policy_v1.pdf"])
    assert all(p["method"] == "native" for p in policy_pages)


def test_upload_registration_is_cheap(container, ocr):
    doc, dup = container.pipeline.register("receipt.png", (SAMPLES / "receipt_hinge.png").read_bytes(),
                                           scope="session:x")
    assert not dup and doc["status"] == "TRIAGED" and ocr.calls == 0   # OCR waits until needed


def test_duplicate_upload_returns_existing_document(container, ocr):
    content = (SAMPLES / "receipt_hinge.png").read_bytes()
    d1, dup1 = container.pipeline.register("a.png", content)
    container.pipeline.ensure_extracted(d1["doc_id"])
    d2, dup2 = container.pipeline.register("b.png", content)
    assert (dup1, dup2) == (False, True)
    assert d2["doc_id"] == d1["doc_id"] and d2["status"] == "EXTRACTED"
    container.pipeline.ensure_extracted(d2["doc_id"])
    assert ocr.calls == 1


def test_unsupported_file_type_rejected(container):
    with pytest.raises(ValueError):
        container.pipeline.register("notes.docx", b"PK\x03\x04 not supported")


def test_crash_mid_ocr_resumes_without_redoing_pages(container, ocr):
    doc, _ = container.pipeline.register("scan.pdf", _image_pdf(3))
    ocr.queue.extend([ocr_page("page one text"), ocr_crash()])
    with pytest.raises(IngestionError) as err:
        container.pipeline.ensure_extracted(doc["doc_id"])
    assert err.value.stage == "EXTRACTED"
    row = container.repo.get_document(doc["doc_id"])
    assert row["status"] == "TRIAGED" and row["error_stage"] == "EXTRACTED" and row["processing"] == 0
    assert [p["page"] for p in container.repo.get_pages(doc["doc_id"])] == [1]   # page 1 survived

    calls_before = ocr.calls
    row = container.pipeline.ensure_extracted(doc["doc_id"])
    assert row["status"] == "EXTRACTED" and row["error"] is None
    assert ocr.calls - calls_before == 2                                          # only pages 2 and 3


def test_stale_lock_cleared_and_pending_document_resumed_on_startup(container, ocr):
    doc, _ = container.pipeline.register("scan.pdf", _image_pdf(1))
    container.repo.update_document(doc["doc_id"], target_stage="INDEXED", processing=1)  # killed while held
    report = container.startup()
    assert doc["doc_id"] in report["stale_locks"]
    assert doc["doc_id"] in report["resumed_documents"]
    assert container.repo.get_document(doc["doc_id"])["status"] == "INDEXED"


def test_busy_document_is_not_processed_twice(container):
    from app.ingestion.pipeline import DocumentBusy

    doc, _ = container.pipeline.register("scan.pdf", _image_pdf(1))
    assert container.repo.try_lock(doc["doc_id"])
    with pytest.raises(DocumentBusy):
        container.pipeline.ensure_extracted(doc["doc_id"])


def test_reindex_is_idempotent_and_promote_changes_scope(container, index):
    doc, _ = container.pipeline.register("policy.pdf", (SAMPLES / "warranty_policy_v1.pdf").read_bytes(),
                                         scope="session:inv_1")
    container.pipeline.ensure_indexed(doc["doc_id"])
    n = index.count(doc["doc_id"])
    container.pipeline._stage_indexed(doc["doc_id"])          # run the stage again
    assert index.count(doc["doc_id"]) == n
    assert {c["scope"] for c in index.docs.values() if c["doc_id"] == doc["doc_id"]} == {"session:inv_1"}
    container.pipeline.ensure_indexed(doc["doc_id"], scope="global")
    assert {c["scope"] for c in index.docs.values() if c["doc_id"] == doc["doc_id"]} == {"global"}


def test_chunk_ids_deterministic_and_low_confidence_marked():
    words = [("Usage", 96.0), ("classification:", 95.0), ("COMMERC1AL", 41.0)] + [("w", 90.0)] * 10
    a = chunk_page("d_x", 3, words, size=5, overlap=1, conf_threshold=80)
    b = chunk_page("d_x", 3, words, size=5, overlap=1, conf_threshold=80)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b] == [f"d_x:3:{i}" for i in range(len(a))]
    assert "[[?COMMERC1AL]]" in a[0].text and a[0].verification == "UNVERIFIED"
    assert clean_text(a[0].text).startswith("Usage classification: COMMERC1AL")
    assert a[-1].verification == "VERIFIED"


def test_api_upload_and_resume_endpoints(client, ocr):
    content = _image_pdf(2)
    ocr.queue.extend([ocr_page("first page"), ocr_crash()])
    r = client.post("/documents", files={"file": ("scan.pdf", content, "application/pdf")})
    assert r.status_code == 200
    doc = client.get(f"/documents/{r.json()['doc_id']}").json()
    assert doc["error_stage"] == "EXTRACTED" and doc["status"] == "TRIAGED"     # background run failed
    r2 = client.post(f"/documents/{doc['doc_id']}/resume")
    assert r2.status_code == 200 and r2.json()["status"] == "INDEXED"
    dup = client.post("/documents", files={"file": ("again.pdf", content, "application/pdf")})
    assert dup.json()["duplicate"] is True

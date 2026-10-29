"""Document endpoints: upload (with dedupe), status, resume, promote to global."""
from __future__ import annotations

from typing import Literal

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.api.deps import get_container
from app.api.schemas import DocumentView
from app.container import Container
from app.ingestion.pipeline import DocumentBusy, IngestionError

router = APIRouter(prefix="/documents", tags=["documents"])


def _process_quietly(c: Container, doc_id: str, target: str) -> None:
    try:
        c.pipeline.run_to(doc_id, target)
    except (IngestionError, DocumentBusy):
        pass  # state (error, error_stage) is recorded in SQLite; GET /documents/{id} shows it


@router.post("", response_model=DocumentView)
def upload_document(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    title: str | None = Form(None),
    doc_type: str | None = Form(None),
    product: str | None = Form(None),
    effective_date: str | None = Form(None),
    supersedes: str | None = Form(None),
    index: bool = Form(True),
    c: Container = Depends(get_container),
):
    """Add a document to the global corpus. Duplicates (same bytes) return the existing record."""
    content = file.file.read()
    if not content:
        raise HTTPException(400, "empty file")
    meta = {"title": title, "doc_type": doc_type, "product": product, "effective_date": effective_date,
            "supersedes": supersedes}
    try:
        doc, duplicate = c.pipeline.register(file.filename or "upload", content, scope="global", meta=meta)
    except ValueError as exc:
        raise HTTPException(415, str(exc)) from exc
    except IngestionError as exc:
        raise HTTPException(422, str(exc)) from exc
    if index and not duplicate:
        background.add_task(_process_quietly, c, doc["doc_id"], "INDEXED")
    return DocumentView.of(doc, duplicate=duplicate)


@router.get("/{doc_id}", response_model=DocumentView)
def get_document(doc_id: str, c: Container = Depends(get_container)):
    doc = c.repo.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "document not found")
    return DocumentView.of(doc)


@router.get("")
def list_documents(c: Container = Depends(get_container)):
    """All documents with metadata and processing state."""
    return [DocumentView.of(d) for d in c.repo.list_documents()]


@router.get("/{doc_id}/file")
def get_document_file(doc_id: str, c: Container = Depends(get_container)):
    """The original uploaded file, shown inline (PDFs open in the browser; append #page=N)."""
    doc = c.repo.get_document(doc_id)
    if not doc or not Path(doc["path"]).exists():
        raise HTTPException(404, "document file not found")
    return FileResponse(doc["path"], media_type=doc["mime"], filename=doc["filename"],
                        content_disposition_type="inline")


@router.get("/{doc_id}/pages")
def get_document_pages(doc_id: str, c: Container = Depends(get_container)):
    """Extracted text per page as the system saw it: method (native/ocr), OCR confidence per word,
    and the threshold below which words are treated as UNVERIFIED."""
    doc = c.repo.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "document not found")
    pages = [{"page": p["page"], "method": p["method"], "mean_conf": p["mean_conf"], "text": p["text"],
              "words": p["words"]} for p in c.repo.get_pages(doc_id)]
    return {"doc_id": doc_id, "title": doc["title"] or doc["filename"], "status": doc["status"],
            "mime": doc["mime"], "ocr_threshold": c.settings.ocr_conf_threshold, "pages": pages}


@router.post("/{doc_id}/resume", response_model=DocumentView)
def resume_document(doc_id: str, target: Literal["EXTRACTED", "INDEXED"] | None = None,
                    c: Container = Depends(get_container)):
    """Continue an interrupted or failed ingestion from its last completed stage."""
    doc = c.repo.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "document not found")
    try:
        doc = c.pipeline.run_to(doc_id, target or doc["target_stage"] or "INDEXED")
    except DocumentBusy as exc:
        raise HTTPException(409, str(exc)) from exc
    except IngestionError as exc:
        raise HTTPException(502 if exc.retryable else 422, str(exc)) from exc
    return DocumentView.of(doc)


@router.post("/{doc_id}/promote", response_model=DocumentView)
def promote_document(doc_id: str, c: Container = Depends(get_container)):
    """Make a session document searchable by every investigation."""
    if not c.repo.get_document(doc_id):
        raise HTTPException(404, "document not found")
    try:
        doc = c.pipeline.ensure_indexed(doc_id, scope="global")
    except (IngestionError, DocumentBusy) as exc:
        raise HTTPException(409, str(exc)) from exc
    return DocumentView.of(doc)

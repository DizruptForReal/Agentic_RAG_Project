"""Resumable ingestion state machine.

    RECEIVED -> TRIAGED -> EXTRACTED -> CHUNKED -> INDEXED

* ``documents.status`` is the last COMPLETED stage, so a crash leaves the document resumable.
* Every stage is idempotent: EXTRACTED skips pages already stored, CHUNKED replaces the chunk set,
  INDEXED upserts chunks by deterministic id.
* ``processing`` is an atomic lock so two requests never process the same document at once;
  stale locks are cleared on startup.
* Upload registration only stores + triages (cheap). OCR/extraction happens only when the agent asks.
"""
from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import sqlite3
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import Settings
from app.db.repo import Repo
from app.ingestion.chunker import chunk_document, context_text, mark_word
from app.ingestion.ocr import OCREngine, OCRError
from app.ingestion.triage import IMAGE_EXT, is_image, triage_file
from app.retrieval.base import Embedder, SearchIndex, SearchUnavailable

log = logging.getLogger(__name__)

STAGES = ["RECEIVED", "TRIAGED", "EXTRACTED", "CHUNKED", "INDEXED"]
SUPPORTED_EXT = IMAGE_EXT | {".pdf", ".txt"}


class DocumentBusy(Exception):
    """Another worker is processing this document right now."""


class IngestionError(Exception):
    def __init__(self, doc_id: str, stage: str, cause: Exception):
        self.doc_id, self.stage, self.cause = doc_id, stage, cause
        self.retryable = isinstance(cause, (SearchUnavailable, TimeoutError)) or (
            isinstance(cause, OCRError) and getattr(cause, "retryable", True)
            and "not found" not in str(cause)
        )
        super().__init__(f"{stage} failed for {doc_id}: {type(cause).__name__}: {cause}")


def stage_index(stage: str) -> int:
    return STAGES.index(stage)


def _fitz():
    try:
        import pymupdf as fitz
    except ImportError:  # older PyMuPDF
        import fitz
    return fitz


def render_pdf_page(path: Path, page: int, dpi: int) -> Image.Image:
    fitz = _fitz()
    with fitz.open(path) as doc:
        pix = doc[page - 1].get_pixmap(dpi=dpi)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def native_page_text(path: Path, page: int) -> str:
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")
    fitz = _fitz()
    with fitz.open(path) as doc:
        return doc[page - 1].get_text("text")


class IngestionPipeline:
    def __init__(self, repo: Repo, ocr: OCREngine, index: SearchIndex, embedder: Embedder, settings: Settings):
        self.repo, self.ocr, self.index, self.embedder, self.s = repo, ocr, index, embedder, settings

    # ------------------------------------------------------------------ public
    def register(self, filename: str, content: bytes, *, scope: str = "global",
                 meta: dict[str, Any] | None = None) -> tuple[dict[str, Any], bool]:
        """Store + triage an upload. Returns (document, duplicate). Duplicates are never reprocessed."""
        sha = hashlib.sha256(content).hexdigest()
        existing = self.repo.get_document_by_hash(sha)
        if existing:
            return existing, True
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXT:
            raise ValueError(f"unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXT)}")
        doc_id = "d_" + sha[:16]
        self.s.upload_dir.mkdir(parents=True, exist_ok=True)
        path = self.s.upload_dir / f"{doc_id}{ext}"
        tmp = path.with_suffix(ext + ".part")
        tmp.write_bytes(content)
        tmp.replace(path)  # atomic: a crash never leaves a half-written file under the final name
        meta = meta or {}
        try:
            self.repo.insert_document({
                "doc_id": doc_id, "sha256": sha, "filename": filename,
                "mime": mimetypes.guess_type(filename)[0] or "application/octet-stream",
                "path": str(path), "size_bytes": len(content), "title": meta.get("title") or filename,
                "doc_type": meta.get("doc_type"), "product": meta.get("product"),
                "effective_date": meta.get("effective_date"), "supersedes": meta.get("supersedes"),
                "scope": scope, "status": "RECEIVED", "target_stage": "TRIAGED", "processing": 0,
            })
        except sqlite3.IntegrityError:  # concurrent upload of the same bytes won the race
            return self.repo.get_document_by_hash(sha), True
        return self.run_to(doc_id, "TRIAGED"), False

    def ensure_extracted(self, doc_id: str) -> dict[str, Any]:
        return self.run_to(doc_id, "EXTRACTED")

    def ensure_indexed(self, doc_id: str, scope: str | None = None) -> dict[str, Any]:
        doc = self._doc(doc_id)
        if scope and scope != doc["scope"]:
            self.repo.update_document(doc_id, scope=scope)
            if doc["status"] == "INDEXED":
                self.index.set_scope(doc_id, scope)
        return self.run_to(doc_id, "INDEXED")

    def run_to(self, doc_id: str, target: str) -> dict[str, Any]:
        doc = self._doc(doc_id)
        if stage_index(doc["status"]) >= stage_index(target):
            return doc
        if not doc["target_stage"] or stage_index(doc["target_stage"]) < stage_index(target):
            self.repo.update_document(doc_id, target_stage=target)
        if not self.repo.try_lock(doc_id):
            raise DocumentBusy(f"{doc_id} is being processed by another request")
        try:
            current = stage_index(self._doc(doc_id)["status"])
            for stage in STAGES[current + 1: stage_index(target) + 1]:
                try:
                    getattr(self, f"_stage_{stage.lower()}")(doc_id)
                except Exception as exc:  # noqa: BLE001 - record and re-raise typed
                    self.repo.update_document(doc_id, error=f"{type(exc).__name__}: {exc}", error_stage=stage)
                    raise IngestionError(doc_id, stage, exc) from exc
                self.repo.update_document(doc_id, status=stage, error=None, error_stage=None)
        finally:
            self.repo.unlock(doc_id)
        return self._doc(doc_id)

    def resume_pending(self) -> list[str]:
        """Finish documents whose processing was interrupted. Returns ids that were resumed."""
        resumed = []
        for doc in self.repo.documents_needing_resume():
            try:
                self.run_to(doc["doc_id"], doc["target_stage"])
                resumed.append(doc["doc_id"])
            except Exception as exc:  # noqa: BLE001
                log.warning("resume of %s failed: %s", doc["doc_id"], exc)
        return resumed

    def page_views(self, doc_id: str) -> list[dict[str, Any]]:
        """Pages as the LLM sees them: low-confidence OCR words marked [[?word]]."""
        out = []
        for p in self.repo.get_pages(doc_id):
            marked = " ".join(mark_word(w, c, self.s.ocr_conf_threshold) for w, c in p["words"])
            out.append({**p, "marked_text": marked if p["method"] == "ocr" else p["text"]})
        return out

    # ------------------------------------------------------------------ stages
    def _doc(self, doc_id: str) -> dict[str, Any]:
        doc = self.repo.get_document(doc_id)
        if not doc:
            raise KeyError(f"unknown document {doc_id}")
        return doc

    def _stage_triaged(self, doc_id: str) -> None:
        doc = self._doc(doc_id)
        pages = triage_file(Path(doc["path"]), self.s.text_layer_min_chars)
        self.repo.update_document(doc_id, page_count=len(pages),
                                  triage_json=json.dumps([p.to_dict() for p in pages]))

    def _stage_extracted(self, doc_id: str) -> None:
        doc = self._doc(doc_id)
        path = Path(doc["path"])
        done = {p["page"] for p in self.repo.get_pages(doc_id)}
        for t in json.loads(doc["triage_json"]):
            page = t["page"]
            if page in done:  # resumability within the stage
                continue
            if t["method"] == "native":
                text = native_page_text(path, page)
                words: list[tuple[str, float | None]] = [(w, None) for w in text.split()]
                self.repo.upsert_page(doc_id, page, "native", text, None, words)
            else:
                image = Image.open(path) if is_image(path) else render_pdf_page(path, page, self.s.ocr_dpi)
                result = self.ocr.ocr(image)
                words = [(w.text, w.conf) for w in result.words]
                self.repo.upsert_page(doc_id, page, "ocr", result.text, result.mean_conf, words)

    def _stage_chunked(self, doc_id: str) -> None:
        doc = self._doc(doc_id)
        chunks = chunk_document(doc_id, self.repo.get_pages(doc_id), size=self.s.chunk_words,
                                overlap=self.s.chunk_overlap, conf_threshold=self.s.ocr_conf_threshold,
                                title=doc["title"] or doc["filename"])
        self.repo.replace_chunks(doc_id, chunks)

    def _stage_indexed(self, doc_id: str) -> None:
        doc = self._doc(doc_id)
        chunks = self.repo.get_chunks(doc_id)
        if not chunks:
            return
        payload = [{
            "chunk_id": c["chunk_id"], "doc_id": doc_id, "doc_name": doc["title"] or doc["filename"],
            "page": c["page"], "text": c["text"], "section": c.get("section") or "", "scope": doc["scope"],
            "doc_type": (doc["doc_type"] or "").lower(), "product": (doc["product"] or "").lower(),
            "verification": c["verification"], "min_conf": c["min_conf"],
        } for c in chunks]
        vectors = self.embedder.embed_documents([context_text(c) for c in payload])
        self.index.upsert(payload, vectors)

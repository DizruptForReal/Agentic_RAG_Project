"""process_document: run an uploaded document through OCR (only if needed) and either
extract specific fields (extract_only) or index it for search (index)."""
from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, Field, model_validator

from app.agent.models import Observation, ObsStatus
from app.db.repo import Repo
from app.ingestion.extract import FieldExtractor
from app.ingestion.pipeline import DocumentBusy, IngestionError, IngestionPipeline
from app.retrieval.base import SearchUnavailable
from app.tools.base import ToolContext, ToolError


class ProcessArgs(BaseModel):
    doc_id: str
    mode: Literal["extract_only", "index"] = "extract_only"
    fields: list[str] = Field(default_factory=list)
    scope: Literal["session", "global"] = "session"

    @model_validator(mode="after")
    def _fields_for_extract(self):
        if self.mode == "extract_only" and not self.fields:
            raise ValueError("extract_only requires at least one field name")
        return self


class ProcessDocumentTool:
    name: ClassVar[str] = "process_document"
    description: ClassVar[str] = (
        "Process an attached document. mode=extract_only pulls the named fields (no indexing; best for "
        "one-off evidence such as a receipt). mode=index chunks+embeds it so `search` can find it "
        "(for this investigation only; uploads are never shared with other investigations by this tool). "
        "OCR runs only on pages without a "
        "usable text layer. Low-confidence OCR values come back UNVERIFIED.")
    Args: ClassVar[type[BaseModel]] = ProcessArgs

    def __init__(self, repo: Repo, pipeline: IngestionPipeline, extractor: FieldExtractor):
        self.repo, self.pipeline, self.extractor = repo, pipeline, extractor

    def run(self, args: ProcessArgs, ctx: ToolContext) -> Observation:
        doc = self.repo.get_document(args.doc_id)
        if not doc:
            raise ToolError(f"unknown doc_id '{args.doc_id}'. Attached: {ctx.attachments}", retryable=False)
        if doc["scope"] != "global" and args.doc_id not in ctx.attachments:
            raise ToolError(f"document {args.doc_id} is not attached to this investigation", retryable=False)
        try:
            doc = self.pipeline.ensure_extracted(args.doc_id)
            if args.mode == "index":
                # Scope only ever widens: a corpus (global) document stays global, and an upload that belongs to
                # another investigation keeps its scope (it is visible here as an attachment anyway).
                # The agent never publishes a user's upload to everyone: an upload stays visible only to the
                # investigations it is attached to (the promote endpoint is the deliberate way to add it to the
                # shared corpus). scope="global" therefore only applies to documents that are already global.
                if doc["scope"] == "global" or doc["scope"].startswith("session:"):
                    scope = None
                else:
                    scope = f"session:{ctx.investigation_id}"
                doc = self.pipeline.ensure_indexed(args.doc_id, scope=scope)
        except DocumentBusy as exc:
            raise ToolError(str(exc), retryable=True) from exc
        except IngestionError as exc:
            raise ToolError(str(exc), retryable=exc.retryable) from exc
        except SearchUnavailable as exc:
            raise ToolError(str(exc), retryable=True) from exc

        pages = self.pipeline.page_views(args.doc_id)
        ocr_pages = [p["page"] for p in pages if p["method"] == "ocr"]
        base = {"doc_id": args.doc_id, "status": doc["status"], "ocr_pages": ocr_pages}
        if args.mode == "index":
            n = len(self.repo.get_chunks(args.doc_id))
            return Observation(tool=self.name, status=ObsStatus.OK, data={**base, "chunks": n},
                               summary=f"indexed {n} chunks (OCR on pages {ocr_pages or 'none'}); "
                                       f"use search to retrieve them")
        evidence, not_found, errors = self.extractor.extract(doc, pages, args.fields)
        if errors and not evidence:
            raise ToolError(f"extraction failed: {errors}", retryable=False)
        unverified = [e.field_name for e in evidence if e.verification.value == "UNVERIFIED"]
        summary = (f"extracted {len(evidence)} field(s); not found: {not_found or 'none'}; "
                   f"UNVERIFIED: {unverified or 'none'}; OCR pages: {ocr_pages or 'none'}")
        status = ObsStatus.OK if evidence else ObsStatus.EMPTY
        return Observation(tool=self.name, status=status, summary=summary, evidence=evidence,
                           data={**base, "not_found": not_found, "unverified": unverified})

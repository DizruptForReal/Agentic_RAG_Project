"""API response shapes."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.agent.progress import BOARD
from app.agent.models import FinalAnswer, InvestigationState, LedgerEntry, StepRecord, SubQuestion


class UploadInfo(BaseModel):
    doc_id: str
    duplicate: bool
    status: str
    ocr_pages: list[int] = []


class InvestigationView(BaseModel):
    id: str
    question: str
    status: str
    outcome: str | None
    answer: FinalAnswer | None
    pending_question: str | None
    sub_questions: list[SubQuestion]
    attachments: list[str]
    ledger: list[LedgerEntry]
    steps: list[StepRecord]
    notes: list[str]
    error: str | None
    activity: dict[str, Any] | None = None     # what it is doing right now (while RUNNING)
    timings: dict[str, int] = {}
    upload: UploadInfo | None = None

    @classmethod
    def of(cls, s: InvestigationState, upload: dict[str, Any] | None = None) -> "InvestigationView":
        return cls(id=s.id, question=s.question, status=s.status.value,
                   outcome=s.final.outcome.value if s.final else None, answer=s.final,
                   pending_question=s.pending_question, sub_questions=s.sub_questions,
                   attachments=s.attachments, ledger=s.ledger, steps=s.steps, notes=s.notes, error=s.error,
                   activity=BOARD.get(s.id) if s.status.value == "RUNNING" else None, timings=s.timings,
                   upload=UploadInfo(**upload) if upload else None)


class DocumentView(BaseModel):
    doc_id: str
    filename: str
    title: str | None
    doc_type: str | None
    product: str | None
    effective_date: str | None
    supersedes: str | None
    scope: str
    status: str
    target_stage: str | None
    processing: bool
    error: str | None
    error_stage: str | None
    page_count: int | None
    duplicate: bool = False

    @classmethod
    def of(cls, d: dict[str, Any], duplicate: bool = False) -> "DocumentView":
        return cls(**{k: d.get(k) for k in cls.model_fields if k not in ("duplicate", "processing")},
                   processing=bool(d.get("processing")), duplicate=duplicate)

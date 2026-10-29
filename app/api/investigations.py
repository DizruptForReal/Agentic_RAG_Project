"""Investigation endpoints: start, continue (answer / upload), inspect, list.

`wait=true` (default) runs the investigation inside the request and returns the finished state.
`wait=false` returns immediately (status RUNNING) and runs in the background; poll GET /investigations/{id}
to watch steps appear. The web UI uses wait=false.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from app.agent.loop import InvestigationBusy
from app.agent.models import InvStatus
from app.api.deps import get_container
from app.api.schemas import InvestigationView
from app.container import Container
from pathlib import Path

from app.ingestion.pipeline import SUPPORTED_EXT, IngestionError

log = logging.getLogger(__name__)
router = APIRouter(prefix="/investigations", tags=["investigations"])


def _attach(c: Container, state, file: UploadFile | None):
    if file is None or not file.filename:
        return None
    content = file.file.read()
    if not content:
        raise HTTPException(400, "empty file")
    try:
        return c.investigator.attach_upload(state, file.filename or "upload", content)
    except ValueError as exc:
        raise HTTPException(415, str(exc)) from exc
    except IngestionError as exc:
        raise HTTPException(422, str(exc)) from exc


def _background_run(c: Container, inv_id: str) -> None:
    try:
        c.investigator.run(inv_id, already_claimed=True)
    except Exception:  # noqa: BLE001 - run() already records failures; never crash the worker
        log.exception("background investigation %s failed", inv_id)


def _execute(c: Container, inv_id: str, wait: bool, background: BackgroundTasks, upload):
    try:
        c.investigator.claim(inv_id)
    except InvestigationBusy as exc:
        raise HTTPException(409, str(exc)) from exc
    if wait:
        return InvestigationView.of(c.investigator.run(inv_id, already_claimed=True), upload)
    background.add_task(_background_run, c, inv_id)
    return InvestigationView.of(c.investigator.get(inv_id), upload)


@router.get("")
def list_investigations(limit: int = 50, c: Container = Depends(get_container)):
    """Most recently updated investigations (id, question, status, outcome)."""
    return c.repo.list_investigations(limit)


@router.post("", response_model=InvestigationView)
def start_investigation(background: BackgroundTasks, question: str = Form(..., min_length=5),
                        file: UploadFile | None = File(None), wait: bool = Form(True),
                        c: Container = Depends(get_container)):
    """Start an investigation, optionally with a document attached from the start."""
    if file is not None and file.filename and Path(file.filename).suffix.lower() not in SUPPORTED_EXT:
        # Reject before creating anything, so a bad upload does not leave an empty investigation behind.
        raise HTTPException(415, f"unsupported file type '{Path(file.filename).suffix.lower()}'. "
                                 f"Supported: {sorted(SUPPORTED_EXT)}")
    state = c.investigator.create(question)
    upload = _attach(c, state, file)
    return _execute(c, state.id, wait, background, upload)


@router.post("/{inv_id}/continue", response_model=InvestigationView)
def continue_investigation(inv_id: str, background: BackgroundTasks, message: str | None = Form(None),
                           file: UploadFile | None = File(None), wait: bool = Form(True),
                           c: Container = Depends(get_container)):
    """Answer the agent's question and/or upload a document, then resume. Also resumes INTERRUPTED,
    FAILED, or COMPLETED-but-unanswered investigations with new input."""
    state = c.investigator.get(inv_id)
    if not state:
        raise HTTPException(404, "investigation not found")
    if state.status == InvStatus.RUNNING:
        raise HTTPException(409, "investigation is running")
    has_file = file is not None and bool(file.filename)
    if state.status == InvStatus.NEEDS_INPUT and not message and not has_file:
        raise HTTPException(400, f"the agent is waiting for input: {state.pending_question}")
    if message:
        c.investigator.add_message(state, message)
    upload = _attach(c, state, file)
    return _execute(c, inv_id, wait, background, upload)


@router.get("/{inv_id}", response_model=InvestigationView)
def get_investigation(inv_id: str, c: Container = Depends(get_container)):
    state = c.investigator.get(inv_id)
    if not state:
        raise HTTPException(404, "investigation not found")
    return InvestigationView.of(state)

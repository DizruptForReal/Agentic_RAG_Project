"""Core data models shared by the agent, tools and API."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- planner I/O
class ActionType(str, Enum):
    SEARCH = "search"
    GET_METADATA = "get_metadata"
    PROCESS_DOCUMENT = "process_document"
    ASK_USER = "ask_user"
    FINALIZE = "finalize"


class SubQStatus(str, Enum):
    OPEN = "OPEN"                  # not investigated yet
    SUPPORTED = "SUPPORTED"        # verified evidence answers it
    MISSING = "MISSING"            # fact not in the resources; the user could supply it
    CONFLICTING = "CONFLICTING"    # sources disagree and it is not resolved
    UNVERIFIED = "UNVERIFIED"      # only uncertain (OCR) evidence
    INSUFFICIENT = "INSUFFICIENT"  # searched properly, evidence weak or irrelevant


class SubQuestion(BaseModel):
    id: str
    text: str
    status: SubQStatus = SubQStatus.OPEN
    evidence_ids: list[str] = Field(default_factory=list)
    note: str = ""
    required: bool = True   # False = useful context only; an unsupported optional part becomes a caveat


class Action(BaseModel):
    type: ActionType
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class DraftAnswer(BaseModel):
    """Answer written by the planner when it finalizes (saves a separate composer call)."""
    text: str
    citations: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class PlannerStep(BaseModel):
    sub_questions: list[SubQuestion]
    action: Action
    robust_to_unverified: bool = False
    conflict_resolution: str = ""
    answer: DraftAnswer | None = None


# ---------------------------------------------------------------- evidence
class Verification(str, Enum):
    VERIFIED = "VERIFIED"          # read from a document (text layer or confident OCR)
    UNVERIFIED = "UNVERIFIED"      # low-confidence OCR or ungrounded extraction
    USER_STATED = "USER_STATED"    # told to the agent by the user; usable, but not confirmed by a document


class EvidenceKind(str, Enum):
    CHUNK = "chunk"          # retrieved passage
    FIELD = "field"          # value extracted from an uploaded document
    METADATA = "metadata"    # record from SQLite
    USER = "user"            # something the user told the agent in a reply


class EvidenceItem(BaseModel):
    kind: EvidenceKind
    doc_id: str
    doc_name: str
    text: str
    page: int | None = None
    chunk_id: str | None = None
    field_name: str | None = None
    verification: Verification = Verification.VERIFIED
    ocr_confidence: float | None = None
    retrieval_score: float | None = None
    rerank_score: float | None = None
    retrieval_rank: int | None = None    # best position this passage reached in any search

    def dedupe_key(self) -> str:
        if self.kind == EvidenceKind.CHUNK:
            return f"chunk:{self.chunk_id}"
        if self.kind == EvidenceKind.USER:
            return f"user:{self.text}"
        if self.kind == EvidenceKind.FIELD:
            return f"field:{self.doc_id}:{self.field_name}:{self.text}"
        return f"meta:{self.doc_id}"


class LedgerEntry(EvidenceItem):
    id: str


# ---------------------------------------------------------------- observations
class ObsStatus(str, Enum):
    OK = "OK"
    EMPTY = "EMPTY"      # tool worked, found nothing
    ERROR = "ERROR"      # tool failed
    BLOCKED = "BLOCKED"  # guard refused the call (repeat)
    PAUSED = "PAUSED"    # waiting for the user


class Observation(BaseModel):
    tool: str
    status: ObsStatus
    summary: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    retryable: bool = False
    attempts: int = 1
    evidence: list[EvidenceItem] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- outcomes
class Outcome(str, Enum):
    ANSWERED = "ANSWERED"
    ANSWERED_WITH_CAVEATS = "ANSWERED_WITH_CAVEATS"
    MISSING_INFO = "MISSING_INFO"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICT = "CONFLICT"
    TOOL_FAILURE = "TOOL_FAILURE"


class InvStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    NEEDS_INPUT = "NEEDS_INPUT"
    COMPLETED = "COMPLETED"
    INTERRUPTED = "INTERRUPTED"
    FAILED = "FAILED"


class Citation(BaseModel):
    evidence_id: str
    doc_id: str
    doc_name: str
    page: int | None = None
    verification: Verification
    excerpt: str


class FinalAnswer(BaseModel):
    outcome: Outcome
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class StepRecord(BaseModel):
    idx: int
    planner_ok: bool
    action: Action | None = None
    observation: Observation | None = None
    notes: list[str] = Field(default_factory=list)
    at: str = Field(default_factory=utcnow)
    label: str = ""                 # plain-language description for the UI
    auto: bool = False              # taken by the system, not chosen by the planner
    llm_ms: int | None = None       # time in the planner call (excluding rate-limit waits)
    wait_ms: int | None = None      # time waiting for the provider's rate limit
    tool_ms: int | None = None      # time running the tool
    model: str | None = None        # LLM that produced this step (differs from the primary after a failover)
    tokens_in: int | None = None    # prompt tokens the model read for this step (local models report them)
    tokens_out: int | None = None   # tokens it generated, including hidden reasoning


class InvestigationState(BaseModel):
    id: str
    question: str
    status: InvStatus = InvStatus.CREATED
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    ledger: list[LedgerEntry] = Field(default_factory=list)
    attachments: list[str] = Field(default_factory=list)
    user_messages: list[str] = Field(default_factory=list)
    steps: list[StepRecord] = Field(default_factory=list)
    call_hashes: list[str] = Field(default_factory=list)
    pending_question: str | None = None
    blocked_repeats: int = 0
    consecutive_failures: int = 0
    failed_tools: list[str] = Field(default_factory=list)  # tools whose last call errored
    tool_errors: dict[str, str] = Field(default_factory=dict)  # last error message per failed tool
    auto_read: list[str] = Field(default_factory=list)     # attachments the system read before planning
    finalize_rejections: int = 0                           # answers sent back by the pre-finalize review
    field_checks: list[str] = Field(default_factory=list)  # "<sub-question>:<doc>" already verified at field level
    claim_verified: list[str] = Field(default_factory=list)  # flagged OCR passages whose cited claim read clearly
    answer_problems: list[str] = Field(default_factory=list)  # review findings still open at finalize

    @property
    def unrecovered_failure(self) -> bool:
        return bool(self.failed_tools)

    robust_to_unverified: bool = False
    conflict_resolution: str = ""
    draft_answer: DraftAnswer | None = None
    unknown_entities: list[str] = Field(default_factory=list)  # identifiers in the question no document mentions
    timings: dict[str, int] = Field(default_factory=dict)   # compose_ms, total_ms of the last run
    notes: list[str] = Field(default_factory=list)
    final: FinalAnswer | None = None
    error: str | None = None
    created_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)

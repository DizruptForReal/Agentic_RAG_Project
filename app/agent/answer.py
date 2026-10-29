"""Final response: LLM-written answer only when the outcome allows one, verified against the ledger.
Refusals are composed deterministically from the guard's reasons (nothing to hallucinate)."""
from __future__ import annotations

import re

from typing import Any

from pydantic import BaseModel, Field

from app.agent.ledger import get_entries
from app.agent.models import (Citation, DraftAnswer, FinalAnswer, InvestigationState, LedgerEntry, Outcome, SubQStatus,
                              Verification)
from app.ingestion.chunker import clean_text
from app.llm.base import LLMClient, call_structured


class ComposedAnswer(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


COMPOSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "description": "Direct answer with inline [e#] citations."},
        "citations": {"type": "array", "items": {"type": "string"}},
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "citations"],
}

COMPOSE_SYSTEM = """You write the final answer of an investigation using ONLY the evidence provided.
- Cite evidence inline as [e#] and list every id you used in `citations`.
- Explain the reasoning chain briefly (e.g. which document version applies and why, date arithmetic).
- Anything marked UNVERIFIED must be described as unverified, never as fact.
- Do not add facts that are not in the evidence. Be concise (under 200 words)."""

REFUSAL_LEAD = {
    Outcome.MISSING_INFO: "A reliable answer cannot be given yet: required information is missing.",
    Outcome.INSUFFICIENT_EVIDENCE: "A reliable answer cannot be given: the available evidence does not "
                                   "adequately support one.",
    Outcome.CONFLICT: "A reliable answer cannot be given: the sources conflict and the conflict could not "
                      "be resolved.",
    Outcome.TOOL_FAILURE: "The investigation could not be completed because a tool failed. This is not "
                          "evidence that the answer is unavailable; retry once the service recovers.",
}


def _describe_failure(tool: str, error: str) -> str:
    """One readable line per failed component (the raw error is kept in the investigation's steps)."""
    e = error.lower()
    if tool == "search":
        if "unreachable" in e or "connection" in e:
            return "Search: Elasticsearch could not be reached (is it running? `docker compose up -d elasticsearch`)."
        return f"Search: {error[:200]}"
    if tool == "llm":
        if "rate limit" in e:
            when = re.search(r"available again in ([^;.]+)", error)
            kind = "daily quota" if "daily" in e or "per day" in e or "tpd" in e else "rate limit"
            return (f"Language model: the provider's {kind} is used up on every configured model"
                    + (f"; available again in {when.group(1)}." if when else "; try again later."))
        if "unusable output" in e:
            return ("Language model: it returned output that could not be used even after a correction attempt "
                    "(not a lack of evidence). Try again; with a local model, run python -m scripts.check_llm.")
        if "configuration" in e or "401" in e or "403" in e:
            return "Language model: the provider rejected the request (check LLM_API_KEY / LLM_MODEL)."
        return f"Language model: {error[:200]}"
    return f"{tool}: {error[:200]}"


def verify_citations(ids: list[str], state: InvestigationState) -> tuple[list[LedgerEntry], list[str]]:
    """Citation verifier: only ledger ids survive, in order, without duplicates."""
    seen, ordered = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    return get_entries(state, ordered)


def to_citation(e: LedgerEntry) -> Citation:
    return Citation(evidence_id=e.id, doc_id=e.doc_id, doc_name=e.doc_name, page=e.page,
                    verification=e.verification, excerpt=clean_text(e.text)[:240])


def _unverified_caveats(entries: list[LedgerEntry], skip: set[str] | None = None) -> list[str]:
    out = []
    skip = skip or set()
    # A page whose claim was re-read word by word and verified: the page-wide passage flag is not a caveat.
    checked = {(e.doc_id, e.page) for e in entries if e.kind.value == "field" and e.verification == Verification.VERIFIED}
    for e in entries:
        if e.kind.value == "chunk" and e.verification == Verification.UNVERIFIED and (
                (e.doc_id, e.page) in checked or e.id in skip):
            continue
        if e.verification == Verification.USER_STATED:
            out.append(f"[{e.id}] is information you provided ({clean_text(e.text)[:120]}); it is used as given "
                       f"and is not confirmed by any document.")
            continue
        if e.verification == Verification.UNVERIFIED:
            conf = f", OCR confidence {e.ocr_confidence:.0f}" if e.ocr_confidence is not None else ""
            out.append(f"[{e.id}] from {e.doc_name} (page {e.page}) is UNVERIFIED{conf}; it was not treated "
                       f"as a verified fact.")
    return out


def compose(llm: LLMClient, state: InvestigationState, outcome: Outcome, reasons: list[str],
            unresolved: list[str], draft: DraftAnswer | None = None) -> FinalAnswer:
    relevant_ids = [i for q in state.sub_questions for i in q.evidence_ids]
    if outcome not in (Outcome.ANSWERED, Outcome.ANSWERED_WITH_CAVEATS):
        entries, _ = verify_citations(relevant_ids, state)
        found = [f"{q.id} ({q.status.value}): {q.text}" + (f" - {q.note}" if q.note else "")
                 for q in state.sub_questions]
        lead = REFUSAL_LEAD[outcome]
        if outcome == Outcome.TOOL_FAILURE and state.tool_errors:
            lead += "\n\nWhat failed:\n- " + "\n- ".join(_describe_failure(t, e) for t, e in state.tool_errors.items()
                                                       if t in state.failed_tools)
        if outcome == Outcome.INSUFFICIENT_EVIDENCE and state.unknown_entities:
            lead = (f"The documents do not cover {', '.join(state.unknown_entities)}: no available document "
                    f"mentions it, so no reliable answer can be given. Information about other models was "
                    f"deliberately not applied.")
        elif outcome == Outcome.INSUFFICIENT_EVIDENCE and not relevant_ids:
            lead = "The available documents do not contain information that answers this question."
        text = lead + ("\n\nWhat was established:\n- " + "\n- ".join(found) if found else "")
        if state.pending_question:
            text += f"\n\nTo continue: {state.pending_question}"
        return FinalAnswer(outcome=outcome, answer=text, citations=[to_citation(e) for e in entries],
                           caveats=_unverified_caveats(entries, set(state.claim_verified)), unresolved=unresolved,
                           reasons=reasons)

    notes: list[str] = []
    import re

    if draft is not None and draft.text.strip() and (
            draft.citations or re.search(r"(?:\[|【)\s*e\d+\s*(?:\]|】)", draft.text)):
        # The planner already wrote the answer while finalizing: no extra LLM call. It is checked by the
        # same citation verifier below as a composer-written answer.
        answer, cited_ids, caveats = draft.text, draft.citations, list(draft.caveats)
        notes.append("answer written by the planner in the same call (no separate composer call)")
        return _verified(state, outcome, reasons, unresolved, answer, cited_ids, caveats, notes)

    evidence = [e for e in state.ledger if e.id in set(relevant_ids)] or state.ledger
    user = (f"Question: {state.question}\nOutcome decided by the system: {outcome.value}\n"
            f"Reasons: {reasons}\nConflict resolution: {state.conflict_resolution or 'n/a'}\n"
            f"Sub-questions: {[q.model_dump(mode='json') for q in state.sub_questions]}\n\nEvidence:\n" +
            "\n".join(f"[{e.id}] ({e.doc_name}, p.{e.page}, {e.verification.value}) {e.text}" for e in evidence))
    res = call_structured(llm, purpose="compose_answer", system=COMPOSE_SYSTEM, user=user,
                          schema=COMPOSE_SCHEMA, model_cls=ComposedAnswer)
    if res.value is None:
        # Deterministic fallback: still a sourced answer, just not prose-polished.
        notes.append(f"answer composer failed ({res.errors[-1:] if res.errors else 'unknown'}); fallback used")
        answer = "Answer assembled from verified sub-questions:\n- " + "\n- ".join(
            f"{q.text}: {q.note or q.status.value} [{', '.join(q.evidence_ids)}]" for q in state.sub_questions)
        cited_ids, caveats = relevant_ids, []
    else:
        answer, cited_ids, caveats = res.value.answer, res.value.citations, list(res.value.caveats)
    return _verified(state, outcome, reasons, unresolved, answer, cited_ids, caveats, notes)


def _verified(state: InvestigationState, outcome: Outcome, reasons: list[str], unresolved: list[str],
              answer: str, cited_ids: list[str], caveats: list[str], notes: list[str]) -> FinalAnswer:
    """Citation verifier + coverage check shared by planner-written and composer-written answers."""
    import re

    inline_ids = re.findall(r"(?:\[|【)\s*(e\d+)\s*(?:\]|】)", answer)
    cited_ids = list(cited_ids) + [i for i in inline_ids if i not in cited_ids]
    entries, invalid = verify_citations(cited_ids, state)
    if invalid:
        notes.append(f"citation verifier removed ids not in the ledger: {invalid}")
    # every answered sub-question must be backed by at least one citation
    cited = {e.id for e in entries}
    for q in state.sub_questions:
        if q.status in (SubQStatus.SUPPORTED, SubQStatus.UNVERIFIED) and q.evidence_ids and not cited & set(q.evidence_ids):
            extra, _ = get_entries(state, q.evidence_ids[:1])
            entries += extra
            cited |= {e.id for e in extra}
            notes.append(f"added citation for {q.id}")
    if not entries:
        return FinalAnswer(outcome=Outcome.INSUFFICIENT_EVIDENCE, unresolved=unresolved,
                           answer=REFUSAL_LEAD[Outcome.INSUFFICIENT_EVIDENCE] + " (no verifiable citations)",
                           reasons=reasons + notes + ["answer had no valid citations"])
    caveats += [c for c in _unverified_caveats(entries, set(state.claim_verified)) if c not in caveats]
    return FinalAnswer(outcome=outcome, answer=answer, citations=[to_citation(e) for e in entries],
                       caveats=caveats, unresolved=unresolved, reasons=reasons + notes)

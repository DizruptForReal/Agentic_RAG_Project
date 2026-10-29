"""Deterministic checks that sit around the LLM: repeat detection, sub-question validation,
and the final outcome decision. Nothing here calls a model."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from app.agent.ledger import get_entries
from app.tools.search_kb import is_relevant
from app.agent.models import (Action, EvidenceKind, InvestigationState, ObsStatus, Outcome, SubQStatus,
                              SubQuestion, Verification)


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.lower().split())
    if isinstance(value, list):
        return sorted((_normalize(v) for v in value), key=json.dumps)
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in sorted(value.items()) if v not in (None, [], "")}
    return value


def call_hash(action: Action) -> str:
    payload = json.dumps({"t": action.type.value, "a": _normalize(action.args)}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def repeat_check(state: InvestigationState, action: Action) -> str | None:
    """Return a reason to block, or None. Successful calls may not be repeated (nothing new to learn);
    a failing call may be retried once more by the planner, then it is blocked too."""
    h = call_hash(action)
    previous = [s.split(":")[1] for s in state.call_hashes if s.startswith(h + ":")]
    if any(st in (ObsStatus.OK.value, ObsStatus.EMPTY.value) for st in previous):
        return "identical call already succeeded; its results are in the ledger - choose a different action"
    if previous.count(ObsStatus.ERROR.value) >= 2:
        return "identical call has failed twice; do not retry it"
    return None


INFO_ACTIONS = {"search", "get_metadata", "process_document"}


# Best reranker score below which nothing retrieved relates to the question. Calibrated on the held-out set
# (python -m scripts.eval_retrieval): every in-scope question's best passage scored >= 0.011, out-of-scope ones
# without a model name <= 0.0054 (the weather question: 0.0001). Re-check on a new corpus.
OFF_TOPIC_SCORE = 0.008


def best_relevance(state: InvestigationState) -> float | None:
    """Highest reranker score of any retrieved passage so far (None if nothing was reranked)."""
    scores = [e.rerank_score for e in state.ledger if e.kind == EvidenceKind.CHUNK and e.rerank_score is not None]
    return max(scores) if scores else None


def off_topic(state: InvestigationState, threshold: float = OFF_TOPIC_SCORE) -> bool:
    """Nothing in the documents relates to the question: every retrieved passage is essentially unrelated, and
    the user has not attached anything. Asking the user then makes no sense - there is no
    missing fact that would let these documents answer - so the outcome is INSUFFICIENT_EVIDENCE."""
    if state.attachments:          # an uploaded document can make the question answerable
        return False
    best = best_relevance(state)
    return best is not None and best < threshold


def ask_user_check(state: InvestigationState, min_actions: int, max_questions: int) -> str | None:
    """None = allowed; "limit" = question budget used up (finalize); otherwise a reason to block.
    Forces the agent to investigate the resources before asking the user, and stops it from asking
    repeatedly instead of concluding."""
    asked = sum(1 for s in state.steps if s.observation and s.observation.status == ObsStatus.PAUSED)
    if asked >= max_questions:
        return "limit"
    if off_topic(state):
        return (f"ask_user blocked: nothing in the documents relates to this question (best relevance "
                f"{best_relevance(state):.4f}); no fact from the user would let them answer it. Finalize - the "
                f"outcome is INSUFFICIENT_EVIDENCE.")
    done = sum(1 for s in state.steps if s.action and s.action.type.value in INFO_ACTIONS
               and s.observation and s.observation.status in (ObsStatus.OK, ObsStatus.EMPTY))
    if done < min_actions:
        return (f"ask_user blocked: only {done} investigation action(s) so far (minimum {min_actions}). "
                "Investigate the resources first: search specifically for each OPEN/MISSING sub-question, "
                "call get_metadata to check document versions, and use process_document(extract_only) on "
                "corpus documents whose relevant chunk is UNVERIFIED (e.g. scanned pages).")
    return None


def record_call(state: InvestigationState, action: Action, status: ObsStatus) -> None:
    state.call_hashes.append(f"{call_hash(action)}:{status.value}")


def validate_subquestions(subqs: list[SubQuestion], state: InvestigationState, rerank_floor: float
                          ) -> tuple[list[SubQuestion], list[str]]:
    """Check every status the planner claims against the ledger. Returns (fixed sub-questions, notes)."""
    notes: list[str] = []
    fixed: list[SubQuestion] = []
    for sq in subqs:
        sq = sq.model_copy(deep=True)
        entries, unknown = get_entries(state, sq.evidence_ids)
        if unknown:
            notes.append(f"{sq.id}: dropped unknown evidence ids {unknown}")
            sq.evidence_ids = [e.id for e in entries]
        if sq.status == SubQStatus.SUPPORTED:
            if not entries:
                sq.status, why = SubQStatus.INSUFFICIENT, "claimed SUPPORTED without ledger evidence"
                notes.append(f"{sq.id}: {why}")
                sq.note = f"{sq.note} [guard: {why}]".strip()
            else:
                strong = [e for e in entries
                          if e.verification in (Verification.VERIFIED, Verification.USER_STATED) and not (
                              e.kind == EvidenceKind.CHUNK
                              and not is_relevant(e.rerank_score, e.retrieval_rank, rerank_floor))]
                if not strong:
                    ok, detail, ids = _claim_on_flagged_passages(sq, entries, rerank_floor)
                    if ok:
                        # A passage is flagged when ANY word in it is uncertain; the words this claim rests on
                        # were read clearly, so the claim stands (word-level rule, app/ingestion/ocr_verify.py).
                        state.claim_verified += [i for i in ids if i not in state.claim_verified]
                        tag = f"[verified word by word: {detail}]"
                        if tag not in sq.note:
                            sq.note = f"{sq.note} {tag}".strip()
                        fixed.append(sq)
                        continue
                    if any(e.verification == Verification.UNVERIFIED for e in entries):
                        sq.status, why = SubQStatus.UNVERIFIED, "all cited evidence is unverified OCR"
                    else:
                        sq.status, why = SubQStatus.INSUFFICIENT, ("all cited passages have low relevance "
                                                                   f"(reranker < {rerank_floor} and not in the top 3)")
                    notes.append(f"{sq.id}: downgraded - {why}")
                    sq.note = f"{sq.note} [guard: {why}]".strip()
        elif sq.status == SubQStatus.UNVERIFIED and not entries:
            sq.status = SubQStatus.INSUFFICIENT
            notes.append(f"{sq.id}: UNVERIFIED without evidence -> INSUFFICIENT")
        elif sq.status == SubQStatus.UNVERIFIED and not any(
                e.verification in (Verification.VERIFIED, Verification.USER_STATED) for e in entries):
            # The planner saw [[?..]] markers in the passage; if the claim's own words are clear, it is supported.
            ok, detail, ids = _claim_on_flagged_passages(sq, entries, rerank_floor)
            if ok:
                state.claim_verified += [i for i in ids if i not in state.claim_verified]
                sq.status = SubQStatus.SUPPORTED
                sq.note = f"{sq.note} [verified word by word: {detail}]".strip()
                notes.append(f"{sq.id}: UNVERIFIED -> SUPPORTED, the claim's words were read clearly")
        elif sq.status == SubQStatus.CONFLICTING and len(entries) < 2:
            sq.status = SubQStatus.INSUFFICIENT
            notes.append(f"{sq.id}: CONFLICTING needs evidence for both sides -> INSUFFICIENT")
        fixed.append(sq)
    return fixed, notes


def _claim_on_flagged_passages(sq: SubQuestion, entries, rerank_floor: float) -> tuple[bool, str, list[str]]:
    """Can the sub-question's own claim (its note) be checked word by word on the flagged OCR passages it cites?
    Only the note is used: it states the value ("Schedule B: COMMERCIAL", "bought 05-03-2025"); the question text
    is not a claim. Fails if any value in the claim is missing from, or uncertain on, every cited passage."""
    from app.ingestion.ocr_verify import check_claim

    claim = re.sub(r"\[guard:[^\]]*\]|\[verified[^\]]*\]", " ", sq.note or "").strip()
    passages = [e for e in entries if e.kind == EvidenceKind.CHUNK and e.verification == Verification.UNVERIFIED
                and is_relevant(e.rerank_score, e.retrieval_rank, rerank_floor)]
    if not claim or not passages:
        return False, "", []
    for e in passages:
        ok, detail = check_claim(claim, e.text)
        if ok:
            return True, f"{e.id}: {detail}", [e.id]
    return False, "", []


IDENT_RE = re.compile(r"\b[A-Za-z]{1,6}-?\d{2,6}(?:-[A-Za-z0-9]+)*\b")


def extract_identifiers(text: str) -> list[str]:
    """Model numbers, serials and references named in a question (X500, AQ-X200-55871, ...)."""
    seen: list[str] = []
    for m in IDENT_RE.findall(text):
        ident = m.upper()
        if ident not in seen and not re.fullmatch(r"E\d+", ident):   # e12 = an evidence id, not an entity
            seen.append(ident)
    return seen


def decide_outcome(state: InvestigationState) -> tuple[Outcome, list[str], list[str]]:
    """Outcome from validated sub-question statuses, plus the unknown-entity rule.
    Returns (outcome, reasons, unresolved sub-questions)."""
    outcome, reasons, unresolved = _outcome_from_subquestions(state)
    if outcome == Outcome.MISSING_INFO and off_topic(state):
        return (Outcome.INSUFFICIENT_EVIDENCE,
                [f"nothing in the documents relates to the question (best relevance {best_relevance(state):.4f}); "
                 f"it is outside what they cover, not a fact the user could supply"], unresolved)
    if state.unknown_entities and outcome in (Outcome.ANSWERED, Outcome.ANSWERED_WITH_CAVEATS):
        names = ", ".join(state.unknown_entities)
        return (Outcome.INSUFFICIENT_EVIDENCE,
                [f"the question is about {names}, which no available document mentions, so the documents "
                 f"cannot answer it (evidence about other models was not applied)"], unresolved)
    return outcome, reasons, unresolved


def _partial_answer_ok(state: InvestigationState, required: list[SubQuestion], open_like: list[SubQuestion]) -> bool:
    """A partial answer is allowed when the planner wrote an answer that cites verified ledger evidence, at
    least one required part is SUPPORTED, and the gaps are only 'not found' (never a conflict, missing user
    input or uncertain OCR, which keep their own outcomes). The gaps are reported as caveats."""
    draft = state.draft_answer
    if draft is None or not draft.text.strip():
        return False
    supported = [q for q in required if q.status == SubQStatus.SUPPORTED]
    if not supported or len(open_like) >= len(required):
        return False
    inline = re.findall(r"(?:\[|【)\s*(e\d+)\s*(?:\]|】)", draft.text)
    entries, _ = get_entries(state, list(draft.citations) + inline)
    return any(e.verification == Verification.VERIFIED for e in entries)


def _outcome_from_subquestions(state: InvestigationState) -> tuple[Outcome, list[str], list[str]]:
    sqs = state.sub_questions
    if not sqs:
        if state.unrecovered_failure:
            return Outcome.TOOL_FAILURE, [f"failed before any evidence was gathered: {state.failed_tools}"], []
        return Outcome.INSUFFICIENT_EVIDENCE, ["the question was never decomposed into answerable parts"], []
    # Optional sub-questions (context that would be nice to have) never block an answer: they become caveats.
    required = [q for q in sqs if q.required] or list(sqs)
    optional_gaps = [q for q in sqs if q not in required and q.status != SubQStatus.SUPPORTED]
    by = {s: [q for q in required if q.status == s] for s in SubQStatus}
    unresolved = [f"{q.id}: {q.text} ({q.status.value})" for q in sqs if q.status != SubQStatus.SUPPORTED]
    reasons: list[str] = []

    if by[SubQStatus.CONFLICTING]:
        reasons += [f"unresolved conflict for {q.id}: {q.note or q.text}" for q in by[SubQStatus.CONFLICTING]]
        return Outcome.CONFLICT, reasons, unresolved
    gaps = by[SubQStatus.MISSING] + by[SubQStatus.OPEN] + by[SubQStatus.INSUFFICIENT]
    if gaps and not by[SubQStatus.UNVERIFIED] and not state.unrecovered_failure and \
            _partial_answer_ok(state, required, gaps):
        # The planner chose to answer (instead of asking or giving up) and cited verified evidence for the
        # parts it could support: return that answer and state exactly what is missing.
        reasons.append("answered from the parts of the question that verified evidence supports")
        reasons += [f"not established: {q.text}" + (f" - {q.note}" if q.note else "") for q in gaps]
        return Outcome.ANSWERED_WITH_CAVEATS, reasons, unresolved
    if by[SubQStatus.MISSING]:
        reasons += [f"missing information for {q.id}: {q.note or q.text}" for q in by[SubQStatus.MISSING]]
        return Outcome.MISSING_INFO, reasons, unresolved
    open_like = by[SubQStatus.OPEN] + by[SubQStatus.INSUFFICIENT]
    if open_like:
        if state.unrecovered_failure:
            reasons.append("a required tool failed and evidence could not be gathered")
            reasons += [f"{q.id} not established: {q.note or q.text}" for q in open_like]
            return Outcome.TOOL_FAILURE, reasons, unresolved
        reasons += [f"insufficient evidence for {q.id}: {q.note or q.text}" for q in open_like]
        return Outcome.INSUFFICIENT_EVIDENCE, reasons, unresolved
    if by[SubQStatus.UNVERIFIED]:
        if state.robust_to_unverified:
            reasons.append("the verdict does not change under the uncertain OCR values")
            return Outcome.ANSWERED_WITH_CAVEATS, reasons, unresolved
        reasons += [f"{q.id} depends on unverified OCR evidence that could change the verdict: {q.note or q.text}"
                    for q in by[SubQStatus.UNVERIFIED]]
        return Outcome.INSUFFICIENT_EVIDENCE, reasons, unresolved
    user_based = [q for q in required if q.status == SubQStatus.SUPPORTED and q.evidence_ids and all(
        e.verification == Verification.USER_STATED for e in get_entries(state, q.evidence_ids)[0])]
    if user_based:
        reasons.append("part of the answer relies on information you provided, which no document confirms: "
                       + "; ".join(q.text for q in user_based))
        if optional_gaps:
            reasons += [f"not established (optional): {q.text}" for q in optional_gaps]
        return Outcome.ANSWERED_WITH_CAVEATS, reasons, unresolved
    if optional_gaps:
        reasons.append("the required parts of the question are supported by verified evidence")
        reasons += [f"not established (optional): {q.text}" + (f" - {q.note}" if q.note else "") for q in optional_gaps]
        return Outcome.ANSWERED_WITH_CAVEATS, reasons, unresolved
    return Outcome.ANSWERED, ["all sub-questions supported by verified evidence"], []


# ---------------------------------------------------------------- pre-finalize review
# Two checks run in code on every answer the planner wants to give. Both are about the answer as a whole,
# not about one question: values must come from the evidence, and amended documents must not be used
# without their amendments.
_MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                       "september", "october", "november", "december"], start=1)}
_MON = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|" \
       r"jun|jul|aug|sep|sept|oct|nov|dec)"
_DATE_PATTERNS = [
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), ("y", "m", "d")),
    (re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\b"), ("d", "m", "y")),
    (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + _MON + r"\.?,?\s+(\d{4})\b", re.I), ("d", "M", "y")),
    (re.compile(r"\b" + _MON + r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", re.I), ("M", "d", "y")),
]


def _month(name: str) -> int:
    name = name.lower().rstrip(".")
    return next(v for k, v in _MONTHS.items() if k.startswith(name[:3]))


def find_dates(text: str) -> list[tuple[str, tuple[int, int, int]]]:
    """Calendar dates in free text as (surface form, (y, m, d)); formats seen in documents and answers."""
    import unicodedata

    text = unicodedata.normalize("NFKC", text)
    out, taken = [], []
    for rx, order in _DATE_PATTERNS:
        for m in rx.finditer(text):
            if any(a < m.end() and m.start() < b for a, b in taken):
                continue
            parts = dict(zip(order, m.groups()))
            try:
                y, mo, d = int(parts["y"]), (_month(parts["M"]) if "M" in parts else int(parts["m"])), int(parts["d"])
            except (StopIteration, ValueError):
                continue
            if 1 <= mo <= 12 and 1 <= d <= 31:
                out.append((m.group(0), (y, mo, d)))
                taken.append((m.start(), m.end()))
    return out


_DURATION = re.compile(r"\b(\d{1,3})\)?\s*(month|year|day|business day)s?\b", re.I)


def _durations(texts: list[str]) -> tuple[set[int], set[int]]:
    """Periods named in the sources: whole months (years x 12) and days."""
    months, days = set(), set()
    for t in texts:
        for n, unit in _DURATION.findall(t):
            n = int(n)
            unit = unit.lower()
            if unit == "year":
                months.add(n * 12)
            elif unit == "month":
                months.add(n)
            else:
                days.add(n)
    return months, days


def _derived(v: tuple[int, int, int], known: set[tuple[int, int, int]], months: set[int], days: set[int]) -> bool:
    """Is v a known date shifted by a period the sources name (e.g. purchase date + 12 months)?"""
    import calendar
    from datetime import date, timedelta

    try:
        target = date(*v)
    except ValueError:
        return False
    for k in known:
        try:
            base = date(*k)
        except ValueError:
            continue
        for n in months:
            for sign in (1, -1):
                total = base.year * 12 + base.month - 1 + sign * n
                y, m = divmod(total, 12)
                d = min(base.day, calendar.monthrange(y, m + 1)[1])
                shifted = date(y, m + 1, d)
                if abs((shifted - target).days) <= 1:
                    return True
        for n in days:
            if any(abs((base + timedelta(days=sign * n) - target).days) <= 1 for sign in (1, -1)):
                return True
    return False


def _date_sources(state: InvestigationState) -> tuple[list[str], list[str]]:
    """(texts whose dates are facts about the case, document-metadata texts). A document's effective date says
    when the document applies; it is not a date of any event involving a unit."""
    facts = [state.question, *state.user_messages,
             *(e.text for e in state.ledger if e.kind != EvidenceKind.METADATA)]
    meta = [e.text for e in state.ledger if e.kind == EvidenceKind.METADATA]
    return facts, meta


def ungrounded_dates(state: InvestigationState, answer: str) -> list[str]:
    """Dates stated in the answer that are neither in the question, the user's replies or the evidence, nor
    derivable from a fact date by a period the sources name (e.g. "warranty ended 20 March 2026" from a
    purchase date + "12 months"). A document's effective date (metadata) may be quoted, but deadlines computed
    from it are not grounded: it is not a purchase or delivery date. Catches invented dates and dates taken in
    the wrong role."""
    facts, meta = _date_sources(state)
    fact_dates = {v for src in facts for _, v in find_dates(src)}
    known = fact_dates | {v for src in meta for _, v in find_dates(src)}
    months, days = _durations(facts + meta)
    return [s for s, v in find_dates(answer) if v not in known and not _derived(v, fact_dates, months, days)]


def metadata_derived_dates(state: InvestigationState, answer: str) -> list[str]:
    """The subset of ungrounded dates that are a document's effective date shifted by a period - the typical
    mistake of treating a contract's effective date as the purchase date."""
    facts, meta = _date_sources(state)
    meta_dates = {v for src in meta for _, v in find_dates(src)}
    months, days = _durations(facts + meta)
    return [s for s in ungrounded_dates(state, answer)
            if any(_derived(v, meta_dates, months, days) for t, v in find_dates(s))]


_DOC_CODE = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9.]*[A-Z0-9])+\b")


def amending_documents(doc_ids: set[str], consulted: set[str], corpus: list[dict]) -> list[tuple[dict, dict]]:
    """(amended, amending) pairs: a document in doc_ids that another, not yet consulted, document amends or
    supersedes according to its `supersedes` metadata (matched on document codes such as AQF-WP-1.0)."""
    by_id = {d["doc_id"]: d for d in corpus}
    pairs = []
    for doc_id in doc_ids:
        d = by_id.get(doc_id)
        if not d:
            continue
        codes = set(_DOC_CODE.findall(d.get("title") or ""))
        for other in corpus:
            sup = other.get("supersedes") or ""
            if other["doc_id"] == doc_id or other["doc_id"] in consulted or not sup:
                continue
            if any(re.search(re.escape(c) + r"(?![-\w])", sup) for c in codes):
                pairs.append((d, other))
    return pairs


def unconsulted_amendments(state: InvestigationState, corpus: list[dict]) -> list[str]:
    """Documents relied on by the answer that another document amends or supersedes, where that other
    document was never consulted (no evidence from it in the ledger). Uses the `supersedes` metadata."""
    cited = {i for q in state.sub_questions if q.status == SubQStatus.SUPPORTED for i in q.evidence_ids}
    if state.draft_answer:
        cited |= set(state.draft_answer.citations)
        cited |= set(re.findall(r"(?:\[|【)\s*(e\d+)\s*(?:\]|】)", state.draft_answer.text))
    used = {e.doc_id for e in state.ledger if e.id in cited}
    consulted = {e.doc_id for e in state.ledger}
    return [f"'{other.get('title')}' amends '{d.get('title')}' (supersedes: {other.get('supersedes')}) but was not "
            f"consulted; check whether it changes the answer (doc_id {other['doc_id']})."
            for d, other in amending_documents(used, consulted, corpus)]


def review_answer(state: InvestigationState, corpus: list[dict]) -> list[str]:
    problems = []
    if state.draft_answer and state.draft_answer.text.strip():
        bad = ungrounded_dates(state, state.draft_answer.text)
        from_meta = metadata_derived_dates(state, state.draft_answer.text)
        if from_meta:
            problems.append(f"the answer computes {', '.join(from_meta)} from a document's effective date (catalog "
                            f"metadata). That is when the document applies, not a purchase or delivery date. If the "
                            f"date the question depends on is not in the evidence, it is MISSING: ask the user for it "
                            f"(e.g. the sales receipt) instead of answering")
            bad = [b for b in bad if b not in from_meta]
        if bad:
            problems.append(f"the answer states date(s) {', '.join(bad)} that appear in no evidence, user reply or "
                            f"the question; take every date from cited evidence, in the role the source gives it")
    problems += unconsulted_amendments(state, corpus)
    return problems

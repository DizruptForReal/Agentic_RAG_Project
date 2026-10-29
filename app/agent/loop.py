"""The investigation loop. One generic loop for every question - no per-scenario branches.

    plan (LLM) -> validate statuses (code) -> guard (repeat/budget) -> run tool -> ledger -> persist
    ... until finalize / pause / budget, then outcome (code) -> answer (LLM, verified) or refusal (code)

State is saved after every step, so a crash loses at most the step in flight and the investigation
can be resumed (status INTERRUPTED -> continue).
"""
from __future__ import annotations

import json
import re
import logging
import time
import uuid
from typing import Any

from app.agent import progress
from app.agent.answer import compose
from app.agent.guards import (ask_user_check, decide_outcome, extract_identifiers, record_call, repeat_check,
                               amending_documents, review_answer, ungrounded_dates,
                               validate_subquestions)
from app.agent.ledger import add_evidence
from app.agent.models import (Action, ActionType, EvidenceItem, EvidenceKind, InvestigationState, InvStatus,
                              Observation, ObsStatus, Outcome, SubQStatus, SubQuestion, Verification,
                              StepRecord)
from app.agent.planner import Planner
from app.config import Settings
from app.db.repo import Repo
from app.ingestion.pipeline import IngestionPipeline
from app.llm.base import LLMClient
from app.tools.base import ToolContext, ToolRegistry

log = logging.getLogger(__name__)
MAX_FINALIZE_REJECTIONS = 2


class InvestigationBusy(Exception):
    pass


class Investigator:
    def __init__(self, *, repo: Repo, llm: LLMClient, registry: ToolRegistry, pipeline: IngestionPipeline,
                 settings: Settings):
        self.repo, self.llm, self.registry, self.pipeline, self.s = repo, llm, registry, pipeline, settings
        self.planner = Planner(llm, settings.max_steps)

    # ------------------------------------------------------------------ lifecycle
    def create(self, question: str) -> InvestigationState:
        state = InvestigationState(id="inv_" + uuid.uuid4().hex[:12], question=question.strip())
        self.repo.save_investigation(state)
        return state

    def get(self, inv_id: str) -> InvestigationState | None:
        return self.repo.get_investigation(inv_id)

    def attach_upload(self, state: InvestigationState, filename: str, content: bytes) -> dict[str, Any]:
        doc, duplicate = self.pipeline.register(filename, content, scope=f"session:{state.id}")
        if doc["doc_id"] not in state.attachments:
            state.attachments.append(doc["doc_id"])
        ocr_pages = [t["page"] for t in json.loads(doc["triage_json"] or "[]") if t["method"] == "ocr"]
        msg = (f"User uploaded '{filename}' as doc_id {doc['doc_id']} ({doc['page_count']} page(s); "
               f"OCR needed on pages {ocr_pages or 'none'}).")
        if duplicate:
            msg += f" This is a duplicate of an existing document (status {doc['status']}); not reprocessed."
        state.user_messages.append(msg)
        self.repo.save_investigation(state)
        return {"doc_id": doc["doc_id"], "duplicate": duplicate, "status": doc["status"], "ocr_pages": ocr_pages}

    def add_message(self, state: InvestigationState, message: str) -> None:
        """A typed reply is recorded as citable evidence of kind 'user' (USER_STATED): the agent may rely on
        it, and the answer says that it rests on the user's statement rather than on a document."""
        text = message.strip()
        state.user_messages.append(f"User: {text}")
        context = f" (in reply to: {state.pending_question})" if state.pending_question else ""
        add_evidence(state, [EvidenceItem(kind=EvidenceKind.USER, doc_id="user", doc_name="Your reply",
                                          text=f"User stated: {text}{context}",
                                          verification=Verification.USER_STATED)])
        self.repo.save_investigation(state)

    def mark_interrupted(self) -> list[str]:
        """Startup recovery: anything still RUNNING belonged to a process that died."""
        ids = self.repo.investigation_ids_with_status(InvStatus.RUNNING.value)
        for inv_id in ids:
            state = self.repo.get_investigation(inv_id)
            if state:
                state.status = InvStatus.INTERRUPTED
                state.notes.append("interrupted by a restart; resume with /continue")
                self.repo.save_investigation(state)
        return ids

    # ------------------------------------------------------------------ the loop
    def claim(self, inv_id: str) -> None:
        if not self.repo.claim_investigation(inv_id):
            raise InvestigationBusy(f"{inv_id} is already running")

    def run(self, inv_id: str, already_claimed: bool = False) -> InvestigationState:
        if not already_claimed and not self.repo.claim_investigation(inv_id):
            raise InvestigationBusy(f"{inv_id} is already running")
        state = self.repo.get_investigation(inv_id)
        state.status, state.final, state.pending_question, state.error = InvStatus.RUNNING, None, None, None
        state.blocked_repeats = state.consecutive_failures = 0
        state.finalize_rejections, state.answer_problems = 0, []
        state.draft_answer = None
        self.repo.save_investigation(state)
        started = time.monotonic()
        with progress.reporting(inv_id):
            try:
                result = self._loop(state)
            except Exception as exc:  # noqa: BLE001 - never leave an investigation stuck in RUNNING
                log.exception("investigation %s crashed", inv_id)
                state.status, state.error = InvStatus.FAILED, f"{type(exc).__name__}: {exc}"
                result = state
        result.timings["total_ms"] = int((time.monotonic() - started) * 1000)
        self.repo.save_investigation(result)
        return result

    # ------------------------------------------------------------------ steps
    def _execute(self, state: InvestigationState, action: Action, *, notes: list[str], auto: bool = False,
                 llm_ms: int | None = None, wait_ms: int | None = None) -> Observation:
        """Guard, run and record one tool action."""
        idx = len(state.steps) + 1
        label = self._describe(state, action)
        block = None
        if action.type == ActionType.ASK_USER:
            block = ask_user_check(state, self.s.min_actions_before_ask, self.s.max_user_questions)
        block = block or repeat_check(state, action)
        t0 = time.monotonic()
        if block:
            obs = Observation(tool=action.type.value, status=ObsStatus.BLOCKED, error=block)
            state.blocked_repeats += 1
            label = f"Blocked: {label}"
        else:
            progress.report(label, "work")
            obs = self.registry.run(action, ToolContext(state.id, list(state.attachments)))
            record_call(state, action, obs.status)
        tool_ms = int((time.monotonic() - t0) * 1000)
        obs.evidence_ids = add_evidence(state, obs.evidence)
        obs.evidence = []  # stored once, in the ledger
        self._track_failures(state, action.type.value, obs)
        state.steps.append(StepRecord(idx=idx, planner_ok=True, action=action, observation=obs, notes=notes,
                                      label=label, auto=auto, llm_ms=llm_ms, wait_ms=wait_ms, tool_ms=tool_ms,
                                      model=None if auto else getattr(self, "_last_model", None),
                                      **({} if auto else self._token_fields())))
        self.repo.save_investigation(state)
        return obs

    def _plan(self, state: InvestigationState):
        first = not state.sub_questions
        progress.report("Understanding the question and planning" if first else "Deciding the next step", "llm")
        progress.take_wait()
        t0 = time.monotonic()
        if hasattr(self.llm, "last_model"):
            self.llm.last_model = None
        t_in, t_out = getattr(self.llm, "tokens_in", None), getattr(self.llm, "tokens_out", None)
        res = self.planner.plan(state, self._attachments_view(state), self._corpus_view(state))
        self._last_model = getattr(self.llm, "last_model", None)
        self._last_tokens = ((self.llm.tokens_in - t_in, self.llm.tokens_out - t_out)
                             if t_in is not None and t_out is not None else (None, None))
        wait_s = progress.take_wait()
        total = time.monotonic() - t0
        return res, int(max(0.0, total - wait_s) * 1000), int(wait_s * 1000)

    def _token_fields(self) -> dict[str, int | None]:
        t_in, t_out = getattr(self, "_last_tokens", (None, None))
        return {"tokens_in": t_in, "tokens_out": t_out}

    def _check_entities(self, state: InvestigationState) -> None:
        """Deterministic scope check: model numbers / references in the question that appear in no document."""
        scopes = ["global", f"session:{state.id}"]
        unknown = [i for i in extract_identifiers(state.question)
                   if not self.repo.text_mentions(i, scopes, state.attachments)]
        if unknown and unknown != state.unknown_entities:
            state.notes.append(f"The question mentions {', '.join(unknown)}, which does not appear in any available "
                               f"document. Evidence about other models must not be applied to it.")
        found = [i for i in state.unknown_entities if i not in unknown]
        if found:
            # Typically an identifier that only an uploaded document contains (a serial on a receipt or card):
            # it was unknown until that document was read.
            state.notes.append(f"{', '.join(found)} now found in an attached document; no longer treated as unknown.")
        state.unknown_entities = unknown

    def _loop(self, state: InvestigationState) -> InvestigationState:
        run_start = len(state.steps)
        self._check_entities(state)
        self._read_new_attachments(state)
        if self.s.initial_search and not any(s.auto and s.action and s.action.type == ActionType.SEARCH
                                             and not (s.action.args or {}).get("doc_ids") for s in state.steps):
            # Fast path: the first search is obvious, so it does not need an LLM call. The planner then
            # starts with evidence and can often answer in its first turn.
            obs = self._execute(state, Action(type=ActionType.SEARCH, reason="initial retrieval for the question",
                                              args={"query": state.question[:300], "top_k": self.s.initial_top_k}),
                                notes=[], auto=True)
            if obs.status == ObsStatus.ERROR and not state.attachments and not state.ledger:
                # Search is down and there is nothing else to work from: report it now instead of spending
                # planner calls (and rate-limit budget) on an investigation that cannot gather evidence.
                state.notes.append("search unavailable and no other evidence; stopped early")
                return self._finalize(state)
            self._fetch_amendments(state)
        while True:
            if len(state.steps) - run_start >= self.s.max_steps:
                state.notes.append(f"step budget ({self.s.max_steps}) exhausted; finalizing with current evidence")
                break
            idx = len(state.steps) + 1
            res, llm_ms, wait_ms = self._plan(state)
            if res.value is None:
                if res.llm_failed:
                    state.failed_tools = sorted(set(state.failed_tools) | {"llm"})
                    state.tool_errors["llm"] = str(res.errors[-1] if res.errors else "")[:400]
                elif not state.sub_questions:
                    # Nothing was decomposed: the model, not the documents, is why there is no answer.
                    state.failed_tools = sorted(set(state.failed_tools) | {"llm"})
                    state.tool_errors["llm"] = f"unusable output: {str(res.errors[-1] if res.errors else '')[:300]}"
                note = f"planner output unusable after {res.attempts} attempt(s): {res.errors[-1:]}; finalizing safely"
                state.steps.append(StepRecord(idx=idx, planner_ok=False, notes=[note], label="Planner failed",
                                              llm_ms=llm_ms, wait_ms=wait_ms, tokens_in=self._last_tokens[0],
                                              tokens_out=self._last_tokens[1]))
                state.notes.append(note)
                self.repo.save_investigation(state)
                break
            plan = res.value
            notes = ["planner output repaired after a validation error"] if res.attempts > 1 else []
            if not plan.sub_questions and not state.sub_questions:
                # Some models skip the decomposition for simple questions. The question itself becomes the
                # single sub-question; if the model finalized with a cited answer, those citations are its
                # evidence (and still go through the same validation as any other claim).
                cited = list(plan.answer.citations) if plan.answer and plan.action.type == ActionType.FINALIZE else []
                plan.sub_questions = [SubQuestion(id="q1", text=state.question,
                                                  status=SubQStatus.SUPPORTED if cited else SubQStatus.OPEN,
                                                  evidence_ids=cited)]
                notes.append("planner returned no sub-questions; using the question itself")
            if plan.sub_questions:
                state.sub_questions, guard_notes = validate_subquestions(plan.sub_questions, state, self.s.relevance_floor)
                notes += guard_notes
            state.robust_to_unverified = plan.robust_to_unverified
            if plan.conflict_resolution:
                state.conflict_resolution = plan.conflict_resolution
            state.failed_tools = [t for t in state.failed_tools if t != "llm"]
            action = plan.action

            if action.type == ActionType.FINALIZE:
                state.draft_answer = plan.answer
                self._verify_ocr_claims(state)
                idx = len(state.steps) + 1        # automatic verification steps may have been added
                problems = review_answer(state, self._corpus_view(state))
                if problems and state.finalize_rejections < MAX_FINALIZE_REJECTIONS and \
                        len(state.steps) - run_start < self.s.max_steps - 1:
                    # Send the answer back once or twice with the specific problem, like a reviewer would.
                    state.finalize_rejections += 1
                    state.draft_answer = None
                    state.notes += [f"answer review: {p}" for p in problems]
                    state.steps.append(StepRecord(
                        idx=idx, planner_ok=True, action=action, notes=notes + problems, label="Answer sent back for a fix",
                        observation=Observation(tool="finalize", status=ObsStatus.BLOCKED, error="; ".join(problems)),
                        llm_ms=llm_ms, wait_ms=wait_ms, model=getattr(self, "_last_model", None)))
                    self.repo.save_investigation(state)
                    continue
                state.answer_problems = problems
                state.steps.append(StepRecord(idx=idx, planner_ok=True, action=action, notes=notes,
                                              label="Concluding the investigation", llm_ms=llm_ms, wait_ms=wait_ms,
                                              model=getattr(self, "_last_model", None), **self._token_fields()))
                self.repo.save_investigation(state)
                break

            if action.type == ActionType.ASK_USER and ask_user_check(
                    state, self.s.min_actions_before_ask, self.s.max_user_questions) == "limit":
                # Already asked the user as often as allowed: stop and explain instead of asking again.
                q = str(action.args.get("question", "")).strip()
                state.pending_question = q or state.pending_question
                note = (f"user-question limit ({self.s.max_user_questions}) reached; finalizing and "
                        f"reporting what is still needed")
                state.notes.append(note)
                state.steps.append(StepRecord(idx=idx, planner_ok=True, action=action, notes=notes + [note],
                                              label="Stopping instead of asking again", llm_ms=llm_ms,
                                              wait_ms=wait_ms))
                self.repo.save_investigation(state)
                break

            if action.type == ActionType.ASK_USER and not str(action.args.get("question") or "").strip() \
                    and len(action.reason.strip()) >= 5:
                # The model gave its reason but left the question empty: the reason says what is needed.
                action.args["question"] = action.reason.strip()
                notes.append("ask_user had no question; used the stated reason")
            obs = self._execute(state, action, notes=notes, llm_ms=llm_ms, wait_ms=wait_ms)
            if action.type == ActionType.PROCESS_DOCUMENT and state.unknown_entities and obs.status in (
                    ObsStatus.OK, ObsStatus.EMPTY):
                self._check_entities(state)          # the document's pages are extracted now
            if obs.status == ObsStatus.PAUSED:
                state.status = InvStatus.NEEDS_INPUT
                state.pending_question = obs.data.get("question")
                self.repo.save_investigation(state)
                return state
            if state.blocked_repeats >= self.s.max_blocked_repeats:
                state.notes.append("planner kept repeating blocked calls; finalizing")
                break
            if state.consecutive_failures >= self.s.max_consecutive_failures:
                state.notes.append(f"{state.consecutive_failures} consecutive tool failures; finalizing")
                break
        return self._finalize(state)

    def _describe(self, state: InvestigationState, action: Action) -> str:
        a = action.args
        if action.type == ActionType.SEARCH:
            extra = ", ".join(f"{k}={a[k]}" for k in ("product", "doc_type") if a.get(k))
            return f"Searching the knowledge base for \u201c{a.get('query', '')}\u201d" + (f" ({extra})" if extra else "")
        if action.type == ActionType.GET_METADATA:
            what = a.get("doc_type") or a.get("product") or (f"{len(a['doc_ids'])} document(s)" if a.get("doc_ids") else "")
            return "Checking document versions and metadata" + (f" ({what})" if what else "")
        if action.type == ActionType.PROCESS_DOCUMENT:
            doc = self.repo.get_document(str(a.get("doc_id", ""))) or {}
            name = doc.get("title") or doc.get("filename") or a.get("doc_id", "document")
            ocr = [t["page"] for t in json.loads(doc.get("triage_json") or "[]") if t["method"] == "ocr"]
            fields = ", ".join(a.get("fields") or [])
            pending_ocr = ocr and doc.get("status") in ("RECEIVED", "TRIAGED")
            verb = f"Scanning {name} with OCR (page {', '.join(map(str, ocr))})" if pending_ocr else f"Reading {name}"
            if a.get("mode") == "index":
                return f"{verb} and indexing it for search"
            return f"{verb}" + (f" for: {fields}" if fields else "")
        if action.type == ActionType.ASK_USER:
            return "Asking you for information"
        return action.type.value

    def _fetch_amendments(self, state: InvestigationState) -> None:
        """Version-aware retrieval: if the first search returned a document that another document amends or
        supersedes (per metadata), search the amending document too, before the first planner call."""
        consulted = {e.doc_id for e in state.ledger}
        # only documents that are actually relevant to the question (not the weakest top-ranked passages of an
        # off-topic search)
        relevant = {e.doc_id for e in state.ledger if e.kind == EvidenceKind.CHUNK and e.rerank_score is not None
                    and e.rerank_score >= self.s.relevance_floor}
        pairs = amending_documents(relevant, consulted, self._corpus_view(state))
        for amending_id in dict.fromkeys(o["doc_id"] for _, o in pairs):
            self._execute(state, Action(type=ActionType.SEARCH, reason="check the amendment of a retrieved document",
                                        args={"query": state.question[:300], "doc_ids": [amending_id], "top_k": 3}),
                          notes=[], auto=True)
            self._execute(state, Action(type=ActionType.GET_METADATA, reason="versions of the amended documents",
                                        args={"doc_ids": sorted({amending_id} | {d["doc_id"] for d, o in pairs
                                                                                if o["doc_id"] == amending_id})}),
                          notes=[], auto=True)

    def _verify_ocr_claims(self, state: InvestigationState) -> None:
        """A sub-question held back only because its passage came from a scanned page with an uncertain word
        somewhere in it: read the claim itself from that page at field level (each word of the quote must be
        read with high confidence). An unrelated smudge, such as a signature, then no longer blocks a clearly
        read value; a genuinely uncertain value (a smudged date) stays UNVERIFIED."""
        for q in state.sub_questions:
            if q.status != SubQStatus.UNVERIFIED or not q.evidence_ids:
                continue
            entries = [e for e in state.ledger if e.id in q.evidence_ids]
            ocr_docs = list(dict.fromkeys(e.doc_id for e in entries if e.kind == EvidenceKind.CHUNK
                                          and e.verification == Verification.UNVERIFIED))
            if not ocr_docs or any(e.kind == EvidenceKind.FIELD for e in entries):
                continue          # already checked at field level, or not an OCR passage
            verified_ids = []
            for doc_id in ocr_docs:
                key = f"{q.id}:{doc_id}"
                if key in state.field_checks:
                    continue
                state.field_checks.append(key)
                obs = self._execute(state, Action(
                    type=ActionType.PROCESS_DOCUMENT, reason=f"verify '{q.text}' word by word on the scanned page",
                    args={"doc_id": doc_id, "mode": "extract_only", "fields": [q.text[:120]]}), notes=[], auto=True)
                found = [e for e in state.ledger if e.id in (obs.evidence_ids or [])]
                q.evidence_ids += [e.id for e in found if e.id not in q.evidence_ids]
                verified_ids += [e.id for e in found if e.verification == Verification.VERIFIED]
            if verified_ids:
                if state.draft_answer:
                    state.draft_answer.citations += [i for i in verified_ids if i not in state.draft_answer.citations]
                q.status = SubQStatus.SUPPORTED
                q.note = (re.sub(r"\s*\[guard: all cited evidence is unverified OCR\]", "", q.note).strip()
                          + f" [verified at field level: {', '.join(verified_ids)}]").strip()
                state.notes.append(f"{q.id}: claim verified word by word on the scanned page ({', '.join(verified_ids)})")
        self.repo.save_investigation(state)

    def _read_new_attachments(self, state: InvestigationState) -> None:
        """A document the user attached to this investigation is evidence they want used, so it is read
        before the first planner call: indexed for this investigation (OCR only on pages without a text
        layer) and searched with the question. The planner can still extract specific fields from it."""
        if not self.s.auto_read_attachments:
            return
        for doc_id in [d for d in state.attachments if d not in state.auto_read]:
            state.auto_read.append(doc_id)
            obs = self._execute(state, Action(type=ActionType.PROCESS_DOCUMENT, reason="read the attached document",
                                              args={"doc_id": doc_id, "mode": "index"}), notes=[], auto=True)
            if obs.status in (ObsStatus.OK, ObsStatus.EMPTY):
                self._check_entities(state)
                self._execute(state, Action(type=ActionType.SEARCH, reason="search the attached document",
                                            args={"query": state.question[:300], "doc_ids": [doc_id], "top_k": 5}),
                              notes=[], auto=True)

    def _track_failures(self, state: InvestigationState, tool: str, obs: Observation) -> None:
        if obs.status == ObsStatus.ERROR:
            state.consecutive_failures += 1
            if obs.error and obs.error.startswith("invalid arguments"):
                return  # bad args from the model is not a backend failure
            state.tool_errors[tool] = (obs.error or "")[:400]
            state.failed_tools = sorted(set(state.failed_tools) | {tool})
        elif obs.status in (ObsStatus.OK, ObsStatus.EMPTY, ObsStatus.PAUSED):
            state.consecutive_failures = 0
            state.failed_tools = [t for t in state.failed_tools if t != tool]

    def _finalize(self, state: InvestigationState) -> InvestigationState:
        if state.unknown_entities:
            self._check_entities(state)
        outcome, reasons, unresolved = decide_outcome(state)
        if state.answer_problems and outcome == Outcome.ANSWERED:
            # The review still found problems after sending the answer back: never present it as fully supported.
            outcome = Outcome.ANSWERED_WITH_CAVEATS
            reasons.append("answer review problems remained after revision")
        reasons += [n for n in state.notes[-3:] if "budget" in n or "repeating" in n or "failures" in n
                    or "unusable" in n]
        progress.report("Writing the answer" if outcome.value.startswith("ANSWERED")
                        else "Explaining why a reliable answer is not possible yet", "llm")
        t0 = time.monotonic()
        state.final = compose(self.llm, state, outcome, reasons, unresolved, draft=state.draft_answer)
        if state.final.outcome.value.startswith("ANSWERED"):
            # Final check on the answer actually returned, whoever wrote it (planner draft, composer, or a run
            # that ended early): a date no source contains usually decides the verdict, so the answer is
            # withheld rather than shown with a caveat.
            bad = ungrounded_dates(state, state.final.answer)
            if bad:
                for q in state.sub_questions:       # do not list a claim resting on that date as established
                    if ungrounded_dates(state, f"{q.text} {q.note}"):
                        q.status = SubQStatus.INSUFFICIENT
                        q.note = f"relied on a date no source contains ({', '.join(ungrounded_dates(state, q.note))})"
                reasons = reasons + [f"the drafted answer relied on date(s) {', '.join(bad)} that no source, reply "
                                     f"or the question contains; it was withheld"]
                state.final = compose(self.llm, state, Outcome.INSUFFICIENT_EVIDENCE, reasons, unresolved)
        if state.answer_problems and state.final.outcome.value.startswith("ANSWERED"):
            state.final.caveats += [f"Review: {p}" for p in state.answer_problems]
        if state.final.outcome == Outcome.ANSWERED_WITH_CAVEATS:
            state.final.caveats += [f"Not established from the documents: {q.text}" for q in state.sub_questions
                                    if q.status.value in ("OPEN", "INSUFFICIENT", "MISSING")]
        state.timings["compose_ms"] = int((time.monotonic() - t0) * 1000)
        state.status = InvStatus.COMPLETED
        self.repo.save_investigation(state)
        return state

    def _corpus_view(self, state: InvestigationState, limit: int = 40) -> list[dict[str, Any]]:
        """Compact catalog of what the agent can consult. For large collections this would be replaced by
        a metadata search tool; the cap keeps the prompt bounded."""
        docs = self.repo.list_documents(visible_scopes=["global", f"session:{state.id}"])
        out = []
        for d in docs[:limit]:
            triage = json.loads(d["triage_json"] or "[]")
            out.append({"doc_id": d["doc_id"], "title": d["title"], "type": d["doc_type"], "product": d["product"],
                        "effective": d["effective_date"], "supersedes": d.get("supersedes"), "status": d["status"],
                        "ocr_pages": [t["page"] for t in triage if t["method"] == "ocr"]})
        return out

    def _attachments_view(self, state: InvestigationState) -> list[dict[str, Any]]:
        out = []
        for doc_id in state.attachments:
            d = self.repo.get_document(doc_id)
            if not d:
                continue
            triage = json.loads(d["triage_json"] or "[]")
            out.append({"doc_id": doc_id, "filename": d["filename"], "status": d["status"],
                        "pages": d["page_count"], "ocr_pages": [t["page"] for t in triage if t["method"] == "ocr"],
                        "error": d["error"]})
        return out

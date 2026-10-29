"""The planner: one LLM call per step that (a) maintains the sub-questions and their status and
(b) chooses the next action. Output is schema-validated, repaired once, else None (caller falls back)."""
from __future__ import annotations

import json
from typing import Any

from app.agent.models import InvestigationState, PlannerStep
from app.llm.base import LLMClient, StructuredResult, call_structured

STATUSES = ["OPEN", "SUPPORTED", "MISSING", "CONFLICTING", "UNVERIFIED", "INSUFFICIENT"]
ACTIONS = ["search", "get_metadata", "process_document", "ask_user", "finalize"]

# Typed (not free-form) so strict decoding can express it directly: the model fills named fields instead of
# writing a JSON object inside a string, which smaller models (e.g. a local gpt-oss:20b) often got wrong.
# Fields that do not apply to the chosen action are left out (or null in strict mode).
ACTION_ARGS: dict[str, Any] = {
    "type": "object",
    "description": "Arguments of the chosen action; fill only the fields that action uses.",
    "properties": {
        "query": {"type": "string", "description": "search: what to look for"},
        "doc_type": {"type": "string", "description": "search/get_metadata: filter, e.g. policy, manual, contract"},
        "product": {"type": "string", "description": "search/get_metadata: filter, e.g. X200"},
        "doc_ids": {"type": "array", "items": {"type": "string"}, "description": "search/get_metadata: restrict"},
        "top_k": {"type": "integer", "description": "search: number of results, at most 10"},
        "doc_id": {"type": "string", "description": "process_document: the document to read"},
        "mode": {"type": "string", "enum": ["extract_only", "index"], "description": "process_document"},
        "fields": {"type": "array", "items": {"type": "string"}, "description": "process_document: fields to read"},
        "scope": {"type": "string", "enum": ["session", "global"], "description": "process_document (index)"},
        "question": {"type": "string", "description": "ask_user: the question to ask the user"},
        "request_upload": {"type": "boolean", "description": "ask_user: true when a document is needed"},
    },
}

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sub_questions": {
            "type": "array",
            "description": "At least one item; a simple question is one sub-question.",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "status": {"type": "string", "enum": STATUSES},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    "note": {"type": "string"},
                    "required": {"type": "boolean",
                                 "description": "false = nice-to-have context; the answer does not depend on it"},
                },
                "required": ["id", "text", "status", "evidence_ids"],
            },
        },
        "action": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ACTIONS},
                "args": ACTION_ARGS,
                "reason": {"type": "string"},
            },
            "required": ["type", "args", "reason"],
        },
        "robust_to_unverified": {"type": "boolean"},
        "conflict_resolution": {"type": "string"},
        "answer": {
            "type": "object",
            "description": "Only with action finalize, when the evidence answers the question.",
            "properties": {
                "text": {"type": "string", "description": "Answer with inline [e#] citations."},
                "citations": {"type": "array", "items": {"type": "string"}},
                "caveats": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["text", "citations"],
        },
    },
    "required": ["sub_questions", "action"],
}

SYSTEM_PROMPT = """You plan an evidence-driven investigation. Never answer from memory. Each turn: update the
sub-questions and choose ONE action. The evidence ledger already contains results of an initial search.

ACTIONS (action.type / action.args)
- search {"query", "doc_type"?, "product"?, "doc_ids"?, "top_k"? (<=10)}: hybrid search over indexed sections.
  Policies/amendments have product "X-series"; product filters suit manuals and contracts.
- get_metadata {"doc_ids"?, "doc_type"?, "product"?}: effective dates, supersedes links, OCR pages.
- process_document {"doc_id", "mode": "extract_only"|"index", "fields"?: [..], "scope"?}: read named fields
  from an attachment or corpus document with field-level OCR verification (use it to verify one fact on a
  scanned page whose chunk is UNVERIFIED). "index" makes an upload searchable.
- ask_user {"question", "request_upload"}: LAST RESORT, ONE per investigation, only for a specific fact or
  document no corpus document can contain (e.g. a customer's own receipt). Never for facts the documents may
  hold, never to confirm or clarify what the user meant, and never when the documents do not relate to the
  question at all (then mark the sub-question INSUFFICIENT and finalize).
- finalize {}: stop. If the evidence answers the question, also fill "answer" (text with inline [e#]
  citations, citations list, caveats). Describe UNVERIFIED evidence as unverified.

Evidence kinds: chunk (document passage; "rank" = its position in the search, "rerank" = relevance 0..1,
treat top-ranked passages as relevant), field (value read from a document), metadata, user (what the user
told you in a reply - you MAY cite it to support a sub-question; the system reports it as user-provided).

SUB-QUESTION STATUS (cite ledger ids such as "e3")
OPEN not yet investigated | SUPPORTED verified evidence answers it | MISSING not in the resources, user
could supply it | CONFLICTING sources disagree, unresolved (cite both) | UNVERIFIED only low-confidence OCR
evidence ([[?word]]) | INSUFFICIENT searched properly, evidence weak/off-topic/other product.

RULES
1. Use only the sub-questions needed. A broad or simple question ("what is covered?") needs one or two.
   If the ledger already answers the question, finalize with the answer in your FIRST turn. Mark
   sub-questions that are only useful context with required=false; they never block an answer.
   Read ALL ledger evidence before saying something is not in the documents; quote figures exactly.
2. Evidence about another product/model supports nothing; re-search with a filter if decoys appear.
3. Several documents may govern (policy + amendment, contract + scanned schedule): check "corpus", consult
   each, and call get_metadata before deciding which version applies. Explain in conflict_resolution.
4. A scanned passage is marked UNVERIFIED when ANY word in it is uncertain ([[?word]]). Still cite it, and write
   the value you rely on in the sub-question note exactly as printed (e.g. "Schedule B: COMMERCIAL",
   "Date: 05-03-2025"): the system checks those words one by one, so an unrelated smudge does not block a
   clearly printed value. UNVERIFIED is never a verified fact. Set robust_to_unverified=true only if every plausible reading of
   the uncertain value gives the same verdict.
5. A tool ERROR is not absence of evidence. Identical successful calls are blocked; do not repeat them.
6. Check scope: a policy only covers the models it names. If the question's product/model is not named in
   any document (see system_notes), mark INSUFFICIENT and finalize; never apply another model's terms.
7. After the user has answered or uploaded, if a needed value is still UNVERIFIED or missing: finalize.
8. Finalize when all sub-questions are SUPPORTED or nothing else can change a status. Never guess.
9. Vocabulary: users phrase things differently from documents. If the evidence for a sub-question is weak
   (rerank < 0.03 and rank > 3) or off-topic, search again using the terms a policy or manual would use
   (e.g. formal synonyms) before concluding it is not covered.
10. Do not add sub-questions about conditions the user did not ask about (e.g. whether a claim was filed in
   time); mention such conditions in the answer instead.
11. If part of a request cannot be answered (e.g. one product has no data), still finalize with an answer for
   the parts you can support, and say clearly which part is not in the documents. Never leave it unanswered
   just because one part is missing. Tables in the answer may use Markdown.
12. Use each value in the role its source gives it: a contract's or lease's start or effective date is not a
   purchase date, and catalog fields ("corpus") are not evidence - cite ledger evidence for every fact.
   Attachments have already been read and searched once (see the ledger); use process_document with
   extract_only to read specific fields from them.
13. Be brief: sub-question text under 20 words, notes under 15 words, reason one short sentence. Re-send
   sub-questions unchanged when nothing changed. Long output is slow and does not improve the result.
"""


FULL_TEXT_CHARS = 1600        # a whole section (chunks are <= ~220 words)
SHORT_TEXT_CHARS = 160
FULL_TEXT_BUDGET = 10000      # total characters of full evidence per prompt (~2.5k tokens)


def _brief(entry, full: bool) -> dict[str, Any]:
    limit = FULL_TEXT_CHARS if full else SHORT_TEXT_CHARS
    d = {"id": entry.id, "kind": entry.kind.value, "doc": entry.doc_name, "page": entry.page,
         "v": entry.verification.value, "text": entry.text[:limit] + ("..." if len(entry.text) > limit else "")}
    if entry.rerank_score is not None:
        d["rerank"] = round(entry.rerank_score, 3)
    if entry.retrieval_rank is not None:
        d["rank"] = entry.retrieval_rank
    if entry.ocr_confidence is not None:
        d["ocr_conf"] = round(entry.ocr_confidence)
    return d


def _full_text_ids(state: InvestigationState) -> set[str]:
    """Which evidence the planner sees in full. Facts often sit at the end of a section (e.g. the claim
    turnaround times), so relevant evidence must never be truncated; only low-relevance evidence is
    abbreviated once the budget is used. Priority: cited by a sub-question, non-chunk evidence (fields,
    metadata), then by relevance score."""
    cited = {i for q in state.sub_questions for i in q.evidence_ids}

    def priority(e):
        return (e.id not in cited, e.kind.value == "chunk", -(e.rerank_score or 0.0))

    full, used = set(), 0
    for e in sorted(state.ledger, key=priority):
        size = min(len(e.text), FULL_TEXT_CHARS)
        if e.id in cited or e.kind.value != "chunk" or used + size <= FULL_TEXT_BUDGET:
            full.add(e.id)
            used += size
    return full


def build_view(state: InvestigationState, attachments: list[dict[str, Any]], max_steps: int,
               corpus: list[dict[str, Any]] | None = None) -> str:
    """Compact state for the planner. The most relevant evidence is shown in full (see _full_text_ids)."""
    full = _full_text_ids(state)
    history = []
    for s in state.steps[-6:]:
        o = s.observation
        item: dict[str, Any] = {"step": s.idx, "action": s.action.type.value if s.action else "fallback"}
        if s.action and s.action.args:
            item["args"] = s.action.args
        if o:
            item["result"] = o.status.value
            item["summary"] = (o.summary or o.error or "")[:220]
            item["evidence"] = o.evidence_ids
        if s.notes:
            item["notes"] = s.notes
        history.append(item)
    view = {
        "question": state.question,
        "corpus": corpus or [],
        "sub_questions": [q.model_dump(mode="json") for q in state.sub_questions],
        "evidence_ledger": [_brief(e, e.id in full) for e in state.ledger],
        "attachments": attachments,
        "user_messages": state.user_messages,
        "history": history,
        "system_notes": state.notes[-5:],
        "steps_left": max_steps - len(state.steps),
    }
    return "Current investigation state (JSON):\n" + json.dumps(view, separators=(",", ":"), default=str) + \
        "\n\nReturn the updated sub_questions and the single next action."


class Planner:
    def __init__(self, llm: LLMClient, max_steps: int):
        self.llm, self.max_steps = llm, max_steps

    def plan(self, state: InvestigationState, attachments: list[dict[str, Any]],
             corpus: list[dict[str, Any]] | None = None) -> StructuredResult:
        return call_structured(self.llm, purpose="plan_step", system=SYSTEM_PROMPT,
                               user=build_view(state, attachments, self.max_steps, corpus),
                               schema=PLAN_SCHEMA, model_cls=PlannerStep)

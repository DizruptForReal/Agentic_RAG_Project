"""Outcome decision, citation verifier and refusal composition (no I/O)."""
from __future__ import annotations

from app.agent.answer import compose, verify_citations
from app.agent.guards import call_hash, decide_outcome, validate_subquestions
from app.agent.ledger import add_evidence
from app.agent.models import (Action, ActionType, EvidenceItem, EvidenceKind, InvestigationState, Outcome,
                              SubQStatus, SubQuestion, Verification)
from tests.fakes import ScriptedLLM


def state_with(*items: EvidenceItem) -> InvestigationState:
    s = InvestigationState(id="inv_t", question="q?")
    add_evidence(s, list(items))
    return s


def chunk(text, score=0.9, ver=Verification.VERIFIED, cid=None):
    return EvidenceItem(kind=EvidenceKind.CHUNK, doc_id="d_1", doc_name="Doc", page=2, chunk_id=cid or text[:8],
                        text=text, rerank_score=score, verification=ver)


def test_ledger_dedupes_and_keeps_best_score():
    s = state_with(chunk("same text", 0.3, cid="c1"))
    ids = add_evidence(s, [chunk("same text", 0.8, cid="c1"), chunk("other", cid="c2")])
    assert ids == ["e1", "e2"] and s.ledger[0].rerank_score == 0.8


def test_verify_citations_drops_unknown_and_duplicates():
    s = state_with(chunk("a", cid="c1"), chunk("b", cid="c2"))
    found, unknown = verify_citations(["e2", "e9", "e2", "e1"], s)
    assert [e.id for e in found] == ["e2", "e1"] and unknown == ["e9"]


def test_outcome_priority():
    s = state_with(chunk("a", cid="c1"), chunk("b", cid="c2"))
    s.sub_questions = [SubQuestion(id="q1", text="a", status=SubQStatus.MISSING),
                       SubQuestion(id="q2", text="b", status=SubQStatus.CONFLICTING, evidence_ids=["e1", "e2"])]
    assert decide_outcome(s)[0] == Outcome.CONFLICT
    s.sub_questions[1].status = SubQStatus.SUPPORTED
    assert decide_outcome(s)[0] == Outcome.MISSING_INFO
    s.sub_questions[0].status = SubQStatus.INSUFFICIENT
    assert decide_outcome(s)[0] == Outcome.INSUFFICIENT_EVIDENCE
    s.failed_tools = ["search"]
    assert decide_outcome(s)[0] == Outcome.TOOL_FAILURE
    s.failed_tools, s.sub_questions[0].status = [], SubQStatus.UNVERIFIED
    assert decide_outcome(s)[0] == Outcome.INSUFFICIENT_EVIDENCE
    s.robust_to_unverified = True
    assert decide_outcome(s)[0] == Outcome.ANSWERED_WITH_CAVEATS
    s.sub_questions[0].status = SubQStatus.SUPPORTED
    assert decide_outcome(s)[0] == Outcome.ANSWERED


def test_supported_downgraded_when_all_evidence_unverified():
    s = state_with(chunk("COMMERC1AL", ver=Verification.UNVERIFIED, cid="c1"))
    fixed, notes = validate_subquestions(
        [SubQuestion(id="q1", text="usage", status=SubQStatus.SUPPORTED, evidence_ids=["e1"])], s, 0.2)
    assert fixed[0].status == SubQStatus.UNVERIFIED and notes


def test_call_hash_normalizes_case_whitespace_and_order():
    a = Action(type=ActionType.SEARCH, args={"query": "Warranty  Period", "doc_ids": ["b", "a"]})
    b = Action(type=ActionType.SEARCH, args={"doc_ids": ["a", "b"], "query": "warranty period", "top_k": None})
    assert call_hash(a) == call_hash(b)


def test_refusal_is_composed_without_llm():
    s = state_with(chunk("a", cid="c1"))
    s.sub_questions = [SubQuestion(id="q1", text="purchase date", status=SubQStatus.MISSING, note="no receipt")]
    llm = ScriptedLLM()
    final = compose(llm, s, Outcome.MISSING_INFO, ["missing information for q1"], ["q1"])
    assert final.outcome == Outcome.MISSING_INFO and "missing" in final.answer.lower()
    assert llm.calls == []


def test_answer_without_valid_citations_becomes_refusal():
    s = state_with(chunk("a", cid="c1"))
    s.sub_questions = [SubQuestion(id="q1", text="x", status=SubQStatus.SUPPORTED, evidence_ids=[])]
    llm = ScriptedLLM({"compose_answer": [{"answer": "Yes [e7]", "citations": ["e7"]}]})
    final = compose(llm, s, Outcome.ANSWERED, [], [])
    assert final.outcome == Outcome.INSUFFICIENT_EVIDENCE

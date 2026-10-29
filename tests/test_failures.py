"""Failure handling in the investigation loop: malformed output, tool failures, repeats, budgets,
interrupted runs, and the difference between missing / failed / insufficient / conflicting."""
from __future__ import annotations

from app.agent.models import (Action, ActionType, InvStatus, InvestigationState, Observation, ObsStatus,
                              StepRecord)
from tests.conftest import SAMPLES
from tests.fakes import llm_down, parse_view


def plan(subqs, action, args=None, **extra):
    return {"sub_questions": subqs, "action": {"type": action, "args": args or {}, "reason": "test"}, **extra}


Q_OPEN = [{"id": "q1", "text": "warranty period", "status": "OPEN", "evidence_ids": []}]


def _run(container, question="What is the warranty period for the X200?"):
    state = container.investigator.create(question)
    return container.investigator.run(state.id)


# ------------------------------------------------------------------ malformed model output
def test_malformed_output_is_repaired_once(container, llm, seeded):
    llm.push("plan_step", "this is not json at all",
             plan(Q_OPEN, "search", {"query": "warranty period"}),
             plan([{"id": "q1", "text": "warranty period", "status": "INSUFFICIENT", "evidence_ids": []}], "finalize"))
    state = _run(container)
    assert llm.count("plan_step") == 3                       # bad, repaired, then finalize
    assert "repaired" in " ".join(state.steps[0].notes)
    assert "INVALID" in llm.calls[1][1]                      # repair prompt carries the error
    assert state.status == InvStatus.COMPLETED


def test_malformed_output_twice_falls_back_safely(container, llm, seeded):
    llm.push("plan_step", {"sub_questions": "nope"}, {"action": {"type": "dance"}})
    state = _run(container)
    assert state.status == InvStatus.COMPLETED
    # Nothing was decomposed, so the model (not the documents) is why there is no answer - reported as such.
    assert state.final.outcome.value == "TOOL_FAILURE"
    assert "could not be used" in state.final.answer and "not a lack of evidence" in state.final.answer
    assert state.steps[-1].planner_ok is False
    assert llm.count("compose_answer") == 0


def test_invalid_tool_arguments_are_not_retried(container, llm, seeded):
    llm.push("plan_step", plan(Q_OPEN, "search", {"query": ""}),
             plan([{"id": "q1", "text": "warranty period", "status": "INSUFFICIENT", "evidence_ids": []}], "finalize"))
    state = _run(container)
    obs = state.steps[0].observation
    assert obs.status.value == "ERROR" and obs.error.startswith("invalid arguments") and obs.attempts == 1
    assert state.final.outcome.value == "INSUFFICIENT_EVIDENCE"   # bad args is not a backend failure


# ------------------------------------------------------------------ tool failures
def test_transient_search_failure_is_retried_and_recovers(container, llm, index, seeded):
    index.fail_next = 1
    llm.push("plan_step", plan(Q_OPEN, "search", {"query": "warranty period"}),
             plan([{"id": "q1", "text": "warranty period", "status": "INSUFFICIENT", "evidence_ids": []}], "finalize"))
    state = _run(container)
    obs = state.steps[0].observation
    assert obs.status.value == "OK" and obs.attempts == 2
    assert not state.failed_tools


def test_persistent_tool_failure_is_reported_as_tool_failure(container, llm, index, seeded):
    index.fail_next = 100
    llm.push("plan_step", plan(Q_OPEN, "search", {"query": "warranty period"}),
             plan(Q_OPEN, "search", {"query": "warranty period"}),       # one planner retry is allowed
             plan(Q_OPEN, "finalize"))
    state = _run(container)
    assert [s.observation.status.value for s in state.steps[:2]] == ["ERROR", "ERROR"]
    assert state.steps[0].observation.attempts == 3                     # 1 + 2 retries inside the registry
    assert state.final.outcome.value == "TOOL_FAILURE"                   # NOT "insufficient evidence"
    assert "not evidence that the answer is unavailable" in state.final.answer


def test_llm_outage_is_a_tool_failure(container, llm, seeded):
    llm.push("plan_step", llm_down())
    state = _run(container)
    assert state.final.outcome.value == "TOOL_FAILURE"


def test_consecutive_failure_budget_forces_finalize(container, llm, index, settings, seeded):
    index.fail_next = 1000
    llm.push("plan_step", *[plan(Q_OPEN, "search", {"query": f"warranty term {i}"}) for i in range(5)])
    state = _run(container)
    assert len(state.steps) == settings.max_consecutive_failures
    assert any("consecutive tool failures" in n for n in state.notes)
    assert state.final.outcome.value == "TOOL_FAILURE"


# ------------------------------------------------------------------ repeats and budgets
def test_repeated_successful_call_is_blocked(container, llm, index, seeded):
    same = plan(Q_OPEN, "search", {"query": "Warranty  PERIOD"})
    llm.push("plan_step", plan(Q_OPEN, "search", {"query": "warranty period"}), same, same, same)
    state = _run(container)
    assert [s.observation.status.value for s in state.steps] == ["OK", "BLOCKED", "BLOCKED", "BLOCKED"]
    assert index.search_calls == 1                                       # blocked calls never hit the index
    assert any("repeating" in n for n in state.notes)
    assert state.status == InvStatus.COMPLETED


def test_step_budget_forces_finalize(container, llm, settings, seeded):
    settings.max_steps = 3
    container.investigator.s.max_steps = 3
    llm.push("plan_step", *[plan(Q_OPEN, "search", {"query": f"query number {i}"}) for i in range(5)])
    state = _run(container)
    assert len(state.steps) == 3
    assert any("budget" in n for n in state.notes)
    assert state.final.outcome.value == "INSUFFICIENT_EVIDENCE"


# ------------------------------------------------------------------ evidence guards
def test_hallucinated_evidence_ids_are_rejected(container, llm, seeded):
    llm.push("plan_step", plan([{"id": "q1", "text": "warranty period", "status": "SUPPORTED",
                                 "evidence_ids": ["e42"]}], "finalize"))
    state = _run(container)
    assert state.sub_questions[0].status.value == "INSUFFICIENT"
    assert state.final.outcome.value == "INSUFFICIENT_EVIDENCE"


def test_low_relevance_chunks_cannot_support(container, llm, seeded):
    def cite_weak(user):
        v = parse_view(user)
        weak = [e["id"] for e in v["evidence_ledger"] if e.get("rerank", 1) < 0.2 and e.get("rank", 1) > 3]
        assert weak, "expected at least one weak hit"
        return plan([{"id": "q1", "text": "X500 warranty", "status": "SUPPORTED", "evidence_ids": weak}], "finalize")

    llm.push("plan_step", plan(Q_OPEN, "search", {"query": "seawater corrosion lighthouse penguin volcano glacier warranty", "top_k": 10}),
             cite_weak)
    state = _run(container)
    assert state.sub_questions[0].status.value == "INSUFFICIENT"
    assert "low relevance" in state.sub_questions[0].note


def test_unresolved_conflict_outcome(container, llm, seeded):
    def conflicting(user):
        v = parse_view(user)
        ids = [e["id"] for e in v["evidence_ledger"]][:2]
        return plan([{"id": "q1", "text": "warranty period", "status": "CONFLICTING", "evidence_ids": ids,
                      "note": "24 vs 12 months"}], "finalize")

    llm.push("plan_step", plan(Q_OPEN, "search", {"query": "warranty period months"}), conflicting)
    state = _run(container)
    assert state.final.outcome.value == "CONFLICT"
    assert state.final.citations


def test_robust_unverified_gives_caveated_answer(container, llm, ocr, seeded):
    from tests.fakes import receipt_nonhinge

    state = container.investigator.create("When was the pump bought?")
    ocr.queue.append(receipt_nonhinge())
    info = container.investigator.attach_upload(state, "r.png", (SAMPLES / "receipt_nonhinge.png").read_bytes())
    llm.push("plan_step", plan(Q_OPEN, "process_document", {"doc_id": info["doc_id"], "fields": ["receipt_number"]}),
             lambda u: plan([{"id": "q1", "text": "receipt number", "status": "SUPPORTED",
                              "evidence_ids": [e["id"] for e in parse_view(u)["evidence_ledger"]]}],
                            "finalize", robust_to_unverified=True))
    llm.push("extract_fields", {"fields": [{"name": "receipt_number", "found": True, "value": "R-2025-03892",
                                            "page": 1, "source_text": "Receipt No: R-2025-03892"}]})
    llm.push("compose_answer", {"answer": "Receipt R-2025-03892 [e1] (unverified).", "citations": ["e1"]})
    state = container.investigator.run(state.id)
    assert state.sub_questions[0].status.value == "UNVERIFIED"
    assert state.final.outcome.value == "ANSWERED_WITH_CAVEATS"
    assert any("UNVERIFIED" in c for c in state.final.caveats)


def test_composer_failure_falls_back_to_sourced_answer(container, llm, seeded):
    def supported(user):
        v = parse_view(user)
        ids = [e["id"] for e in v["evidence_ledger"] if "twelve (12) months" in e["text"]]
        return plan([{"id": "q1", "text": "commercial warranty period", "status": "SUPPORTED", "evidence_ids": ids,
                      "note": "12 months"}], "finalize")

    llm.push("plan_step", plan(Q_OPEN, "search", {"query": "commercial warranty period months"}), supported)
    llm.push("compose_answer", "garbage", "still garbage")
    state = _run(container)
    assert state.final.outcome.value == "ANSWERED"
    assert state.final.citations
    assert any("fallback" in r for r in state.final.reasons)


# ------------------------------------------------------------------ interruption / concurrency
def test_interrupted_investigation_resumes(container, llm, seeded):
    state = container.investigator.create("What is the warranty period?")
    llm.push("plan_step", plan(Q_OPEN, "search", {"query": "warranty period"}))
    llm.push("plan_step", RuntimeError("process killed"))         # crash mid-run (not an LLMError)
    crashed = container.investigator.run(state.id)
    assert crashed.status == InvStatus.FAILED and len(crashed.steps) == 1   # step 1 was persisted

    # simulate a hard kill instead: the DB still says RUNNING when the server restarts
    crashed.status = InvStatus.RUNNING
    container.repo.save_investigation(crashed)
    report = container.startup()
    assert state.id in report["interrupted"]
    assert container.investigator.get(state.id).status == InvStatus.INTERRUPTED

    llm.push("plan_step", plan([{"id": "q1", "text": "warranty period", "status": "INSUFFICIENT",
                                 "evidence_ids": []}], "finalize"))
    done = container.investigator.run(state.id)
    assert done.status == InvStatus.COMPLETED and len(done.steps) == 2      # nothing redone


def test_running_investigation_cannot_be_started_twice(client, container):
    state = container.investigator.create("What is the warranty period?")
    assert container.repo.claim_investigation(state.id)                     # someone else holds it
    r = client.post(f"/investigations/{state.id}/continue", data={"message": "hello"})
    assert r.status_code == 409


def test_needs_input_requires_input(client, container):
    state = container.investigator.create("What is the warranty period?")
    state.status = InvStatus.NEEDS_INPUT
    state.pending_question = "Upload the receipt"
    container.repo.save_investigation(state)
    r = client.post(f"/investigations/{state.id}/continue")
    assert r.status_code == 400 and "Upload the receipt" in r.text


def test_state_round_trips_through_sqlite(container):
    s = container.investigator.create("round trip question")
    loaded = container.repo.get_investigation(s.id)
    assert isinstance(loaded, InvestigationState) and loaded.question == "round trip question"


# ------------------------------------------------------------------ ask_user discipline
def test_ask_user_blocked_until_resources_investigated(container, llm, seeded):
    ask = plan([{"id": "q1", "text": "purchase date", "status": "MISSING", "evidence_ids": []}], "ask_user",
               {"question": "When did you buy it?"})
    llm.push("plan_step", ask,
             plan(Q_OPEN, "search", {"query": "purchase date receipt"}),
             plan(Q_OPEN, "get_metadata", {"doc_type": "contract"}),
             plan(Q_OPEN, "search", {"query": "lease delivery date"}),
             ask)
    state = _run(container)
    first = state.steps[0].observation
    assert first.status.value == "BLOCKED" and "investigate" in first.error.lower()
    assert state.status == InvStatus.NEEDS_INPUT and state.steps[-1].observation.status.value == "PAUSED"


def test_second_question_finalizes_with_explanation(container, llm, seeded):
    state = container.investigator.create("When was the pump bought?")
    state.steps.append(StepRecord(idx=1, planner_ok=True,
                                  action=Action(type=ActionType.ASK_USER, args={"question": "receipt?"}),
                                  observation=Observation(tool="ask_user", status=ObsStatus.PAUSED)))
    container.repo.save_investigation(state)
    llm.push("plan_step", plan([{"id": "q1", "text": "purchase date", "status": "UNVERIFIED", "evidence_ids": []}],
                               "ask_user", {"question": "Please upload a clearer copy of the receipt."}))
    done = container.investigator.run(state.id)
    assert done.status == InvStatus.COMPLETED
    assert done.final.outcome.value == "INSUFFICIENT_EVIDENCE"
    assert "clearer copy" in done.final.answer                      # tells the user what is needed
    assert any("user-question limit" in n for n in done.notes)


def test_planner_sees_corpus_catalog(container, llm, seeded):
    llm.push("plan_step", plan(Q_OPEN, "finalize"))
    _run(container)
    view = parse_view(llm.calls[0][1])
    titles = [d["title"] for d in view["corpus"]]
    assert len(titles) == 5 and any("Amendment" in t for t in titles)
    contract = next(d for d in view["corpus"] if d["type"] == "contract")
    assert contract["ocr_pages"] == [3]

"""Pre-finalize answer review (dates grounded in evidence, amendments consulted), version-aware first
retrieval, attachments read before planning, and failing fast when search or the LLM quota is down.
Each test is phrased as a general rule; the observed incident is only the example."""
from __future__ import annotations

import json

import httpx
import pytest

from app.agent.guards import find_dates, ungrounded_dates
from app.agent.models import EvidenceKind, InvestigationState, LedgerEntry, Verification
from app.llm.base import LLMError
from app.llm.openai_compat import OpenAICompatClient
from app.retrieval.base import SearchUnavailable
from tests.conftest import SAMPLES
from tests.fakes import ocr_page, parse_view
from tests.scenario import _step, sq

MAIN = ("Is the motor seizure of Acme Facilities' Aquaflow X200 pump (serial AQ-X200-55871) on 14 March 2026 "
        "covered under warranty?")


def _state(texts: list[str], question: str = MAIN) -> InvestigationState:
    s = InvestigationState(id="inv_t", question=question)
    s.ledger = [LedgerEntry(id=f"e{i + 1}", kind=EvidenceKind.CHUNK, doc_id="d", doc_name="doc", page=1, text=t,
                            verification=Verification.VERIFIED) for i, t in enumerate(texts)]
    return s


# ------------------------------------------------------------------ date grounding (pure)
def test_dates_are_found_in_common_formats():
    found = [v for _, v in find_dates("bought 20-03-2025, effective 2025-01-01, on 14 March 2026, "
                                      "or March 5, 2025")]
    assert found == [(2025, 1, 1), (2025, 3, 20), (2026, 3, 14), (2025, 3, 5)]


def test_date_from_outside_the_evidence_is_flagged():
    # Live incident: the answer said the lease began 1 March 2025 (a catalog field, not evidence).
    s = _state(["Warranty claims are handled under the Aquaflow Limited Warranty Policy.",
                "twenty-four (24) months from the date of purchase; e.g. bought 10 May 2023, covered until 10 May 2025"])
    assert ungrounded_dates(s, "The lease began 1 March 2025, so 14 March 2026 is within warranty.") == ["1 March 2025"]


def test_dates_derived_by_a_period_in_the_sources_are_accepted():
    s = _state(["Date: 20-03-2025", "the warranty period is reduced to twelve (12) months from the date of purchase"])
    assert ungrounded_dates(s, "Bought 20 March 2025, so the warranty ran until 20 March 2026 [e1].") == []
    # ... but not by a period no source mentions
    assert ungrounded_dates(s, "Covered until 20 September 2027.") == ["20 September 2027"]


# ------------------------------------------------------------------ review loop
def test_answer_with_ungrounded_date_is_sent_back_then_accepted(container, llm, settings, seeded):
    settings.initial_search = container.investigator.s.initial_search = True
    seen = []

    def chunk_ids(v):
        return [e["id"] for e in v["evidence_ledger"] if e["kind"] == "chunk"][:2]

    def answer(text):
        def plan(user):
            v = parse_view(user)
            seen.append(v["system_notes"])
            ids = chunk_ids(v)
            return {**_step([sq("q1", "covered?", "SUPPORTED", ids)], "finalize"),
                    "answer": {"text": text.format(ids[0]), "citations": ids}}
        return plan

    llm.push("plan_step", answer("Covered: the lease began 1 March 2025 [{}]."),
             answer("Coverage depends on the purchase date on the receipt [{}]."))
    state = container.investigator.run(container.investigator.create("What is covered by the warranty?").id)
    sent_back = [s for s in state.steps if s.label == "Answer sent back for a fix"]
    assert len(sent_back) == 1 and "1 March 2025" in sent_back[0].observation.error
    assert any("1 March 2025" in n for n in seen[1])           # the planner was told what was wrong
    assert "1 March 2025" not in state.final.answer


def test_problem_that_survives_revision_downgrades_the_answer(container, llm, settings, seeded):
    settings.initial_search = container.investigator.s.initial_search = True
    def plan(user):
        ids = [e["id"] for e in parse_view(user)["evidence_ledger"] if e["kind"] == "chunk"][:1]
        return {**_step([sq("q1", "covered?", "SUPPORTED", ids)], "finalize"),
                "answer": {"text": f"Covered since 1 March 2025 [{ids[0]}].", "citations": ids}}

    llm.push("plan_step", plan, plan, plan)
    state = container.investigator.run(container.investigator.create("What is covered by the warranty?").id)
    # a verdict resting on a date no source contains is withheld, not shown with a caveat
    assert state.final.outcome.value == "INSUFFICIENT_EVIDENCE"
    assert any("1 March 2025" in r and "withheld" in r for r in state.final.reasons)


def test_run_ending_early_after_a_rejected_answer_does_not_return_it(container, llm, settings, seeded):
    """Live test 14: the answer was sent back twice for an invented lease date, then the planner call failed
    (rate limit) and the run finalized from the sub-questions, still saying 'not covered'. The final answer is
    now checked whoever wrote it."""
    settings.initial_search = container.investigator.s.initial_search = True

    def plan(user):
        ids = [e["id"] for e in parse_view(user)["evidence_ledger"] if e["kind"] == "chunk"][:2]
        return {**_step([sq("q1", "lease start date (purchase date)", "SUPPORTED", ids[:1], "lease began 1 Mar 2025"),
                         sq("q2", "in warranty on 14 March 2026?", "SUPPORTED", ids)], "finalize"),
                "answer": {"text": f"The lease began on 1 Mar 2025 [{ids[0]}], so the warranty expired on 28 Feb "
                                   f"2026; not covered.", "citations": ids}}

    llm.push("plan_step", plan, plan, LLMError("rate limit reached on every configured model"))
    llm.push("compose_answer", {"answer": "The lease began on 1 Mar 2025 [e1]; expired 28 Feb 2026, not covered.",
                                "citations": ["e1"], "caveats": []})
    state = container.investigator.run(container.investigator.create(MAIN).id)
    assert len([s for s in state.steps if s.label == "Answer sent back for a fix"]) == 2
    assert state.final.outcome.value == "INSUFFICIENT_EVIDENCE"
    assert "1 Mar 2025" not in state.final.answer.split("What was established")[0]
    q1 = next(q for q in state.sub_questions if q.id == "q1")
    assert q1.status.value == "INSUFFICIENT" and "no source contains" in q1.note


# ------------------------------------------------------------------ version-aware first retrieval
def test_amendment_of_a_retrieved_document_is_fetched_before_planning(container, llm, settings, seeded):
    settings.initial_search = container.investigator.s.initial_search = True
    first = {}

    def plan(user):
        v = parse_view(user)
        first["docs"] = {e["doc"] for e in v["evidence_ledger"]}
        ids = [e["id"] for e in v["evidence_ledger"] if "3.1 Covered components" in e["text"]][:1]
        return {**_step([sq("q1", "what is covered", "SUPPORTED", ids)], "finalize"),
                "answer": {"text": f"Motor, housing, impeller ... [{ids[0]}]", "citations": ids}}

    llm.push("plan_step", plan)
    state = container.investigator.run(container.investigator.create("What all is covered under the warranty?").id)
    assert any("Amendment" in d for d in first["docs"])            # the planner saw the amendment at once
    assert llm.count("plan_step") == 1                              # ... so no answer had to be sent back
    assert state.final.outcome.value == "ANSWERED"


# ------------------------------------------------------------------ attachments are read before planning
def test_attachment_is_read_before_the_first_planner_call(container, llm, ocr, settings, seeded):
    settings.auto_read_attachments = container.investigator.s.auto_read_attachments = True
    ocr.queue.append(ocr_page("INSTALLATION CERTIFICATE\nSerial: AQ-X200-55871\nInstaller: D. Okafor\n"
                              "Aquaflow Certified Installer no. ACI-2291\nInstallation date: 22-03-2025"))
    inv = container.investigator
    state = inv.create("Was Acme's X200 installed by an Aquaflow Certified Installer?")
    up = inv.attach_upload(state, "cert.png", (SAMPLES / "receipt_nonhinge.png").read_bytes() + b"cert")
    seen = {}

    def plan(user):
        v = parse_view(user)
        seen["ledger"] = [e for e in v["evidence_ledger"] if "ACI-2291" in e["text"]]
        ids = [e["id"] for e in seen["ledger"]]
        return {**_step([sq("q1", "certified installer?", "SUPPORTED", ids)], "finalize"),
                "answer": {"text": f"Yes, D. Okafor, ACI-2291 [{ids[0]}]", "citations": ids}}

    llm.push("plan_step", plan)
    state = inv.run(state.id)
    assert seen["ledger"], "the attachment's text must be in the ledger before the planner's first call"
    assert [s.action.type.value for s in state.steps[:2]] == ["process_document", "search"]
    assert all(s.auto for s in state.steps[:2]) and state.auto_read == [up["doc_id"]]
    assert state.final.outcome.value == "ANSWERED"


def test_indexing_never_narrows_a_global_document(container, seeded):
    from app.agent.models import Action, ActionType
    from app.tools.base import ToolContext

    doc_id = seeded["warranty_policy_v1.pdf"]
    obs = container.investigator.registry.run(
        Action(type=ActionType.PROCESS_DOCUMENT, reason="r", args={"doc_id": doc_id, "mode": "index"}),
        ToolContext("inv_x", []))
    assert obs.status.value == "OK"
    assert container.repo.get_document(doc_id)["scope"] == "global"


# ------------------------------------------------------------------ fail fast
def test_search_outage_stops_early_with_a_clear_reason(container, llm, index, settings, seeded):
    settings.initial_search = container.investigator.s.initial_search = True
    index.fail_next = 99
    state = container.investigator.run(container.investigator.create("How long does the Acme lease run?").id)
    assert state.final.outcome.value == "TOOL_FAILURE"
    assert llm.count("plan_step") == 0                         # no LLM budget spent on a run that cannot work
    assert "Elasticsearch could not be reached" in state.final.answer or "Search:" in state.final.answer


def _groq_429(minutes_left: str, kind: str = "tokens per day (TPD)"):
    msg = (f"Rate limit reached for model `m` on {kind}: Limit 200000, Used 198402. "
           f"Please try again in {minutes_left}. Need more tokens?")
    return httpx.Response(429, json={"error": {"message": msg, "code": "rate_limit_exceeded"}})


def test_long_quota_wait_fails_fast_instead_of_waiting():
    waits, calls = [], []

    def handler(request):
        calls.append(json.loads(request.content)["model"])
        return _groq_429("4m7.536s")

    c = OpenAICompatClient("https://api.groq.com/openai/v1", "k", "openai/gpt-oss-120b",
                           fallback_models=["openai/gpt-oss-20b"], transport=httpx.MockTransport(handler),
                           sleep=waits.append)
    with pytest.raises(LLMError) as err:
        c.complete_json(purpose="p", system="s", user="u", schema={"type": "object", "properties": {}})
    assert sum(waits) == 0                                   # previously: 3 x 60 s of waiting
    assert calls == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    assert "daily quota" in str(err.value) and "about 4 min" in str(err.value)


def test_short_per_minute_wait_is_still_waited_out():
    waits, n = [], []

    def handler(request):
        n.append(1)
        return _groq_429("7.5s", "tokens per minute (TPM)") if len(n) == 1 else \
            httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    c = OpenAICompatClient("https://llm.test/v1", "k", "m", fallback_models=[], transport=httpx.MockTransport(handler),
                           sleep=waits.append, mode="json")
    c.complete_json(purpose="p", system="s", user="u", schema={"type": "object", "properties": {}})
    assert waits and 7 <= waits[0] <= 8


def test_llm_quota_failure_is_explained_in_the_answer(container, llm, seeded):
    llm.push("plan_step", LLMError("rate limit reached on every configured model (a, b; limit: daily quota). "
                                   "The next model is available again in about 4 min; not waiting that long."))
    state = container.investigator.run(container.investigator.create("How long does the Acme lease run?").id)
    assert state.final.outcome.value == "TOOL_FAILURE"
    assert "daily quota" in state.final.answer and "about 4 min" in state.final.answer


def test_elasticsearch_circuit_breaker_fails_fast():
    from elastic_transport import ConnectionError as ESConnectionError

    from app.retrieval.es_client import ElasticIndex

    idx = ElasticIndex("http://127.0.0.1:1", "chunks", 4)
    calls = []

    def down(*a, **k):
        calls.append(1)
        raise ESConnectionError("refused")

    with pytest.raises(SearchUnavailable):
        idx._call(down)
    with pytest.raises(SearchUnavailable, match="not retried"):
        idx._call(down)
    assert len(calls) == 1                                   # second call did not touch the network


# ------------------------------------------------------------------ off-topic questions
def test_off_topic_question_is_not_asked_about_and_is_insufficient(container, llm, settings, seeded):
    """Live inv_c7551529b58a: 'What will the weather be in Leeds tomorrow?' - the planner asked the user to
    confirm the question and the run ended MISSING_INFO. Nothing retrieved relates to it, so asking is blocked
    and the outcome is INSUFFICIENT_EVIDENCE."""
    from app.agent.guards import OFF_TOPIC_SCORE

    settings.initial_search = container.investigator.s.initial_search = True
    container.reranker.score = lambda q, texts: [0.0001] * len(texts)       # nothing relates to the question
    ask = {**_step([sq("q1", "weather in Leeds?", "MISSING", [])], "ask_user",
                   {"question": "Are you looking for the weather forecast for Leeds tomorrow?"}),
           "action": {"type": "ask_user", "reason": "clarify",
                      "args": {"question": "Are you looking for the weather forecast for Leeds tomorrow?"}}}
    llm.push("plan_step", ask, ask, ask, {**_step([sq("q1", "weather in Leeds?", "MISSING", [])], "finalize")})
    state = container.investigator.run(container.investigator.create("What will the weather be in Leeds tomorrow?").id)
    assert state.status.value == "COMPLETED"                                 # never paused for input
    assert not any(s.observation and s.observation.status.value == "PAUSED" for s in state.steps)
    assert state.final.outcome.value == "INSUFFICIENT_EVIDENCE"
    assert any("nothing in the documents relates" in r for r in state.final.reasons)
    assert OFF_TOPIC_SCORE < 0.011                                           # below every in-scope question


def test_in_scope_missing_fact_still_asks(container, llm, settings, seeded):
    settings.initial_search = container.investigator.s.initial_search = True
    container.reranker.score = lambda q, texts: [0.4] * len(texts)          # the documents relate
    container.investigator.s.min_actions_before_ask = 1
    llm.push("plan_step", _step([sq("q1", "purchase date?", "MISSING", [])], "ask_user",
                                {"question": "Please upload the sales receipt.", "request_upload": True}))
    state = container.investigator.run(container.investigator.create(MAIN).id)
    assert state.status.value == "NEEDS_INPUT"


def test_deadline_computed_from_a_documents_effective_date_is_sent_back(container, llm, settings, seeded):
    """Live inv_1f41511dc051 (test 10a, no receipt): the model read the contract's catalog effective date
    (2025-03-01) as the purchase date and answered 'expired 1 March 2026 - not covered'. A document's effective
    date may be quoted but is not a basis for warranty deadlines; the answer goes back and the agent asks."""
    settings.initial_search = container.investigator.s.initial_search = True
    contract = seeded["customer_contract_acme.pdf"]

    def lookup(user):
        return _step([sq("q1", "lease effective date", "OPEN", [])], "get_metadata", {"doc_ids": [contract]})

    def wrong_answer(user):
        v = parse_view(user)
        meta = [e["id"] for e in v["evidence_ledger"] if e["kind"] == "metadata" and "2025-03-01" in e["text"]]
        return {**_step([sq("q1", "lease effective date", "SUPPORTED", meta, "2025-03-01")], "finalize"),
                "answer": {"text": f"The lease is effective 1 March 2025 [{meta[0]}], so the warranty expired on "
                                   f"1 March 2026: not covered.", "citations": meta}}

    def ask(user):
        assert any("effective date" in n and "ask the user" in n for n in parse_view(user)["system_notes"])
        return _step([sq("q1", "purchase date", "MISSING", [])], "ask_user",
                     {"question": "Please upload the original sales receipt showing the purchase date.",
                      "request_upload": True})

    llm.push("plan_step", lookup, wrong_answer, ask)
    state = container.investigator.run(container.investigator.create(MAIN).id)
    assert any(s.label == "Answer sent back for a fix" for s in state.steps)
    assert state.status.value == "NEEDS_INPUT" and "receipt" in state.pending_question


def test_metadata_effective_date_is_labelled_as_a_document_date(container, seeded):
    from app.agent.models import Action, ActionType
    from app.tools.base import ToolContext

    obs = container.investigator.registry.run(
        Action(type=ActionType.GET_METADATA, reason="r", args={"doc_ids": [seeded["customer_contract_acme.pdf"]]}),
        ToolContext("inv_x", []))
    assert "not a purchase, delivery or installation date" in obs.evidence[0].text


def test_agent_cannot_publish_an_upload_to_other_investigations(container, seeded):
    from app.agent.models import Action, ActionType
    from app.tools.base import ToolContext

    inv = container.investigator
    a = inv.create("Is this receipt for Acme's X200?")
    up = inv.attach_upload(a, "receipt.png", (SAMPLES / "receipt_nonhinge.png").read_bytes())
    obs = inv.registry.run(Action(type=ActionType.PROCESS_DOCUMENT, reason="r",
                                  args={"doc_id": up["doc_id"], "mode": "index", "scope": "global"}),
                           ToolContext(a.id, [up["doc_id"]]))
    assert obs.status.value == "OK"
    assert container.repo.get_document(up["doc_id"])["scope"] == f"session:{a.id}"     # not global
    from app.retrieval.base import SearchFilters, Visibility

    other = container.retriever.search("AQUAFLOW SALES RECEIPT Serial", SearchFilters(),
                                       Visibility.for_investigation("inv_other", []))
    assert all(h.chunk["doc_id"] != up["doc_id"] for h in other.hits)               # invisible elsewhere

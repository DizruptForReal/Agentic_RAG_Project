"""Fast path (initial search + single-call answer), strict structured output, live activity, timings,
and broad questions over the richer corpus."""
from __future__ import annotations

import json

import httpx

from app.agent import progress
from app.agent.planner import PLAN_SCHEMA
from app.llm.openai_compat import OpenAICompatClient, restore_from_strict, to_strict_schema
from app.retrieval.base import SearchFilters, Visibility
from tests.conftest import SAMPLES as SAMPLES_DIR
from tests.fakes import parse_view

GENERIC_Q = "What all is covered under the warranty?"


def _covered_ids(view):
    return [e["id"] for e in view["evidence_ledger"] if "3.1 Covered components" in e["text"]]


def test_broad_question_finds_the_coverage_section(container, seeded):
    res = container.retriever.search(GENERIC_Q, SearchFilters(), Visibility())
    sections = [h.chunk["section"] for h in res.hits[:3]]
    assert "3. What is covered" in sections


def test_fast_path_answers_in_one_planner_call(container, llm, settings, seeded):
    settings.initial_search = True
    container.investigator.s.initial_search = True
    seen = {}

    def plan(user):
        v = parse_view(user)
        seen["ledger"] = len(v["evidence_ledger"])
        ids = _covered_ids(v)
        assert ids, "initial search should already have found section 3"
        return {"sub_questions": [{"id": "q1", "text": "What does the warranty cover?", "status": "SUPPORTED",
                                   "evidence_ids": ids}],
                "action": {"type": "finalize", "args": {}, "reason": "ledger answers it"},
                "answer": {"text": f"The warranty covers defects in the motor, housing, impeller ... [{ids[0]}]",
                           "citations": ids}}

    llm.push("plan_step", plan)
    state = container.investigator.create(GENERIC_Q)
    state = container.investigator.run(state.id)
    assert state.steps[0].auto and state.steps[0].action.type.value == "search"
    assert state.steps[0].label.startswith("Searching the knowledge base")
    assert seen["ledger"] > 0
    assert llm.count("plan_step") == 1 and llm.count("compose_answer") == 0      # one LLM call in total
    assert state.final.outcome.value == "ANSWERED"
    assert any("same call" in r for r in state.final.reasons)
    assert state.final.citations[0].doc_name.startswith("Aquaflow Limited Warranty Policy")


def test_steps_record_timings_and_labels(container, llm, seeded):
    llm.push("plan_step",
             lambda u: {"sub_questions": [{"id": "q1", "text": "t", "status": "OPEN", "evidence_ids": []}],
                        "action": {"type": "get_metadata", "args": {"doc_type": "amendment"}, "reason": "r"}},
             lambda u: {"sub_questions": [{"id": "q1", "text": "t", "status": "INSUFFICIENT", "evidence_ids": []}],
                        "action": {"type": "finalize", "args": {}, "reason": "r"}})
    state = container.investigator.run(container.investigator.create("Which amendment applies?").id)
    step = state.steps[0]
    assert step.label == "Checking document versions and metadata (amendment)"
    assert step.llm_ms is not None and step.tool_ms is not None and step.wait_ms == 0
    assert "total_ms" in state.timings and "compose_ms" in state.timings


def test_live_activity_is_reported_while_running(container, llm, seeded):
    state = container.investigator.create("Which amendment applies?")
    observed = []

    def plan(user):
        observed.append(progress.BOARD.get(state.id))
        return {"sub_questions": [{"id": "q1", "text": "t", "status": "INSUFFICIENT", "evidence_ids": []}],
                "action": {"type": "finalize", "args": {}, "reason": "r"}}

    llm.push("plan_step", plan)
    container.investigator.run(state.id)
    assert observed[0]["text"] == "Understanding the question and planning" and observed[0]["kind"] == "llm"
    assert progress.BOARD.get(state.id) is None                      # cleared when the run ends


def test_rate_limit_wait_is_reported_and_timed():
    seen = []

    class Board(progress.ActivityBoard):
        def set(self, inv_id, text, kind):
            seen.append((text, kind))
            super().set(inv_id, text, kind)

    def handler(request):
        if not seen or seen[-1][1] != "wait":
            return httpx.Response(429, headers={"retry-after": "7"}, json={"error": {"message": "slow"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    c = OpenAICompatClient("https://llm.test/v1", "k", "m", mode="json", transport=httpx.MockTransport(handler),
                           sleep=lambda s: None)
    with progress.reporting("inv_x", Board()):
        c.complete_json(purpose="p", system="s", user="u", schema={})
        waited = progress.take_wait()
    assert ("Waiting for the model's rate limit (7s)", "wait") in seen and waited == 7


def test_strict_mode_for_groq_gpt_oss():
    sent = []
    plan = {"sub_questions": [{"id": "q1", "text": "t", "status": "OPEN", "evidence_ids": [], "note": None}],
            "action": {"type": "search", "args": {"query": "warranty", "doc_type": None, "product": None,
                                                  "doc_ids": None, "top_k": None, "doc_id": None, "mode": None,
                                                  "fields": None, "scope": None, "question": None,
                                                  "request_upload": None}, "reason": "r"},
            "robust_to_unverified": None, "conflict_resolution": None, "answer": None}

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(plan)}}]})

    c = OpenAICompatClient("https://api.groq.com/openai/v1", "k", "openai/gpt-oss-120b",
                           transport=httpx.MockTransport(handler), sleep=lambda s: None)
    assert c.mode == "strict"
    out = c.complete_json(purpose="plan_step", system="s", user="u", schema=PLAN_SCHEMA)
    rf = sent[0]["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    assert "tools" not in sent[0] and sent[0]["reasoning_effort"] == "low"
    assert out["action"]["args"] == {"query": "warranty"}                  # unused (null) arguments dropped
    assert "note" not in out["sub_questions"][0] and "answer" not in out    # nulls of optional fields dropped


def test_strict_schema_rules():
    s = to_strict_schema(PLAN_SCHEMA)

    def check(node):
        if isinstance(node, dict):
            if node.get("type") in ("object", ["object", "null"]) and "properties" in node:
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for v in node.values():
                check(v)
        elif isinstance(node, list):
            for v in node:
                check(v)

    check(s)
    args = s["properties"]["action"]["properties"]["args"]
    assert args["type"] == "object" and "question" in args["properties"]        # typed, not a JSON string
    restored = restore_from_strict({"action": {"type": "ask_user", "reason": "r", "args": {
        "question": "When was it bought?", "query": None, "doc_id": None}}}, PLAN_SCHEMA)
    assert restored["action"]["args"] == {"question": "When was it bought?"}


def test_strict_rejected_falls_back_to_tools():
    def handler(request):
        body = json.loads(request.content)
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "json_schema not supported for this model"}})
        return httpx.Response(200, json={"choices": [{"message": {"tool_calls": [{"function": {
            "arguments": json.dumps({"ok": True})}}]}}]})

    c = OpenAICompatClient("https://api.groq.com/openai/v1", "k", "openai/gpt-oss-120b",
                           transport=httpx.MockTransport(handler), sleep=lambda s: None)
    assert c.complete_json(purpose="p", system="s", user="u", schema={"type": "object"}) == {"ok": True}
    assert c.mode == "tools"


# ------------------------------------------------------------------ accuracy fixes from the live test run
def _finalize_supported(ids_fn, answer=True, extra_sq=None):
    def plan(user):
        v = parse_view(user)
        ids = ids_fn(v)
        sqs = [{"id": "q1", "text": "main question", "status": "SUPPORTED", "evidence_ids": ids}]
        if extra_sq:
            sqs.append(extra_sq)
        out = {"sub_questions": sqs, "action": {"type": "finalize", "args": {}, "reason": "r"}}
        if answer:
            out["answer"] = {"text": f"Answer [{ids[0]}]", "citations": ids}
        return out
    return plan


def _run_fast(container, llm, settings, question, plan):
    settings.initial_search = True
    container.investigator.s.initial_search = True
    llm.push("plan_step", plan)
    return container.investigator.run(container.investigator.create(question).id)


def test_unknown_model_is_refused_even_if_the_planner_answers(container, llm, settings, seeded):
    """Live run: the planner applied the X-series policy to an 'X500'. The entity check now blocks that."""
    seen = {}

    def ids(v):
        seen["notes"] = v["system_notes"]
        return [e["id"] for e in v["evidence_ledger"] if e["kind"] == "chunk"][:1]   # planner wrongly applies any policy text

    state = _run_fast(container, llm, settings, "I purchased my Aquaflow X500 on December 17, 2024, is it still "
                      "under warranty?", _finalize_supported(ids))
    assert any("X500" in n for n in seen["notes"])                       # the planner was told
    assert state.unknown_entities == ["X500"]
    assert state.final.outcome.value == "INSUFFICIENT_EVIDENCE"
    assert "do not cover X500" in state.final.answer


def test_known_model_is_not_blocked(container, llm, settings, seeded):
    def ids(v):
        return [e["id"] for e in v["evidence_ledger"] if "42 m3/h" in e["text"]][:1]

    state = _run_fast(container, llm, settings, "What is the max flow rating of the X200 industrial pump?",
                      _finalize_supported(ids))
    assert state.unknown_entities == [] and state.final.outcome.value == "ANSWERED"


def test_optional_gap_gives_answer_with_caveat(container, llm, settings, seeded):
    def ids(v):
        return [e["id"] for e in v["evidence_ledger"] if "Key time limits" in e["text"] or "thirty (30) days" in e["text"]][:2]

    extra = {"id": "q2", "text": "Are there regional differences?", "status": "INSUFFICIENT", "evidence_ids": [],
             "required": False}
    state = _run_fast(container, llm, settings, "Till when can I claim my warranty?", _finalize_supported(ids, extra_sq=extra))
    assert state.final.outcome.value == "ANSWERED_WITH_CAVEATS"
    assert any("regional differences" in c for c in state.final.caveats)


def test_out_of_scope_question_wording(container, llm, settings, seeded):
    plan = lambda u: {"sub_questions": [{"id": "q1", "text": "days in January", "status": "INSUFFICIENT",
                                         "evidence_ids": []}],
                      "action": {"type": "finalize", "args": {}, "reason": "unrelated"}}
    state = _run_fast(container, llm, settings, "How many days are in January?", plan)
    assert state.final.answer.startswith("The available documents do not contain information")


def test_failover_to_second_model_on_rate_limit():
    used = []

    def handler(request):
        body = json.loads(request.content)
        used.append(body["model"])
        if body["model"] == "openai/gpt-oss-120b":
            return httpx.Response(429, headers={"retry-after": "40"}, json={"error": {"message": "TPM"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    waits = []
    c = OpenAICompatClient("https://api.groq.com/openai/v1", "k", "openai/gpt-oss-120b",
                           fallback_models=["openai/gpt-oss-20b"], transport=httpx.MockTransport(handler),
                           sleep=waits.append)
    c.complete_json(purpose="p", system="s", user="u", schema={"type": "object", "properties": {}})
    assert used == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"] and waits == []     # switched, no waiting
    assert c.last_model == "openai/gpt-oss-20b"
    c.complete_json(purpose="p", system="s", user="u", schema={"type": "object", "properties": {}})
    assert used[-1] == "openai/gpt-oss-20b"                     # primary still blocked: goes straight to fallback


def test_all_models_exhausted_gives_clear_error():
    import pytest

    from app.llm.base import LLMError

    c = OpenAICompatClient("https://api.groq.com/openai/v1", "k", "openai/gpt-oss-120b", max_retries=2,
                           fallback_models=["openai/gpt-oss-20b"], sleep=lambda s: None,
                           transport=httpx.MockTransport(lambda r: httpx.Response(429, json={"error": {}})))
    with pytest.raises(LLMError, match="every configured model"):
        c.complete_json(purpose="p", system="s", user="u", schema={"type": "object", "properties": {}})


def test_default_fallback_for_groq(tmp_path):
    from app.config import Settings
    from app.container import fallback_models

    s = Settings(data_dir=tmp_path, llm_provider="openai_compat", llm_base_url="https://api.groq.com/openai/v1",
                 llm_model="openai/gpt-oss-120b", _env_file=None)
    assert fallback_models(s) == ["openai/gpt-oss-20b"]
    s.llm_fallback_models = "none"
    assert fallback_models(s) == []


# ------------------------------------------------------------------ evidence visibility and partial answers
def test_relevant_evidence_is_shown_in_full_even_when_old_and_uncited(container, llm, seeded):
    """Live bug: the turnaround times sit at the end of the claims section and were cut off at 400 chars."""
    from app.agent.ledger import add_evidence
    from app.agent.models import EvidenceItem, EvidenceKind
    from app.agent.planner import build_view

    state = container.investigator.create("How long does a warranty claim take?")
    res = container.retriever.search("claims procedure acknowledges assessment business days", SearchFilters(),
                                     Visibility())
    claims = next(h for h in res.hits if h.chunk["section"].startswith("7. Claims procedure"))
    add_evidence(state, [EvidenceItem(kind=EvidenceKind.CHUNK, doc_id=claims.chunk["doc_id"],
                                      doc_name=claims.chunk["doc_name"], page=claims.chunk["page"],
                                      chunk_id=claims.chunk["chunk_id"], text=claims.chunk["text"],
                                      rerank_score=0.53)])
    add_evidence(state, [EvidenceItem(kind=EvidenceKind.CHUNK, doc_id="d_x", doc_name="noise", page=1,
                                      chunk_id=f"noise:{i}", text="unrelated words " * 60, rerank_score=0.001)
                         for i in range(12)])
    view = parse_view(build_view(state, [], 12, []))
    shown = next(e for e in view["evidence_ledger"] if e["id"] == "e1")
    assert "fifteen (15)" in shown["text"] and not shown["text"].endswith("...")
    assert any(e["text"].endswith("...") for e in view["evidence_ledger"][1:])     # low-relevance abbreviated
    assert len(json.dumps(view)) < 16000


def test_cited_partial_answer_reports_the_gap_as_a_caveat(container, llm, settings, seeded):
    def plan(user):
        v = parse_view(user)
        ids = [e["id"] for e in v["evidence_ledger"] if "Exclusions" in e["text"]][:1]
        return {"sub_questions": [
                    {"id": "q1", "text": "What does the policy exclude?", "status": "SUPPORTED", "evidence_ids": ids},
                    {"id": "q2", "text": "Does the amendment add exclusions?", "status": "INSUFFICIENT",
                     "evidence_ids": []}],
                "action": {"type": "finalize", "args": {}, "reason": "r"},
                "answer": {"text": f"Not covered: normal wear, dry running ... [{ids[0]}]", "citations": ids}}

    state = _run_fast(container, llm, settings, "What does the warranty not cover?", plan)
    assert state.final.outcome.value == "ANSWERED_WITH_CAVEATS"
    assert any("Does the amendment add exclusions?" in c for c in state.final.caveats)


def test_no_partial_answer_without_cited_evidence(container, llm, settings, seeded):
    plan = lambda u: {"sub_questions": [
                          {"id": "q1", "text": "a", "status": "SUPPORTED", "evidence_ids": ["e1"]},
                          {"id": "q2", "text": "b", "status": "INSUFFICIENT", "evidence_ids": []}],
                      "action": {"type": "finalize", "args": {}, "reason": "r"},
                      "answer": {"text": "made up [e99]", "citations": ["e99"]}}
    state = _run_fast(container, llm, settings, "What does the warranty not cover?", plan)
    assert state.final.outcome.value == "INSUFFICIENT_EVIDENCE"


def test_stale_rerank_floor_setting_is_ignored(tmp_path, monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("RERANK_FLOOR", "0.2")
    assert Settings(data_dir=tmp_path, _env_file=None).relevance_floor == 0.03


# ------------------------------------------------------------------ third live round
def test_reranker_is_a_voter_not_the_judge():
    """Measured: bge-reranker gave the correct 'what is covered' section 0.014 although BM25 and kNN both ranked
    it first. The fused ranking keeps it on top, and rank-based relevance keeps it citable."""
    from app.retrieval.retriever import HybridRetriever
    from app.tools.search_kb import is_relevant

    class Index:
        def bm25(self, q, f, v, k):
            return [{"chunk_id": "good", "text": "a"}, {"chunk_id": "x", "text": "b"}, {"chunk_id": "y", "text": "c"}]

        def knn(self, vec, f, v, k):
            return [{"chunk_id": "good", "text": "a"}, {"chunk_id": "y", "text": "c"}, {"chunk_id": "x", "text": "b"}]

    class Reranker:
        def score(self, q, passages):
            return [{"a": 0.014, "b": 0.15, "c": 0.12}[p.strip()[-1]] for p in passages]

    class Emb:
        def embed_query(self, q):
            return [1.0]

    res = HybridRetriever(Index(), Emb(), Reranker()).search("q", SearchFilters(), Visibility())
    assert res.hits[0].chunk["chunk_id"] == "good" and res.hits[0].rank == 1
    assert is_relevant(res.hits[0].rerank_score, res.hits[0].rank, 0.03)
    assert not is_relevant(0.01, 7, 0.03) and is_relevant(0.05, 9, 0.03)


def test_typed_reply_becomes_citable_user_evidence(container, llm, seeded):
    """Live bug: the purchase date typed as a reply was never usable as evidence."""
    state = container.investigator.create("Is my X200 still under warranty on 14 March 2026?")
    state.pending_question = "What is the purchase date?"
    container.investigator.add_message(state, "14 March 2024")
    entry = state.ledger[-1]
    assert entry.kind.value == "user" and entry.verification.value == "USER_STATED"
    assert "14 March 2024" in entry.text and "purchase date" in entry.text

    def plan(user):
        v = parse_view(user)
        uid = next(e["id"] for e in v["evidence_ledger"] if e["kind"] == "user")
        pol = [e["id"] for e in v["evidence_ledger"] if "twenty-four (24) months" in e["text"]][:1] or \
              [e["id"] for e in v["evidence_ledger"] if e["kind"] == "chunk"][:1]
        return {"sub_questions": [
                    {"id": "q1", "text": "Purchase date", "status": "SUPPORTED", "evidence_ids": [uid]},
                    {"id": "q2", "text": "Warranty period", "status": "SUPPORTED", "evidence_ids": pol}],
                "action": {"type": "finalize", "args": {}, "reason": "r"},
                "answer": {"text": f"Bought 14 March 2024 [{uid}] + 24 months [{pol[0]}]: covered until 14 March 2026.",
                           "citations": [uid] + pol}}

    llm.push("plan_step",
             lambda u: {"sub_questions": [{"id": "q1", "text": "Purchase date", "status": "OPEN", "evidence_ids": []}],
                        "action": {"type": "search", "args": {"query": "warranty period months purchase"}, "reason": "r"}},
             plan)
    done = container.investigator.run(state.id)
    assert done.final.outcome.value == "ANSWERED_WITH_CAVEATS"
    assert any("relies on information you provided" in r for r in done.final.reasons)
    assert any("information you provided" in c for c in done.final.caveats)
    assert any(c.doc_name == "Your reply" for c in done.final.citations)


def test_partial_comparison_when_one_product_has_no_data(container, llm, settings, seeded):
    """Live run: X200 and X300 specs were found, X100 has no sheet; the whole comparison was refused."""
    def plan(user):
        v = parse_view(user)
        x200 = [e["id"] for e in v["evidence_ledger"] if "42 m3/h" in e["text"]][:1]
        x300 = [e["id"] for e in v["evidence_ledger"] if "3.5 m3/h" in e["text"]][:1]
        return {"sub_questions": [
                    {"id": "q1", "text": "X200 specifications", "status": "SUPPORTED", "evidence_ids": x200},
                    {"id": "q2", "text": "X300 specifications", "status": "SUPPORTED", "evidence_ids": x300},
                    {"id": "q3", "text": "X100 specifications", "status": "MISSING", "evidence_ids": [],
                     "note": "no X100 specification sheet in the documents"}],
                "action": {"type": "finalize", "args": {}, "reason": "answer what is available"},
                "answer": {"text": f"| Model | Max flow |\n|---|---|\n| X200 | 42 m3/h [{x200[0]}] |\n"
                                   f"| X300 | 3.5 m3/h [{x300[0]}] |\n\nNo X100 specifications are in the documents.",
                           "citations": x200 + x300}}

    state = _run_fast(container, llm, settings, "Compare the specifications of the X-series pumps (flow, pressure, "
                      "motor) in a table", plan)
    assert state.final.outcome.value == "ANSWERED_WITH_CAVEATS"
    assert any("X100 specifications" in c for c in state.final.caveats)
    assert "| X200 |" in state.final.answer


def test_fullwidth_citation_brackets_are_recognised(container, llm, settings, seeded):
    def plan(user):
        v = parse_view(user)
        ids = [e["id"] for e in v["evidence_ledger"] if e["kind"] == "chunk"][:1]
        return {"sub_questions": [{"id": "q1", "text": "t", "status": "SUPPORTED", "evidence_ids": ids}],
                "action": {"type": "finalize", "args": {}, "reason": "r"},
                "answer": {"text": f"Covered parts are listed【{ids[0]}】.", "citations": []}}

    state = _run_fast(container, llm, settings, "What all is covered under the warranty?", plan)
    assert state.final.outcome.value == "ANSWERED" and state.final.citations


def test_planner_answer_without_sub_questions_is_not_refused(container, llm, settings, seeded):
    """Held-out eval: the fallback model returned a cited answer but no sub-questions; it was refused."""
    def plan(user):
        v = parse_view(user)
        ids = [e["id"] for e in v["evidence_ledger"] if e["kind"] == "chunk"][:1]
        return {"sub_questions": [], "action": {"type": "finalize", "args": {}, "reason": "r"},
                "answer": {"text": f"The answer [{ids[0]}].", "citations": ids}}

    state = _run_fast(container, llm, settings, "What is the rated pressure of the X200?", plan)
    assert state.final.outcome.value == "ANSWERED"
    assert state.sub_questions[0].text == "What is the rated pressure of the X200?"


def test_no_sub_questions_and_no_answer_still_refuses(container, llm, settings, seeded):
    plan = lambda u: {"sub_questions": [], "action": {"type": "finalize", "args": {}, "reason": "r"}}
    state = _run_fast(container, llm, settings, "What is the rated pressure of the X200?", plan)
    assert state.final.outcome.value == "INSUFFICIENT_EVIDENCE"


def test_identifier_only_in_an_upload_stops_being_unknown(container, llm, ocr, seeded):
    """A serial that appears only on an uploaded card is 'unknown' until the card is read; after
    process_document it must no longer force a refusal (general rule, not tied to one document)."""
    from tests.fakes import ocr_page
    from tests.scenario import _step, sq

    ocr.queue.append(ocr_page("AQUAFLOW X300 EXTENDED CARE\nSerial: AQ-X300-10457\nPurchase date: 02-09-2024"))
    inv = container.investigator
    state = inv.create("Is X300 serial AQ-X300-10457 under warranty?")
    upload = inv.attach_upload(state, "card.png", (SAMPLES_DIR / "receipt_nonhinge.png").read_bytes() + b"card")
    llm.push("extract_fields", {"fields": [{"name": "purchase_date", "found": True, "value": "02-09-2024",
                                            "page": 1, "source_text": "Purchase date: 02-09-2024"}]})

    def read_card(user):
        return _step([sq("q1", "purchase date", "OPEN")], "process_document",
                     {"doc_id": upload["doc_id"], "mode": "extract_only", "fields": ["purchase_date"]})

    def finish(user):
        ids = [e["id"] for e in parse_view(user)["evidence_ledger"] if e["kind"] == "field"]
        return {**_step([sq("q1", "purchase date", "SUPPORTED", ids)], "finalize"),
                "answer": {"text": f"Bought 02-09-2024 [{ids[0]}]", "citations": ids}}

    llm.push("plan_step", read_card, finish)
    state = inv.run(state.id)
    assert state.unknown_entities == []
    assert any("now found in an attached document" in n for n in state.notes)
    assert state.final.outcome.value.startswith("ANSWERED")

"""Scripted planner for the warranty scenario. Each step is a function of the state view the real
planner would receive, so evidence ids are looked up the same way a model would read them."""
from __future__ import annotations

from typing import Any

from tests.fakes import parse_view

QUESTION = ("Is the motor seizure of Acme Facilities' Aquaflow X200 pump (serial AQ-X200-55871) on "
            "14 March 2026 covered under warranty?")


def ids_where(view: dict[str, Any], needle: str, kind: str | None = None, doc_contains: str | None = None) -> list[str]:
    out = []
    for e in view["evidence_ledger"]:
        if needle.lower() in e["text"].lower() and (kind is None or e["kind"] == kind) and (
                doc_contains is None or doc_contains.lower() in e["doc"].lower()):
            out.append(e["id"])
    return out


def sq(id_, text, status="OPEN", ev=None, note=""):
    return {"id": id_, "text": text, "status": status, "evidence_ids": ev or [], "note": note}


SUBQ_TEXT = {
    "q1": "Which warranty period applies to this unit (policy v1 vs amendment)?",
    "q2": "Is the unit used commercially?",
    "q3": "Is motor seizure a failure rather than normal wear?",
    "q4": "What is the purchase date?",
}


def _step(subqs, action, args=None, reason="", **extra):
    return {"sub_questions": subqs, "action": {"type": action, "args": args or {}, "reason": reason}, **extra}


def warranty_run1() -> list:
    """Start -> conflict -> metadata -> scanned contract -> manual (decoy filtered) -> ask for receipt."""

    def s1(user):
        return _step([sq(k, v) for k, v in SUBQ_TEXT.items()], "search",
                     {"query": "warranty period months commercial"}, "find the governing warranty terms")

    def s2(user):
        v = parse_view(user)
        both = ids_where(v, "twenty-four (24) months", doc_contains="Policy v1") + ids_where(v, "twelve (12) months")
        return _step([sq("q1", SUBQ_TEXT["q1"], "CONFLICTING", both, "policy says 24 months, amendment 12"),
                      sq("q2", SUBQ_TEXT["q2"]), sq("q3", SUBQ_TEXT["q3"]), sq("q4", SUBQ_TEXT["q4"])],
                     "get_metadata", {"doc_type": "amendment"}, "check effective date / supersedes")

    def s3(user):
        v = parse_view(user)
        q1 = ids_where(v, "twelve (12) months") + ids_where(v, "supersedes=AQF-WP-1.0", kind="metadata")
        return _step([sq("q1", SUBQ_TEXT["q1"], "SUPPORTED", q1, "amendment supersedes s.2 for 2025+ purchases; "
                                                                     "commercial units 12 months"),
                      sq("q2", SUBQ_TEXT["q2"]), sq("q3", SUBQ_TEXT["q3"]), sq("q4", SUBQ_TEXT["q4"])],
                     "process_document", {"doc_id": _contract_id(v), "mode": "extract_only",
                                          "fields": ["usage_classification"]},
                     "usage classification is on the scanned Schedule B",
                     conflict_resolution="Amendment AQF-WP-1.0-A1 (effective 2025-01-01) supersedes section 2 of "
                                         "AQF-WP-1.0 for units bought on/after 2025-01-01.")

    def s4(user):
        v = parse_view(user)
        q1 = ids_where(v, "twelve (12) months") + ids_where(v, "supersedes=AQF-WP-1.0", kind="metadata")
        q2 = ids_where(v, "usage_classification", kind="field")
        return _step([sq("q1", SUBQ_TEXT["q1"], "SUPPORTED", q1), sq("q2", SUBQ_TEXT["q2"], "SUPPORTED", q2),
                      sq("q3", SUBQ_TEXT["q3"]), sq("q4", SUBQ_TEXT["q4"])],
                     "search", {"query": "motor seizure failure normal wear"}, "classify the failure")

    def s5(user):
        v = parse_view(user)
        # The X300 decoy matched too; re-query restricted to the X200 product.
        q1 = ids_where(v, "twelve (12) months") + ids_where(v, "supersedes=AQF-WP-1.0", kind="metadata")
        q2 = ids_where(v, "usage_classification", kind="field")
        return _step([sq("q1", SUBQ_TEXT["q1"], "SUPPORTED", q1), sq("q2", SUBQ_TEXT["q2"], "SUPPORTED", q2),
                      sq("q3", SUBQ_TEXT["q3"], "OPEN", note="X300 manual is a different model"),
                      sq("q4", SUBQ_TEXT["q4"])],
                     "search", {"query": "motor seizure failure normal wear", "product": "X200"},
                     "exclude the X300 decoy")

    def s6(user):
        v = parse_view(user)
        return _step(run1_subqs(v, q4="MISSING"), "ask_user",
                     {"question": "Please upload the original sales receipt showing the purchase date.",
                      "request_upload": True}, "purchase date is not in any document")

    return [s1, s2, s3, s4, s5, s6]


def _contract_id(view) -> str:
    for d in view.get("corpus", []):          # the planner sees the document catalog
        if d["type"] == "contract":
            return d["doc_id"]
    raise AssertionError("contract not in corpus view")


def run1_subqs(v, q4: str, q4_ev=None, q4_note=""):
    q1 = ids_where(v, "twelve (12) months") + ids_where(v, "supersedes=AQF-WP-1.0", kind="metadata")
    q2 = ids_where(v, "usage_classification", kind="field")
    q3 = ids_where(v, "motor seizure", doc_contains="X200 Service")
    return [sq("q1", SUBQ_TEXT["q1"], "SUPPORTED", q1), sq("q2", SUBQ_TEXT["q2"], "SUPPORTED", q2),
            sq("q3", SUBQ_TEXT["q3"], "SUPPORTED", q3),
            sq("q4", SUBQ_TEXT["q4"], q4, q4_ev or [], q4_note)]


def warranty_run2(robust: bool) -> list:
    """After the upload: extract the purchase date, then finalize."""

    def s7(user):
        v = parse_view(user)
        doc_id = v["attachments"][-1]["doc_id"]
        return _step(run1_subqs(v, q4="MISSING"), "process_document",
                     {"doc_id": doc_id, "mode": "extract_only", "fields": ["purchase_date", "receipt_number"]},
                     "one-off evidence: extract only")

    def s8(user):
        v = parse_view(user)
        q4 = ids_where(v, "purchase_date", kind="field")
        # The planner claims SUPPORTED; the guard downgrades it if the evidence is unverified OCR.
        return _step(run1_subqs(v, q4="SUPPORTED", q4_ev=q4, q4_note="from receipt"), "finalize", {},
                     "all facts gathered", robust_to_unverified=robust)

    return [s7, s8]

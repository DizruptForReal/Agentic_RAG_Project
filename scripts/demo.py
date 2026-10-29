"""Run the required demonstration against a running API.

    python -m scripts.demo                      # hinge receipt (expected: cannot determine reliably)
    python -m scripts.demo --receipt nonhinge   # clean date (expected: cited answer, not covered)
    python -m scripts.demo --skip-refusal       # skip the X500 question

Prerequisites: Elasticsearch running, corpus seeded, API on http://localhost:8000.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import httpx

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
QUESTION = ("Is the motor seizure of Acme Facilities' Aquaflow X200 pump (serial AQ-X200-55871) on "
            "14 March 2026 covered under warranty?")
REFUSAL_QUESTION = "What is the warranty period for the Aquaflow X500?"


def show(body: dict, title: str) -> None:
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)
    print(f"investigation {body['id']}  status={body['status']}  outcome={body['outcome']}")
    print("\nSteps:")
    for s in body["steps"]:
        a, o = s.get("action"), s.get("observation")
        if a is None:
            print(f"  {s['idx']:>2}. (planner fallback) {s['notes']}")
            continue
        args = json.dumps(a["args"])[:90]
        res = f"{o['status']}: {(o.get('summary') or o.get('error') or '')[:110]}" if o else "finalize"
        print(f"  {s['idx']:>2}. {a['type']:<16} {args}\n      -> {res}")
        for n in s.get("notes", []):
            print(f"      ! {n}")
    print("\nSub-questions:")
    for q in body["sub_questions"]:
        print(f"  {q['id']} [{q['status']}] {q['text']}  evidence={q['evidence_ids']}")
        if q.get("note"):
            print(f"       note: {q['note']}")
    if body["status"] == "NEEDS_INPUT":
        print(f"\nAGENT ASKS: {body['pending_question']}")
        return
    ans = body.get("answer")
    if ans:
        print("\nANSWER:\n" + textwrap.indent(ans["answer"], "  "))
        print("\nSources:")
        for c in ans["citations"]:
            print(f"  [{c['evidence_id']}] {c['doc_name']} p.{c['page']} ({c['verification']})")
        for c in ans["caveats"]:
            print(f"  caveat: {c}")
        print("\nWhy this outcome:")
        for r in ans["reasons"]:
            print(f"  - {r}")
    if body.get("error"):
        print(f"\nERROR: {body['error']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--receipt", choices=["hinge", "nonhinge"], default="hinge")
    ap.add_argument("--skip-refusal", action="store_true")
    args = ap.parse_args()
    with httpx.Client(base_url=args.api, timeout=1800) as http:
        print(json.dumps(http.get("/health").json(), indent=2))
        r = http.post("/investigations", data={"question": QUESTION})
        r.raise_for_status()
        body = r.json()
        show(body, "1) Question across several documents")
        if body["status"] == "NEEDS_INPUT":
            path = SAMPLES / f"receipt_{args.receipt}.png"
            with path.open("rb") as f:
                r = http.post(f"/investigations/{body['id']}/continue",
                              data={"message": "Here is the scanned sales receipt."},
                              files={"file": (path.name, f, "image/png")})
            r.raise_for_status()
            show(r.json(), f"2) After uploading {path.name}")
        if not args.skip_refusal:
            r = http.post("/investigations", data={"question": REFUSAL_QUESTION})
            r.raise_for_status()
            show(r.json(), "3) A question the resources cannot answer")


if __name__ == "__main__":
    main()

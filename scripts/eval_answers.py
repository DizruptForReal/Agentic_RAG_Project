"""End-to-end evaluation: run the whole agent (retrieval + LLM + guards) on a question set and score it.

    python -m scripts.eval_answers                    # held-out set, all questions
    python -m scripts.eval_answers --limit 10         # first 10 (useful on free-tier rate limits)
    python -m scripts.eval_answers --cat yes_no       # one category
    python -m scripts.eval_answers --set regression

Needs Elasticsearch, the seeded corpus and the LLM configured in .env (no API server needed). For each
question it records the outcome, whether that is the expected kind (answer vs refuse), whether the answer
states the required facts, the number of steps and model calls, and the time. Writes
data/eval_answers_<set>_<model>.json (one report per model; compare them with
python -m scripts.compare_models). This measures behaviour by question type instead of trusting individual fixes.
"""
from __future__ import annotations

import argparse
import re
import json
import time

from app.agent.models import Outcome
from app.config import Settings
from app.container import build_container
from scripts.eval_retrieval import SETS, load_questions

ANSWERED = {Outcome.ANSWERED.value, Outcome.ANSWERED_WITH_CAVEATS.value}


def _norm(text: str) -> str:
    import re
    import unicodedata

    text = unicodedata.normalize("NFKC", text)          # narrow/no-break spaces -> spaces, fullwidth -> ASCII
    text = re.sub(r"[\u2010-\u2015]", "-", text)        # dashes
    return re.sub(r"\s+", " ", text.replace("**", "")).lower()


def has_fact(answer: str, alternatives: str) -> bool:
    text = _norm(answer)
    return any(_norm(a).strip() in text for a in alternatives.split("|"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(SETS), default="heldout")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cat", default=None)
    args = ap.parse_args()
    s = Settings()
    c = build_container(s)
    c.startup()
    print(f"model: {s.llm_model} ({s.llm_base_url or s.llm_provider})")
    qs = [q for q in load_questions(args.set) if not args.cat or q["cat"] == args.cat]
    if args.limit:
        qs = qs[: args.limit]
    rows = []
    for n, item in enumerate(qs, start=1):
        t0 = time.time()
        state = c.investigator.run(c.investigator.create(item["q"]).id)
        f = state.final
        outcome = f.outcome.value if f else state.status.value
        want = item.get("outcome", "answer" if item["expect"] else "refuse")
        got = "answer" if outcome in ANSWERED else "refuse"
        facts = item.get("must_contain", [])
        facts_ok = all(has_fact(f.answer, m) for m in facts) if (f and got == "answer" and facts) else None
        row = {"q": item["q"], "cat": item["cat"], "expected": want, "outcome": outcome,
               "outcome_ok": got == want, "facts_ok": facts_ok, "steps": len(state.steps),
               "llm_calls": sum(1 for st in state.steps if st.llm_ms is not None or not st.planner_ok),
               "wait_s": round(sum((st.wait_ms or 0) for st in state.steps) / 1000, 1),
               "seconds": round(time.time() - t0, 1), "investigation": state.id,
               "answer": (f.answer[:400] if f else state.error), "reasons": f.reasons[:3] if f else []}
        rows.append(row)
        mark = "ok " if row["outcome_ok"] and facts_ok is not False else "BAD"
        print(f"[{n:2}/{len(qs)}] {mark} {item['cat']:12} {outcome:22} facts={facts_ok!s:5} "
              f"{row['seconds']:5.1f}s  {item['q'][:60]}")
    print("\nsummary")
    for cat in sorted({r["cat"] for r in rows}) + ["ALL"]:
        rs = rows if cat == "ALL" else [r for r in rows if r["cat"] == cat]
        ok = sum(r["outcome_ok"] for r in rs)
        fc = [r["facts_ok"] for r in rs if r["facts_ok"] is not None]
        print(f"  {cat:14} n={len(rs):2}  right outcome {ok}/{len(rs)}"
              + (f"  correct facts {sum(fc)}/{len(fc)}" if fc else "")
              + f"  avg {sum(r['seconds'] for r in rs) / len(rs):.1f}s")
    slug = re.sub(r"[^A-Za-z0-9.]+", "-", s.llm_model).strip("-")
    out = s.data_dir / f"eval_answers_{args.set}_{slug}.json"
    out.write_text(json.dumps({"set": args.set, "model": s.llm_model, "base_url": s.llm_base_url, "rows": rows},
                              indent=1), encoding="utf-8")
    print(f"\nfull report: {out}  (each row links to its investigation, viewable in the UI)")


if __name__ == "__main__":
    main()

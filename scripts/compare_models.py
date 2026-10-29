"""Compare answer-evaluation reports of different models side by side.

    python -m scripts.eval_answers            # once per model (change LLM_MODEL / provider in .env in between)
    python -m scripts.compare_models          # all data/eval_answers_heldout_*.json
    python -m scripts.compare_models --set regression

Per question category: right outcome (answer vs refuse), correct facts, and average seconds per question.
"""
from __future__ import annotations

import argparse
import json

from app.config import Settings


def summarise(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for cat in sorted({r["cat"] for r in rows}) + ["ALL"]:
        rs = rows if cat == "ALL" else [r for r in rows if r["cat"] == cat]
        facts = [r["facts_ok"] for r in rs if r["facts_ok"] is not None]
        out[cat] = {"n": len(rs), "outcome": sum(r["outcome_ok"] for r in rs),
                    "facts": (sum(facts), len(facts)), "secs": sum(r["seconds"] for r in rs) / len(rs)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="heldout")
    args = ap.parse_args()
    files = sorted(Settings().data_dir.glob(f"eval_answers_{args.set}_*.json"))
    if not files:
        raise SystemExit(f"no reports yet - run: python -m scripts.eval_answers --set {args.set}")
    reports = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        reports.append((d.get("model") or f.stem.split(f"{args.set}_", 1)[-1], summarise(d["rows"])))
    cats = sorted({c for _, s in reports for c in s if c != "ALL"}) + ["ALL"]
    width = max(18, *(len(m) + 2 for m, _ in reports))
    print(f"{'category':14}" + "".join(f"{m:>{width}}" for m, _ in reports))
    for cat in cats:
        cells = []
        for _, s in reports:
            r = s.get(cat)
            if not r:
                cells.append(f"{'-':>{width}}")
                continue
            facts = f" f{r['facts'][0]}/{r['facts'][1]}" if r["facts"][1] else ""
            cells.append(f"{f'{r['outcome']}/{r['n']}{facts} {r['secs']:.0f}s':>{width}}")
        print(f"{cat:14}" + "".join(cells))
    print("\ncells: right outcome / questions, f = correct facts / answers with required facts, average seconds")
    print("note: models run on different question subsets (--cat/--limit) are not directly comparable")


if __name__ == "__main__":
    main()

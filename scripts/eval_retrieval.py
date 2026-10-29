"""Measure retrieval quality on the real stack (Elasticsearch + BGE) and calibrate the relevance floor.

    python -m scripts.eval_retrieval                  # held-out set (default)
    python -m scripts.eval_retrieval --set regression # questions observed in live testing
    python -m scripts.eval_retrieval --set all

Needs Elasticsearch running and the corpus seeded. The held-out set (samples/heldout_questions.json) is
written per section and question type and is never used for tuning; the regression set holds questions that
failed during live testing. Results are reported overall and per question category.

For every question in samples/golden_questions.json it reports where the expected section ranks and what
score the reranker gives it, for four retrieval variants:

    app (fused)              - what the app uses: RRF of BM25, kNN and the reranker (on "document - section + text")
    hybrid+rerank (context)  - BM25 + kNN candidates re-sorted by the reranker alone
    hybrid+rerank (raw text) - same, reranker sees the bare chunk text
    hybrid (no rerank)       - RRF order only
    bm25 / knn only          - each retriever alone (recall of the candidate stage)

It then suggests a relevance floor from the score distributions of relevant and irrelevant passages, and
writes everything to data/eval_retrieval.json (so the results can be inspected or shared). No LLM is used.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from app.config import Settings
from app.container import build_container
from app.ingestion.chunker import clean_text, context_text
from app.retrieval.base import SearchFilters, Visibility
from app.retrieval.retriever import rrf_fuse

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
SETS = {"heldout": ["heldout_questions.json"], "regression": ["golden_questions.json"],
        "all": ["heldout_questions.json", "golden_questions.json"]}


def load_questions(which: str) -> list[dict]:
    out = []
    for name in SETS[which]:
        for q in json.loads((SAMPLES / name).read_text(encoding="utf-8"))["questions"]:
            out.append({**q, "cat": q.get("cat") or ("out_of_scope" if not q["expect"] else "regression"),
                        "set": name.split("_")[0]})
    return out


def is_expected(chunk: dict, expect: list[list[str]]) -> bool:
    name, section = chunk.get("doc_name", ""), chunk.get("section") or ""
    return any(d.lower() in name.lower() and section.lower().startswith(s.lower()) for d, s in expect)


def first_rank(chunks: list[dict], expect) -> int | None:
    for i, c in enumerate(chunks, start=1):
        if is_expected(c, expect):
            return i
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(SETS), default="heldout")
    args = ap.parse_args()
    s = Settings()
    c = build_container(s)
    c.index.ensure_index()
    golden = load_questions(args.set)
    vis, flt, k = Visibility(), SearchFilters(), s.retrieve_candidates
    rows, rel_scores, irrel_scores, neg_top = [], [], [], []
    variants = ["app", "context", "raw", "rrf", "bm25", "knn"]
    ranks = {v: [] for v in variants}
    t0 = time.time()
    for item in golden:
        q, expect = item["q"], item["expect"]
        bm = c.index.bm25(q, flt, vis, k)
        kn = c.index.knn(c.embedder.embed_query(q), flt, vis, k)
        by_id = {x["chunk_id"]: x for x in bm + kn}
        fused = rrf_fuse([[x["chunk_id"] for x in bm], [x["chunk_id"] for x in kn]], s.rrf_k)
        order = [by_id[i] for i in sorted(fused, key=fused.get, reverse=True)][:k]
        ctx = c.reranker.score(q, [context_text(x) for x in order])
        raw = c.reranker.score(q, [clean_text(x["text"]) for x in order])
        by_ctx = [x for _, x in sorted(zip(ctx, order), key=lambda p: -p[0])]
        by_raw = [x for _, x in sorted(zip(raw, order), key=lambda p: -p[0])]
        app_scores = rrf_fuse([[x["chunk_id"] for x in bm], [x["chunk_id"] for x in kn],
                               [x["chunk_id"] for x in by_ctx]], s.rrf_k)
        by_app = [by_id[i] for i in sorted(app_scores, key=app_scores.get, reverse=True)]
        lists = {"app": by_app, "context": by_ctx, "raw": by_raw, "rrf": order, "bm25": bm, "knn": kn}
        row = {"q": q, "cat": item["cat"], "expected": [f"{d} / {sec}" for d, sec in expect]}
        for v in variants:
            r = first_rank(lists[v], expect) if expect else None
            row[f"rank_{v}"] = r
            if expect:
                ranks[v].append(r)
        scored = sorted(zip(ctx, order), key=lambda p: -p[0])
        row["top"] = [{"score": round(sc, 4), "doc": x.get("doc_name"), "section": x.get("section"),
                       "relevant": bool(expect) and is_expected(x, expect)} for sc, x in scored[:6]]
        if expect:
            for sc, x in scored:
                (rel_scores if is_expected(x, expect) else irrel_scores).append(sc)
        else:
            neg_top.append(scored[0][0] if scored else 0.0)
        rows.append(row)

    def recall(v, n):
        rs = ranks[v]
        return sum(1 for r in rs if r is not None and r <= n) / len(rs)

    def mrr(v):
        rs = ranks[v]
        return sum(1 / r for r in rs if r) / len(rs)

    print(f"\n{len(golden)} questions ({sum(1 for g in golden if g['expect'])} answerable), "
          f"{time.time() - t0:.1f}s\n")
    print(f"{'variant':26} {'R@1':>6} {'R@3':>6} {'R@6':>6} {'R@20':>6} {'MRR':>6}")
    names = {"app": "app (fused, used now)", "context": "hybrid+rerank (context)", "raw": "hybrid+rerank (raw text)", "rrf": "hybrid (no rerank)",
             "bm25": "bm25 only", "knn": "knn only"}
    summary = {}
    for v in variants:
        summary[v] = {"r1": recall(v, 1), "r3": recall(v, 3), "r6": recall(v, 6), "r20": recall(v, 20), "mrr": mrr(v)}
        m = summary[v]
        print(f"{names[v]:26} {m['r1']:6.2f} {m['r3']:6.2f} {m['r6']:6.2f} {m['r20']:6.2f} {m['mrr']:6.2f}")

    def pct(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))] if xs else None

    best_rel = [max((t["score"] for t in r["top"] if t["relevant"]), default=None) for r in rows if r["expected"]]
    best_rel = [b for b in best_rel if b is not None]
    floor_hint = round(min(pct(best_rel, 0.05) or 0.03, 0.5) * 0.8, 3) if best_rel else s.relevance_floor
    print("\nreranker scores (context variant)")
    print(f"  best relevant passage per question: min {min(best_rel):.4f}  median {statistics.median(best_rel):.4f}")
    print(f"  irrelevant passages: median {statistics.median(irrel_scores):.4f}  90th pct {pct(irrel_scores, 0.9):.4f}")
    print(f"  top score for out-of-scope questions: {', '.join(f'{x:.4f}' for x in neg_top)}")
    print(f"\ncurrent RELEVANCE_FLOOR = {s.relevance_floor}; suggested <= {floor_hint} "
          f"(keeps the weakest correct passage above the floor)")
    cats = sorted({r["cat"] for r in rows if r["expected"]})
    per_cat = {}
    print(f"\nper category (app ranking)          n    R@1    R@3    R@6")
    for cat in cats:
        rs = [r["rank_app"] for r in rows if r["cat"] == cat and r["expected"]]
        per_cat[cat] = {n: sum(1 for r in rs if r and r <= n) / len(rs) for n in (1, 3, 6)}
        print(f"  {cat:32} {len(rs):3}  {per_cat[cat][1]:5.2f}  {per_cat[cat][3]:5.2f}  {per_cat[cat][6]:5.2f}")
    misses = [r for r in rows if r["expected"] and (r["rank_app"] is None or r["rank_app"] > 3)]
    if misses:
        print("\nquestions where the expected section is not in the top 3:")
        for r in misses:
            print(f"  - {r['q']}  (rank {r['rank_app']}; top: {r['top'][0]['section']} {r['top'][0]['score']})")
    out = s.data_dir / f"eval_retrieval_{args.set}.json"
    out.write_text(json.dumps({"set": args.set, "summary": summary, "per_category": per_cat, "relevance_floor": s.relevance_floor, "floor_hint": floor_hint,
                               "relevant_best": best_rel, "irrelevant": irrel_scores, "negatives_top": neg_top,
                               "rows": rows}, indent=1), encoding="utf-8")
    print(f"\nfull report: {out}")


if __name__ == "__main__":
    main()

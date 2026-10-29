"""Hybrid retrieval: BM25 + kNN -> reciprocal rank fusion (in Python) -> cross-encoder rerank.

Degrades instead of failing where it safely can:
- embedder down  -> BM25 only (warning)
- reranker down  -> fused order, rerank_score=None (warning; guards then treat chunks as unscored)
- index down     -> SearchUnavailable (a tool failure, never "no evidence")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.ingestion.chunker import context_text
from app.retrieval.base import Embedder, Reranker, SearchFilters, SearchIndex, Visibility  # noqa: F401

log = logging.getLogger(__name__)


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


@dataclass
class Hit:
    chunk: dict[str, Any]
    rrf_score: float
    rerank_score: float | None
    rank: int = 0            # 1-based position in the final ranking of this search


@dataclass
class RetrievalResult:
    hits: list[Hit]
    reranked: bool
    warnings: list[str] = field(default_factory=list)


class HybridRetriever:
    def __init__(self, index: SearchIndex, embedder: Embedder, reranker: Reranker, *,
                 candidates: int = 30, top_k: int = 6, rrf_k: int = 60):
        self.index, self.embedder, self.reranker = index, embedder, reranker
        self.candidates, self.top_k, self.rrf_k = candidates, top_k, rrf_k

    def search(self, query: str, filters: SearchFilters, vis: Visibility, top_k: int | None = None) -> RetrievalResult:
        top_k = top_k or self.top_k
        warnings: list[str] = []
        bm = self.index.bm25(query, filters, vis, self.candidates)       # SearchUnavailable propagates
        try:
            vec = self.embedder.embed_query(query)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"embedding unavailable, BM25 only: {exc}")
            vec = None
        kn = self.index.knn(vec, filters, vis, self.candidates) if vec is not None else []
        by_id = {c["chunk_id"]: c for c in bm + kn}
        fused = rrf_fuse([[c["chunk_id"] for c in bm], [c["chunk_id"] for c in kn]], self.rrf_k)
        ordered = sorted(fused, key=fused.get, reverse=True)[: self.candidates]
        if not ordered:
            return RetrievalResult([], reranked=False, warnings=warnings)
        try:
            scores = dict(zip(ordered, self.reranker.score(query, [context_text(by_id[i]) for i in ordered])))
            reranked = True
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"reranker unavailable, using fused order: {exc}")
            scores, reranked = {}, False
        if reranked:
            # The reranker is a third voter, not the judge: measured on the golden set it alone mis-ranked
            # correct sections that BM25 and kNN both put first (e.g. "what is covered": rerank 0.014).
            by_rerank = sorted(ordered, key=lambda i: scores[i], reverse=True)
            final = rrf_fuse([[c["chunk_id"] for c in bm], [c["chunk_id"] for c in kn], by_rerank], self.rrf_k)
        else:
            final = fused
        ranked = sorted(ordered, key=lambda i: final[i], reverse=True)
        hits = [Hit(by_id[i], final[i], scores.get(i), rank=n) for n, i in enumerate(ranked, start=1)]
        return RetrievalResult(hits[:top_k], reranked=reranked, warnings=warnings)

"""search: hybrid retrieval over the chunk index, visible to this investigation only."""
from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from app.agent.models import EvidenceItem, EvidenceKind, Observation, ObsStatus, Verification
from app.retrieval.base import SearchFilters, SearchUnavailable, Visibility
from app.retrieval.retriever import HybridRetriever
from app.tools.base import ToolContext, ToolError


TOP_RANKS = 3   # a passage in the top 3 of the fused ranking counts as relevant whatever the reranker says


def is_relevant(rerank: float | None, rank: int | None, floor: float) -> bool:
    """Relevant = the reranker scores it above the floor OR the fused ranking (BM25 + kNN + reranker) puts it in
    the top 3. Measured: the reranker alone gave correct sections as little as 0.014, while irrelevant passages
    reached 0.13 at the 90th percentile, so no single reranker threshold separates them."""
    if rerank is None:
        return True              # not reranked: no basis to reject
    return rerank >= floor or (rank is not None and rank <= TOP_RANKS)


class SearchArgs(BaseModel):
    query: str = Field(min_length=2)
    doc_type: str | None = None
    product: str | None = None
    doc_ids: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=10)


class SearchTool:
    name: ClassVar[str] = "search"
    description: ClassVar[str] = (
        "Hybrid (BM25 + vector) search over indexed document chunks, reranked. "
        "Optional filters: doc_type (policy|amendment|manual|contract|receipt), product (e.g. X200), doc_ids.")
    Args: ClassVar[type[BaseModel]] = SearchArgs

    def __init__(self, retriever: HybridRetriever, rerank_floor: float):
        self.retriever, self.floor = retriever, rerank_floor

    def run(self, args: SearchArgs, ctx: ToolContext) -> Observation:
        filters = SearchFilters(doc_type=args.doc_type, product=args.product, doc_ids=args.doc_ids)
        vis = Visibility.for_investigation(ctx.investigation_id, ctx.attachments)
        try:
            result = self.retriever.search(args.query, filters, vis, args.top_k)
        except SearchUnavailable as exc:
            raise ToolError(str(exc), retryable=True) from exc
        if not result.hits:
            return Observation(tool=self.name, status=ObsStatus.EMPTY, summary="no matching chunks",
                               data={"warnings": result.warnings})
        evidence = []
        for h in result.hits:
            c = h.chunk
            evidence.append(EvidenceItem(
                kind=EvidenceKind.CHUNK, doc_id=c["doc_id"], doc_name=c.get("doc_name", c["doc_id"]),
                page=c.get("page"), chunk_id=c["chunk_id"], text=c["text"],
                verification=Verification(c.get("verification", "VERIFIED")),
                ocr_confidence=c.get("min_conf"), retrieval_score=round(h.rrf_score, 5),
                rerank_score=None if h.rerank_score is None else round(h.rerank_score, 4),
                retrieval_rank=h.rank,
            ))
        strong = [h for h in result.hits if is_relevant(h.rerank_score, h.rank, self.floor)]
        summary = (f"{len(result.hits)} chunks from {len({h.chunk['doc_id'] for h in result.hits})} documents; "
                   f"{len(strong)} relevant")
        if not result.reranked:
            summary += " (NOT reranked - relevance unknown)"
        return Observation(tool=self.name, status=ObsStatus.OK, summary=summary, evidence=evidence,
                           data={"reranked": result.reranked, "warnings": result.warnings,
                                 "above_floor": len(strong)})

"""Hybrid retrieval: fusion, reranking, graceful degradation, visibility, provenance."""
from __future__ import annotations

import pytest

from app.retrieval.base import SearchFilters, SearchUnavailable, Visibility
from app.retrieval.es_client import build_filter
from app.retrieval.retriever import HybridRetriever, rrf_fuse
from tests.fakes import HashEmbedder, OverlapReranker


def test_rrf_rewards_agreement():
    s = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert s["a"] == s["b"] > s["c"] > 0 and s["c"] == s["d"]


def test_hits_keep_source_and_page(container, seeded):
    res = container.retriever.search("twelve months commercial warranty", SearchFilters(), Visibility())
    top = res.hits[0].chunk
    assert res.reranked and top["doc_id"] and top["page"] == 1 and "Amendment" in top["doc_name"]


def test_product_filter_excludes_decoy(container, seeded):
    res = container.retriever.search("motor seizure failure", SearchFilters(product="X200"), Visibility())
    assert res.hits and all(h.chunk["product"] == "x200" for h in res.hits)


def test_reranker_failure_degrades_to_fused_order(container, index, seeded):
    r = HybridRetriever(index, HashEmbedder(), OverlapReranker(fail=True))
    res = r.search("warranty period", SearchFilters(), Visibility())
    assert res.hits and not res.reranked and all(h.rerank_score is None for h in res.hits)
    assert "reranker unavailable" in res.warnings[0]


def test_embedder_failure_degrades_to_bm25(container, index, seeded):
    class Broken(HashEmbedder):
        def embed_query(self, text):
            raise RuntimeError("model download failed")

    res = HybridRetriever(index, Broken(), OverlapReranker()).search("warranty period", SearchFilters(), Visibility())
    assert res.hits and "BM25 only" in res.warnings[0]


def test_index_outage_raises(container, index, seeded):
    index.fail_next = 1
    with pytest.raises(SearchUnavailable):
        container.retriever.search("warranty", SearchFilters(), Visibility())


def test_session_documents_are_invisible_to_other_investigations(container, seeded):
    doc, _ = container.pipeline.register("note.txt", b"Secret session note about the zebrafish pump.",
                                         scope="session:inv_a")
    container.pipeline.ensure_indexed(doc["doc_id"])
    q = "zebrafish pump"
    mine = container.retriever.search(q, SearchFilters(), Visibility.for_investigation("inv_a"))
    other = container.retriever.search(q, SearchFilters(), Visibility.for_investigation("inv_b"))
    attached = container.retriever.search(q, SearchFilters(), Visibility.for_investigation("inv_b", [doc["doc_id"]]))
    assert any(h.chunk["doc_id"] == doc["doc_id"] for h in mine.hits)
    assert not any(h.chunk["doc_id"] == doc["doc_id"] for h in other.hits)
    assert any(h.chunk["doc_id"] == doc["doc_id"] for h in attached.hits)


def test_es_filter_shape():
    f = build_filter(SearchFilters(product="X200", doc_type="Manual"), Visibility(["global", "session:i"], ["d_1"]))
    assert {"term": {"product": "x200"}} in f["bool"]["filter"]
    assert f["bool"]["minimum_should_match"] == 1 and len(f["bool"]["should"]) == 2

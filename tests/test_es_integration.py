"""Optional: runs against a real Elasticsearch (docker compose up -d). Skipped if unreachable.

    ES_URL=http://localhost:9200 pytest tests/test_es_integration.py
"""
from __future__ import annotations

import os
import uuid

import pytest

ES_URL = os.environ.get("ES_URL", "http://localhost:9200")


def _es_up() -> bool:
    try:
        import httpx

        return httpx.get(ES_URL, timeout=2).status_code == 200
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _es_up(), reason=f"Elasticsearch not reachable at {ES_URL}")


def test_index_search_scope_roundtrip():
    from app.retrieval.base import SearchFilters, Visibility
    from app.retrieval.es_client import ElasticIndex
    from tests.fakes import HashEmbedder

    emb = HashEmbedder()
    idx = ElasticIndex(ES_URL, f"test-chunks-{uuid.uuid4().hex[:8]}", dims=emb.dims)
    try:
        chunks = [
            {"chunk_id": "d_a:1:0", "doc_id": "d_a", "doc_name": "A", "page": 1, "scope": "global",
             "text": "commercial units twelve months warranty", "doc_type": "amendment", "product": "x-series",
             "verification": "VERIFIED", "min_conf": None},
            {"chunk_id": "d_b:1:0", "doc_id": "d_b", "doc_name": "B", "page": 1, "scope": "session:i1",
             "text": "receipt purchase date warranty", "doc_type": "receipt", "product": "x200",
             "verification": "UNVERIFIED", "min_conf": 62.0},
        ]
        idx.upsert(chunks, emb.embed_documents([c["text"] for c in chunks]))
        idx.upsert(chunks, emb.embed_documents([c["text"] for c in chunks]))   # idempotent
        assert idx.count() == 2
        hits = idx.bm25("warranty", SearchFilters(), Visibility.for_investigation("i2"), 10)
        assert [h["doc_id"] for h in hits] == ["d_a"]                             # session doc hidden
        hits = idx.knn(emb.embed_query("purchase date"), SearchFilters(), Visibility.for_investigation("i1"), 10)
        assert "d_b" in [h["doc_id"] for h in hits]
        idx.set_scope("d_b", "global")
        assert len(idx.bm25("warranty", SearchFilters(), Visibility(), 10)) == 2
    finally:
        idx.es.indices.delete(index=idx.index, ignore_unavailable=True)

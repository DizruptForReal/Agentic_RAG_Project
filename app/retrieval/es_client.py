"""Elasticsearch chunk index: BM25 on text, dense_vector kNN, scope/attachment visibility filter."""
from __future__ import annotations

import logging
from typing import Any

from app.retrieval.base import SearchFilters, SearchUnavailable, Visibility

log = logging.getLogger(__name__)

SOURCE_FIELDS = ["chunk_id", "doc_id", "doc_name", "page", "text", "section", "scope", "doc_type", "product",
                 "verification", "min_conf"]


def build_filter(filters: SearchFilters, vis: Visibility) -> dict[str, Any]:
    must: list[dict] = []
    if filters.doc_type:
        must.append({"term": {"doc_type": filters.doc_type.lower()}})
    if filters.product:
        must.append({"term": {"product": filters.product.lower()}})
    if filters.doc_ids:
        must.append({"terms": {"doc_id": filters.doc_ids}})
    should = [{"terms": {"scope": vis.scopes}}]
    if vis.doc_ids:
        should.append({"terms": {"doc_id": vis.doc_ids}})
    return {"bool": {"filter": must, "should": should, "minimum_should_match": 1}}


class ElasticIndex:
    def __init__(self, url: str, index: str, dims: int, timeout_s: float = 10.0):
        from elasticsearch import Elasticsearch

        # No client-level retries: the tool layer already retries, and stacked retries turned one outage into
        # ~40 s of waiting (each refused connection to "localhost" can take seconds on Windows).
        self.es = Elasticsearch(url, request_timeout=timeout_s, retry_on_timeout=False, max_retries=0)
        self.index = index
        self.dims = dims
        self.url = url
        self._down_until = 0.0     # circuit breaker: after a connection failure, fail fast for a while
        self._down_reason = ""

    DOWN_FOR_S = 20.0

    def _call(self, fn, *args, **kwargs):
        import time

        from elasticsearch import ApiError, ConnectionError as ESConnectionError, TransportError

        now = time.monotonic()
        if now < self._down_until:
            raise SearchUnavailable(f"elasticsearch unreachable at {self.url} (connection failed moments ago: "
                                    f"{self._down_reason}); not retried for {self.DOWN_FOR_S:.0f}s")
        try:
            return fn(*args, **kwargs)
        except ESConnectionError as exc:
            self._down_until, self._down_reason = time.monotonic() + self.DOWN_FOR_S, str(exc)[:160]
            raise SearchUnavailable(f"elasticsearch unreachable at {self.url}: {exc}") from exc
        except (TransportError, ApiError) as exc:
            raise SearchUnavailable(f"elasticsearch: {exc}") from exc

    def ensure_index(self) -> None:
        if self._call(self.es.indices.exists, index=self.index):
            return
        mappings = {"properties": {
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "doc_name": {"type": "text", "analyzer": "english", "fields": {"raw": {"type": "keyword"}}},
            "page": {"type": "integer"},
            "text": {"type": "text", "analyzer": "english"},
            "section": {"type": "text", "analyzer": "english"},
            "scope": {"type": "keyword"},
            "doc_type": {"type": "keyword", "normalizer": "lower"},
            "product": {"type": "keyword", "normalizer": "lower"},
            "verification": {"type": "keyword"},
            "min_conf": {"type": "float"},
            "vector": {"type": "dense_vector", "dims": self.dims, "index": True, "similarity": "cosine"},
        }}
        settings = {"analysis": {"normalizer": {"lower": {"type": "custom", "filter": ["lowercase"]}}}}
        self._call(self.es.indices.create, index=self.index, mappings=mappings, settings=settings)

    def upsert(self, chunks: list[dict[str, Any]], vectors: list[list[float]]) -> None:
        from elasticsearch import helpers

        self.ensure_index()
        actions = [{"_op_type": "index", "_index": self.index, "_id": c["chunk_id"], "_source": {**c, "vector": v}}
                   for c, v in zip(chunks, vectors)]
        try:
            helpers.bulk(self.es, actions, refresh="wait_for")
        except Exception as exc:  # noqa: BLE001
            raise SearchUnavailable(f"bulk index failed: {exc}") from exc

    def _hits(self, resp) -> list[dict[str, Any]]:
        return [{**h["_source"], "_score": h["_score"]} for h in resp["hits"]["hits"]]

    def bm25(self, query: str, filters: SearchFilters, vis: Visibility, k: int) -> list[dict[str, Any]]:
        q = {"bool": {"must": [{"multi_match": {"query": query, "fields": ["text", "section^2", "doc_name"]}}],
                      "filter": [build_filter(filters, vis)]}}
        resp = self._call(self.es.search, index=self.index, query=q, size=k, source=SOURCE_FIELDS)
        return self._hits(resp)

    def knn(self, vector: list[float], filters: SearchFilters, vis: Visibility, k: int) -> list[dict[str, Any]]:
        knn = {"field": "vector", "query_vector": vector, "k": k, "num_candidates": max(50, k * 5),
               "filter": build_filter(filters, vis)}
        resp = self._call(self.es.search, index=self.index, knn=knn, size=k, source=SOURCE_FIELDS)
        return self._hits(resp)

    def set_scope(self, doc_id: str, scope: str) -> None:
        self._call(self.es.update_by_query, index=self.index, refresh=True,
                   query={"term": {"doc_id": doc_id}},
                   script={"source": "ctx._source.scope = params.s", "params": {"s": scope}})

    def count(self, doc_id: str | None = None) -> int:
        q = {"term": {"doc_id": doc_id}} if doc_id else {"match_all": {}}
        return self._call(self.es.count, index=self.index, query=q)["count"]

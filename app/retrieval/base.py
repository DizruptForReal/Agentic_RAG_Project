"""Retrieval interfaces shared by the real Elasticsearch index and the in-memory test index."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class SearchUnavailable(Exception):
    """The search backend could not be reached or rejected the request (transient)."""


@dataclass
class Visibility:
    """Which chunks an investigation may see: global ones, its own session ones, and its attachments."""
    scopes: list[str] = field(default_factory=lambda: ["global"])
    doc_ids: list[str] = field(default_factory=list)

    @classmethod
    def for_investigation(cls, inv_id: str | None, attachments: list[str] | None = None) -> "Visibility":
        scopes = ["global"] + ([f"session:{inv_id}"] if inv_id else [])
        return cls(scopes=scopes, doc_ids=list(attachments or []))

    def allows(self, chunk: dict[str, Any]) -> bool:
        return chunk.get("scope") in self.scopes or chunk.get("doc_id") in self.doc_ids


@dataclass
class SearchFilters:
    doc_type: str | None = None
    product: str | None = None
    doc_ids: list[str] | None = None

    def matches(self, chunk: dict[str, Any]) -> bool:
        if self.doc_type and (chunk.get("doc_type") or "").lower() != self.doc_type.lower():
            return False
        if self.product and (chunk.get("product") or "").lower() != self.product.lower():
            return False
        if self.doc_ids and chunk.get("doc_id") not in self.doc_ids:
            return False
        return True


class SearchIndex(Protocol):
    def ensure_index(self) -> None: ...
    def upsert(self, chunks: list[dict[str, Any]], vectors: list[list[float]]) -> None: ...
    def bm25(self, query: str, filters: SearchFilters, vis: Visibility, k: int) -> list[dict[str, Any]]: ...
    def knn(self, vector: list[float], filters: SearchFilters, vis: Visibility, k: int) -> list[dict[str, Any]]: ...
    def set_scope(self, doc_id: str, scope: str) -> None: ...
    def count(self, doc_id: str | None = None) -> int: ...


class Embedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance in 0..1, one per passage."""

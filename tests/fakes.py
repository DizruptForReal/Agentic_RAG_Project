"""Test doubles for every external dependency (LLM, Elasticsearch, embeddings, reranker, OCR)."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import deque
from typing import Any, Callable

from app.ingestion.chunker import clean_text
from app.ingestion.ocr import OCRError, OcrPage, OcrWord
from app.llm.base import LLMError
from app.retrieval.base import SearchFilters, SearchUnavailable, Visibility

_WORD = re.compile(r"[a-z0-9]+")
STOP = {"the", "and", "for", "are", "with", "this", "that", "from", "under", "what", "was", "were", "any",
        "its", "not", "all", "has", "have", "into", "per", "use", "each", "unit", "units"}


def tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall(clean_text(text).lower()) if len(t) > 1 and t not in STOP]


# ------------------------------------------------------------------------------------------ LLM
def parse_view(user_text: str) -> dict[str, Any]:
    """Extract the planner's JSON state view from its prompt (works for repair prompts too)."""
    start = user_text.index("{")
    end = user_text.index("\n\nReturn the updated")
    return json.loads(user_text[start:end])


Script = dict | str | Exception | Callable[[str], Any]


class ScriptedLLM:
    """Returns queued outputs per purpose. An item may be a dict, a raw string (to simulate malformed
    output), an exception to raise, or a callable(user_text) -> output."""

    def __init__(self, scripts: dict[str, list[Script]] | None = None):
        self.queues = {k: deque(v) for k, v in (scripts or {}).items()}
        self.calls: list[tuple[str, str]] = []

    def push(self, purpose: str, *items: Script) -> None:
        self.queues.setdefault(purpose, deque()).extend(items)

    def complete_json(self, *, purpose, system, user, schema, model=None):
        self.calls.append((purpose, user))
        q = self.queues.get(purpose)
        if not q:
            raise AssertionError(f"ScriptedLLM: no scripted output left for '{purpose}'")
        item = q.popleft()
        if isinstance(item, Exception):
            raise item
        return item(user) if callable(item) else item

    def count(self, purpose: str) -> int:
        return sum(1 for p, _ in self.calls if p == purpose)


def llm_down() -> LLMError:
    return LLMError("simulated provider outage")


# ------------------------------------------------------------------------------------------ retrieval
class HashEmbedder:
    dims = 64

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dims
        for t in tokens(text):
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % self.dims] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class OverlapReranker:
    """Fraction of query terms present in the passage (0..1)."""

    def __init__(self, fail: bool = False):
        self.fail = fail

    def score(self, query: str, passages: list[str]) -> list[float]:
        if self.fail:
            raise RuntimeError("reranker model not loaded")
        q = set(tokens(query))
        return [len(q & set(tokens(p))) / max(1, len(q)) for p in passages]


class InMemoryIndex:
    def __init__(self):
        self.docs: dict[str, dict[str, Any]] = {}
        self.vectors: dict[str, list[float]] = {}
        self.fail_next = 0       # number of upcoming search calls that raise SearchUnavailable
        self.search_calls = 0

    def ensure_index(self) -> None:
        pass

    def upsert(self, chunks, vectors) -> None:
        for c, v in zip(chunks, vectors):
            self.docs[c["chunk_id"]] = dict(c)
            self.vectors[c["chunk_id"]] = v

    def _visible(self, filters: SearchFilters, vis: Visibility):
        return [c for c in self.docs.values() if vis.allows(c) and filters.matches(c)]

    def _maybe_fail(self):
        self.search_calls += 1
        if self.fail_next > 0:
            self.fail_next -= 1
            raise SearchUnavailable("simulated: connection refused")

    def bm25(self, query, filters, vis, k):
        self._maybe_fail()
        q = set(tokens(query))
        scored = [(len(q & set(tokens(c["text"] + " " + (c.get("section") or "")))), c)
                  for c in self._visible(filters, vis)]
        scored = [(s, c) for s, c in scored if s > 0]
        scored.sort(key=lambda x: (-x[0], x[1]["chunk_id"]))
        return [{**c, "_score": float(s)} for s, c in scored[:k]]

    def knn(self, vector, filters, vis, k):
        def cos(a, b):
            return sum(x * y for x, y in zip(a, b))
        scored = [(cos(vector, self.vectors[c["chunk_id"]]), c) for c in self._visible(filters, vis)]
        scored = [(s, c) for s, c in scored if s > 0]
        scored.sort(key=lambda x: (-x[0], x[1]["chunk_id"]))
        return [{**c, "_score": s} for s, c in scored[:k]]

    def set_scope(self, doc_id, scope) -> None:
        for c in self.docs.values():
            if c["doc_id"] == doc_id:
                c["scope"] = scope

    def count(self, doc_id=None) -> int:
        return sum(1 for c in self.docs.values() if doc_id is None or c["doc_id"] == doc_id)


# ------------------------------------------------------------------------------------------ OCR
def ocr_page(text: str, low: dict[str, float] | None = None, default_conf: float = 93.0) -> OcrPage:
    """Build an OcrPage from text; words listed in `low` get that confidence."""
    low = low or {}
    return OcrPage([[OcrWord(w, low.get(w, default_conf)) for w in line.split()]
                    for line in text.splitlines() if line.strip()])


class FakeOCR:
    def __init__(self, default: OcrPage | None = None):
        self.default = default or ocr_page("scanned page")
        self.queue: deque[OcrPage | Exception] = deque()
        self.calls = 0

    def ocr(self, image) -> OcrPage:
        self.calls += 1
        if self.queue:
            item = self.queue.popleft()
            if isinstance(item, Exception):
                raise item
            return item
        return self.default


CONTRACT_SCAN = ocr_page(
    "SCHEDULE B - EQUIPMENT USAGE DECLARATION\n"
    "Agreement: ACME-LEASE-2025-014\n"
    "Unit: Aquaflow X200, serial AQ-X200-55871\n"
    "Usage classification: COMMERCIAL\n"
    "Purpose: facility water circulation\n"
    "Signed for the Lessee: J. Morgan", low={"Morgan": 55.0})

RECEIPT_TEXT = ("AQUAFLOW PUMPS LTD\nSALES RECEIPT\nReceipt No: {no}\nDate: {date}\n"
                "Customer: Acme Facilities Ltd\nItem: Aquaflow X200 Pump\nSerial: AQ-X200-55871\n"
                "Amount: GBP 4,850.00")


def receipt_hinge() -> OcrPage:
    return ocr_page(RECEIPT_TEXT.format(no="R-2025-04417", date="1-03-2025"), low={"1-03-2025": 62.0})


def receipt_nonhinge() -> OcrPage:
    return ocr_page(RECEIPT_TEXT.format(no="R-2025-03892", date="05-03-2025"), low={"R-2025-03892": 28.0})


def ocr_crash() -> OCRError:
    return OCRError("simulated tesseract crash")

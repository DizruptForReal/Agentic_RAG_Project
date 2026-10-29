"""BGE cross-encoder reranker. Scores are mapped to 0..1 so a single floor applies."""
from __future__ import annotations

import math
import threading


class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _get(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, max_length=512)
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        raw = [float(s) for s in self._get().predict([(query, p) for p in passages])]
        # Some sentence-transformers versions return logits, others sigmoid probabilities.
        if any(s < 0 or s > 1 for s in raw):
            raw = [1 / (1 + math.exp(-s)) for s in raw]
        return raw

"""BGE embeddings via sentence-transformers (loaded lazily on first use)."""
from __future__ import annotations

import threading

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class BGEEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _get(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_query(self, text: str) -> list[float]:
        # BGE v1.5 is trained with an instruction prefix on queries only.
        return self._get().encode(BGE_QUERY_PREFIX + text, normalize_embeddings=True).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._get().encode(texts, normalize_embeddings=True, batch_size=32).tolist()

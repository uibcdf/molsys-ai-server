
"""Embedding model wrapper for MolSys-AI RAG."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Protocol

import numpy as np


class EmbeddingModel(Protocol):
    def embed(self, texts: List[str]) -> List[List[float]]: ...


class SentenceTransformerEmbeddingModel:
    """Embedding model based on a sentence-transformers model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for SentenceTransformerEmbeddingModel "
                "but is not installed. Install it, or set MOLSYS_AI_EMBEDDINGS=hashing."
            ) from exc

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts into vectors."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


class HashingEmbeddingModel:
    """Lightweight, offline embedding model.

    This is a deterministic hashing-based baseline intended for:
    - development environments without `sentence-transformers`,
    - offline smoke tests,
    - keeping the docs-chat backend runnable even without embedding models.

    It is not intended for high-quality retrieval.
    """

    def __init__(self, dim: int = 384):
        self.dim = int(dim)

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for token in text.lower().split():
                h = hash(token)
                idx = h % self.dim
                vec[idx] += 1.0 if (h & 1) else -1.0
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
            vectors.append(vec.tolist())
        return vectors


@lru_cache()
def get_default_embedding_model() -> EmbeddingModel:
    """Return the default embedding model instance.

    - Default: sentence-transformers (`all-MiniLM-L6-v2`) when available.
    - Offline fallback: hashing baseline when sentence-transformers is missing.
    - Override via `MOLSYS_AI_EMBEDDINGS` ("sentence-transformers" | "hashing").
    """

    choice = (os.environ.get("MOLSYS_AI_EMBEDDINGS") or "sentence-transformers").strip().lower()
    if choice in {"hashing", "hash"}:
        return HashingEmbeddingModel()

    try:
        return SentenceTransformerEmbeddingModel()
    except Exception:
        return HashingEmbeddingModel()

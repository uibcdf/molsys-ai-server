
"""Embedding model wrapper for MolSys-AI RAG."""

from __future__ import annotations

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddingModel:
    """Embedding model based on a sentence-transformers model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts into vectors."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


@lru_cache()
def get_default_embedding_model() -> SentenceTransformerEmbeddingModel:
    """Return the default embedding model instance."""
    return SentenceTransformerEmbeddingModel()

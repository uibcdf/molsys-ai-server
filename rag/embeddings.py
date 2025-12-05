
"""Embedding model wrapper for MolSys-AI RAG.

The implementation will be decided later (e.g. a small HF model on GPU).
This module should hide those details from the rest of the codebase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class EmbeddingModel(ABC):
    """Abstract interface for text embedding models."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts into vectors."""


class DummyEmbeddingModel(EmbeddingModel):
    """Placeholder embedding model.

    It returns zero vectors and is only meant for testing wiring and
    prototyping the RAG API.
    """

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [[0.0] for _ in texts]


def get_default_embedding_model() -> EmbeddingModel:
    """Return the default embedding model instance.

    In the MVP this is a dummy implementation; later this will return a
    real model configured according to project settings and ADRs.
    """
    return DummyEmbeddingModel()

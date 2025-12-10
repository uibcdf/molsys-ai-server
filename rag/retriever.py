
"""Retrieval utilities for MolSys-AI."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .embeddings import get_default_embedding_model


@dataclass
class Document:
    """Small container for retrieved content and metadata."""

    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


def load_index(index_path: Path) -> List[Document]:
    """Load the RAG index from disk."""
    if not index_path.exists():
        return []
    with index_path.open("rb") as f:
        return pickle.load(f)


def retrieve(query: str, index: List[Document], k: int = 5) -> List[Document]:
    """Retrieve up to *k* documents relevant to the given query."""

    if not index or not query:
        return []

    embedding_model = get_default_embedding_model()
    query_embedding = np.array(embedding_model.embed([query])[0])

    # Normalize the query embedding
    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        return []
    query_embedding /= query_norm

    # Calculate cosine similarity
    scores = []
    for doc in index:
        if doc.embedding is not None:
            doc_embedding = np.array(doc.embedding)
            doc_norm = np.linalg.norm(doc_embedding)
            if doc_norm > 0:
                similarity = np.dot(query_embedding, doc_embedding) / doc_norm
                scores.append((similarity, doc))

    # Sort by score and return top-k
    scores.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scores[:k]]

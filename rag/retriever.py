
"""Retrieval utilities for MolSys-AI.

The retriever exposes a simple interface:

    retrieve(query: str, k: int = 5) -> list[Document]

where :class:`Document` is a small dataclass with ``content`` and ``metadata``.

For now this module provides placeholders that return no results; the goal is
to stabilise the public interface before wiring a real vector store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Document:
    """Small container for retrieved content and metadata.

    Attributes
    ----------
    content:
        The text content of the retrieved chunk.
    metadata:
        Arbitrary metadata associated with the chunk (e.g. source file,
        section, line numbers).
    """

    content: str
    metadata: Dict[str, Any]


_INDEX: List[Document] = []


def set_index(documents: List[Document]) -> None:
    """Replace the in-memory index with the given documents."""

    global _INDEX
    _INDEX = list(documents)


def retrieve(query: str, k: int = 5) -> List[Document]:
    """Retrieve up to *k* documents relevant to the given query.

    This MVP implementation performs a very simple in-memory ranking based on
    case-insensitive substring matches of the query in each document.
    A future implementation will query a FAISS index built from documentation.
    """

    if not _INDEX or not query:
        return []

    q = query.lower()

    scored: List[tuple[int, Document]] = []
    for doc in _INDEX:
        text = doc.content.lower()
        score = text.count(q)
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:k]]

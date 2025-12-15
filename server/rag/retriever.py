
"""Retrieval utilities for MolSys-AI."""

from __future__ import annotations

import os
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


def _tokenize(text: str) -> list[str]:
    # Simple tokenizer for hybrid retrieval (stdlib-only).
    # Keep dots/underscores to preserve identifiers like `molsysmt.structure.get_rmsd`.
    out: list[str] = []
    buf: list[str] = []
    for ch in (text or ""):
        if ch.isalnum() or ch in {"_", "."}:
            buf.append(ch.lower())
        else:
            if buf:
                out.append("".join(buf))
                buf = []
    if buf:
        out.append("".join(buf))
    return out


def _lexical_score(query_tokens: list[str], doc_text: str) -> float:
    if not query_tokens or not doc_text:
        return 0.0
    doc_lower = doc_text.lower()
    score = 0.0
    for t in query_tokens:
        if not t or len(t) < 3:
            continue
        if t in doc_lower:
            # Bonus for exact identifier-ish tokens.
            if "." in t or "_" in t:
                score += 3.0
            else:
                score += 1.0
    return score


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

    k = max(int(k), 1)
    max_per_source = int((os.environ.get("MOLSYS_AI_RAG_MAX_CHUNKS_PER_SOURCE") or "3").strip() or "3")
    max_per_source = max(max_per_source, 1)

    embedding_model = get_default_embedding_model()
    query_embedding = np.array(embedding_model.embed([query])[0])

    # Normalize the query embedding
    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        return []
    query_embedding /= query_norm

    # Calculate cosine similarity (with optional hybrid lexical rerank).
    base_scores: list[tuple[float, Document]] = []
    for doc in index:
        if doc.embedding is None:
            continue
        doc_embedding = np.array(doc.embedding)
        doc_norm = np.linalg.norm(doc_embedding)
        if doc_norm <= 0:
            continue
        similarity = float(np.dot(query_embedding, doc_embedding) / doc_norm)
        base_scores.append((similarity, doc))

    if not base_scores:
        return []

    # Preselect top-N by embedding similarity, then rerank with a lexical score.
    base_scores.sort(key=lambda item: item[0], reverse=True)
    preselect = min(len(base_scores), max(k * 10, 50))
    candidates = base_scores[:preselect]

    hybrid_weight = float((os.environ.get("MOLSYS_AI_RAG_HYBRID_WEIGHT") or "0.15").strip() or "0.15")
    hybrid_weight = min(max(hybrid_weight, 0.0), 1.0)

    q_tokens = _tokenize(query)
    reranked: list[tuple[float, Document]] = []
    for sim, doc in candidates:
        lex = _lexical_score(q_tokens, doc.content)
        # Normalize lex score by a soft factor so it doesn't dominate.
        lex_norm = lex / (lex + 10.0) if lex > 0 else 0.0
        score = (1.0 - hybrid_weight) * sim + hybrid_weight * lex_norm
        reranked.append((score, doc))

    scores = reranked

    # Sort by score and return top-k
    scores.sort(key=lambda item: item[0], reverse=True)
    out: list[Document] = []
    per_source: dict[str, int] = {}
    for _, doc in scores:
        source = str(doc.metadata.get("path") or "")
        if source:
            if per_source.get(source, 0) >= max_per_source:
                continue
            per_source[source] = per_source.get(source, 0) + 1
        out.append(doc)
        if len(out) >= k:
            break
    return out

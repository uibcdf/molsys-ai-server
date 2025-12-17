
"""Retrieval utilities for MolSys-AI."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .embeddings import get_default_embedding_model
from .bm25 import BM25Index, bm25_sidecar_path, load_bm25


@dataclass
class Document:
    """Small container for retrieved content and metadata."""

    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class RAGIndex(list):
    """A list-like container for Documents with optional sidecar indices."""

    def __init__(self, docs: Sequence[Document], *, index_path: Path | None = None) -> None:
        super().__init__(docs)
        self.index_path = index_path
        self.bm25: BM25Index | None = None


@dataclass(frozen=True)
class ScoredDocument:
    doc: Document
    score: float
    sim: float
    lex_norm: float
    bm25_norm: float
    bm25_raw: float


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
        return RAGIndex([], index_path=index_path)
    with index_path.open("rb") as f:
        docs = pickle.load(f)
    if not isinstance(docs, list):
        return RAGIndex([], index_path=index_path)
    out = RAGIndex(docs, index_path=index_path)
    # Optional BM25 sidecar: <index>.bm25.pkl
    try:
        out.bm25 = load_bm25(bm25_sidecar_path(index_path))
    except Exception:
        out.bm25 = None
    return out


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def retrieve_scored(query: str, index: List[Document], k: int = 5) -> List[ScoredDocument]:
    """Retrieve up to *k* documents relevant to the given query (with score breakdown)."""

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

    # Calculate cosine similarity.
    base_scores: list[Tuple[float, int]] = []
    sims: list[float] = []
    for i, doc in enumerate(index):
        if doc.embedding is None:
            sims.append(0.0)
            continue
        doc_embedding = np.array(doc.embedding)
        doc_norm = np.linalg.norm(doc_embedding)
        if doc_norm <= 0:
            sims.append(0.0)
            continue
        similarity = float(np.dot(query_embedding, doc_embedding) / doc_norm)
        sims.append(similarity)
        base_scores.append((similarity, i))

    if not base_scores:
        return []

    # Preselect top-N by embedding similarity.
    base_scores.sort(key=lambda item: item[0], reverse=True)
    preselect_factor = int((os.environ.get("MOLSYS_AI_RAG_EMBED_PRESELECT_FACTOR") or "10").strip() or "10")
    preselect_factor = max(preselect_factor, 1)
    preselect_min = int((os.environ.get("MOLSYS_AI_RAG_EMBED_PRESELECT_MIN") or "50").strip() or "50")
    preselect_min = max(preselect_min, 1)
    preselect = min(len(base_scores), max(k * preselect_factor, preselect_min))
    embed_candidates = base_scores[:preselect]
    candidate_ids: set[int] = {i for _, i in embed_candidates}

    # Optional BM25 lexical candidates: complements embeddings for identifier-heavy queries.
    bm25_weight = _env_float("MOLSYS_AI_RAG_BM25_WEIGHT", 0.0)
    bm25_weight = min(max(bm25_weight, 0.0), 1.0)
    bm25_scores: dict[int, float] = {}
    bm25: BM25Index | None = getattr(index, "bm25", None)  # type: ignore[attr-defined]
    if bm25 is not None and bm25_weight > 0.0:
        bm25_factor = int((os.environ.get("MOLSYS_AI_RAG_BM25_PRESELECT_FACTOR") or "30").strip() or "30")
        bm25_factor = max(bm25_factor, 1)
        bm25_min = int((os.environ.get("MOLSYS_AI_RAG_BM25_PRESELECT_MIN") or "200").strip() or "200")
        bm25_min = max(bm25_min, 1)
        top_bm25 = max(k * bm25_factor, bm25_min)
        for doc_id, score in bm25.search(query, top_n=top_bm25):
            bm25_scores[int(doc_id)] = float(score)
            candidate_ids.add(int(doc_id))

    hybrid_weight = _env_float("MOLSYS_AI_RAG_HYBRID_WEIGHT", 0.15)
    hybrid_weight = min(max(hybrid_weight, 0.0), 1.0)

    q_tokens = _tokenize(query)
    max_bm25 = max(bm25_scores.values()) if bm25_scores else 0.0
    reranked: list[tuple[float, int, float, float, float, float]] = []
    for doc_id in candidate_ids:
        if doc_id < 0 or doc_id >= len(index):
            continue
        doc = index[doc_id]
        sim = float(sims[doc_id]) if doc_id < len(sims) else 0.0
        sim = max(sim, 0.0)

        lex = _lexical_score(q_tokens, doc.content)
        lex_norm = lex / (lex + 10.0) if lex > 0 else 0.0

        bm = float(bm25_scores.get(doc_id, 0.0))
        bm_norm = (bm / max_bm25) if (bm > 0 and max_bm25 > 0) else 0.0

        # 3-way mixture: embeddings + lexical boost + BM25.
        embed_weight = max(0.0, 1.0 - hybrid_weight - bm25_weight)
        total = embed_weight + hybrid_weight + bm25_weight
        if total <= 0:
            embed_weight, total = 1.0, 1.0
        embed_w = embed_weight / total
        lex_w = hybrid_weight / total
        bm_w = bm25_weight / total
        score = embed_w * sim + lex_w * lex_norm + bm_w * bm_norm
        reranked.append((float(score), doc_id, float(sim), float(lex_norm), float(bm_norm), float(bm)))

    # Sort by score and return top-k.
    reranked.sort(key=lambda item: item[0], reverse=True)
    out: list[ScoredDocument] = []
    per_source: dict[str, int] = {}
    for score, doc_id, sim, lex_norm, bm_norm, bm_raw in reranked:
        doc = index[doc_id]
        source = str(doc.metadata.get("path") or "")
        if source:
            if per_source.get(source, 0) >= max_per_source:
                continue
            per_source[source] = per_source.get(source, 0) + 1
        out.append(
            ScoredDocument(
                doc=doc,
                score=float(score),
                sim=float(sim),
                lex_norm=float(lex_norm),
                bm25_norm=float(bm_norm),
                bm25_raw=float(bm_raw),
            )
        )
        if len(out) >= k:
            break
    return out


def retrieve(query: str, index: List[Document], k: int = 5) -> List[Document]:
    """Retrieve up to *k* documents relevant to the given query."""

    return [sd.doc for sd in retrieve_scored(query, index, k=k)]

"""BM25 lexical index for MolSys-AI RAG (stdlib-only).

This module provides an offline-built lexical index to complement embedding-based
retrieval. It is designed to work without network access and without additional
dependencies beyond the Python standard library.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _tokenize(text: str) -> list[str]:
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


def _keep_token(tok: str) -> bool:
    t = (tok or "").strip()
    if not t:
        return False
    if len(t) < 3:
        return False
    if len(t) > 96:
        return False
    return True


@dataclass(frozen=True)
class BM25Index:
    """Compact BM25 inverted index.

    - `postings`: term -> (doc_ids, term_freqs)
    - `doc_len`: document length in tokens
    - `idf`: precomputed IDF values for each term
    """

    postings: Dict[str, Tuple[List[int], List[int]]]
    doc_len: List[int]
    avgdl: float
    idf: Dict[str, float]
    k1: float = 1.2
    b: float = 0.75

    @property
    def n_docs(self) -> int:
        return int(len(self.doc_len))

    def search(self, query: str, *, top_n: int) -> List[Tuple[int, float]]:
        q_tokens = [t for t in _tokenize(query) if _keep_token(t)]
        if not q_tokens:
            return []

        scores: dict[int, float] = {}
        avgdl = float(self.avgdl) if self.avgdl > 0 else 1.0
        k1 = float(self.k1)
        b = float(self.b)

        for term in set(q_tokens):
            post = self.postings.get(term)
            if not post:
                continue
            doc_ids, tfs = post
            idf = float(self.idf.get(term, 0.0))
            if idf == 0.0:
                continue
            for doc_id, tf in zip(doc_ids, tfs):
                dl = float(self.doc_len[doc_id]) if doc_id < len(self.doc_len) else 0.0
                denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
                if denom <= 0:
                    continue
                inc = idf * (tf * (k1 + 1.0)) / denom
                scores[doc_id] = scores.get(doc_id, 0.0) + float(inc)

        if not scores:
            return []

        # Return top-N doc ids with their BM25 scores.
        top_n = max(int(top_n), 1)
        # Avoid importing heapq for tiny dicts.
        items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return items[:top_n]


def build_bm25_index(
    docs: Iterable[str],
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> BM25Index:
    """Build a BM25 index from an iterable of document texts."""
    postings: Dict[str, Tuple[List[int], List[int]]] = {}
    df: Dict[str, int] = {}
    doc_len: List[int] = []

    for doc_id, text in enumerate(docs):
        tokens = [t for t in _tokenize(text) if _keep_token(t)]
        doc_len.append(int(len(tokens)))
        if not tokens:
            continue
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for term, freq in tf.items():
            df[term] = df.get(term, 0) + 1
            doc_ids, tfs = postings.get(term, ([], []))
            doc_ids.append(int(doc_id))
            tfs.append(int(freq))
            postings[term] = (doc_ids, tfs)

    n_docs = len(doc_len)
    avgdl = (sum(doc_len) / n_docs) if n_docs else 0.0

    idf: Dict[str, float] = {}
    for term, dfi in df.items():
        # Standard BM25 IDF with +1 inside log for positivity.
        idf[term] = math.log((n_docs - dfi + 0.5) / (dfi + 0.5) + 1.0) if n_docs else 0.0

    return BM25Index(postings=postings, doc_len=doc_len, avgdl=float(avgdl), idf=idf, k1=float(k1), b=float(b))


def bm25_sidecar_path(index_path: Path) -> Path:
    # rag_index.pkl -> rag_index.bm25.pkl
    return index_path.with_suffix(".bm25.pkl")


def save_bm25(index: BM25Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(index, f)


def load_bm25(path: Path) -> BM25Index | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            obj = pickle.load(f)
    except Exception:
        return None
    return obj if isinstance(obj, BM25Index) else None


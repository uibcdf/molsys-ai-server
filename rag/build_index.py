
"""Index building utilities for MolSys-AI RAG.

In the MVP this will:
- read Sphinx-generated HTML/text,
- chunk it,
- embed chunks using a chosen embedding model,
- store them in a FAISS index.

For now we provide a placeholder API so that other parts of the codebase
can depend on it without requiring a concrete implementation.
"""

from __future__ import annotations

from pathlib import Path

from .retriever import Document, set_index


def build_index(source_dir: Path, index_path: Path) -> None:
    """Build or rebuild the RAG index from the given source directory.

    Parameters
    ----------
    source_dir:
        Directory containing documentation artifacts (e.g. Sphinx HTML/text).
    index_path:
        Path where the index should be stored on disk.

    Notes
    -----
    This MVP implementation:
    - reads ``*.txt`` files under ``source_dir``,
    - creates :class:`Document` objects with basic metadata,
    - and stores them in an in-memory index used by :func:`rag.retriever.retrieve`.

    The ``index_path`` argument is currently unused; a future implementation
    will persist a FAISS index at that location.
    """
    docs: list[Document] = []

    if not source_dir.exists():
        set_index([])
        return None

    for path in sorted(source_dir.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue

        docs.append(Document(content=text, metadata={"path": str(path)}))

    set_index(docs)
    return None

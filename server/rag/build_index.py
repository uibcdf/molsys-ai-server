
"""Index building utilities for MolSys-AI RAG."""

from __future__ import annotations

import os
import json
import pickle
from pathlib import Path
from typing import Iterable, Tuple

from .embeddings import get_default_embedding_model
from .retriever import Document


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _looks_like_myst_label(line: str) -> str | None:
    s = line.strip()
    if not (s.startswith("(") and s.endswith(")=")):
        return None
    inner = s[1:-2].strip()
    if not inner:
        return None
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.:")
    if any(ch not in allowed for ch in inner):
        return None
    return inner


def _load_ipynb_markdown(path: Path) -> str:
    """Extract Markdown text from a notebook (markdown cells only)."""
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    cells = nb.get("cells") or []
    parts: list[str] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        if cell.get("cell_type") != "markdown":
            continue
        src = cell.get("source")
        if isinstance(src, list):
            parts.append("".join(src))
        elif isinstance(src, str):
            parts.append(src)
    return "\n\n".join(p.strip() for p in parts if isinstance(p, str) and p.strip())


def _iter_text_chunks(text: str, *, max_chars: int, min_chars: int) -> Iterable[Tuple[str, dict]]:
    """Yield (chunk_text, metadata) from Markdown/RST-like text.

    MVP chunking strategy:
    - track the most recent Markdown heading as section context,
    - track explicit MyST labels of the form `(Label)=` as stable anchors,
    - split by blank lines (paragraphs),
    - merge paragraphs into chunks with a max character limit,
    - prefix each chunk with its section title for better retrieval.
    """

    # Parse into paragraphs tagged with the current section + label.
    current_section: str | None = None
    current_label: str | None = None
    pending_label: str | None = None

    def flush_chunk(buf: list[str], *, section: str | None, label: str | None) -> list[Tuple[str, dict]]:
        if not buf:
            return []
        body = "\n\n".join(buf).strip()
        buf.clear()
        if not body:
            return []
        meta: dict = {}
        if section:
            meta["section"] = section
        if label:
            meta["label"] = label
        if section:
            return [(f"{section}\n\n{body}".strip(), meta)]
        return [(body, meta)]

    out: list[Tuple[str, dict]] = []
    chunk_buf: list[str] = []
    chunk_section: str | None = None
    chunk_label: str | None = None

    lines = text.splitlines()
    i = 0
    para_lines: list[str] = []

    def flush_para() -> None:
        nonlocal para_lines, chunk_section, chunk_label, chunk_buf, out
        paragraph = "\n".join(para_lines).strip()
        para_lines = []
        if not paragraph:
            return

        # Start a new chunk when section/label changes.
        if (chunk_section, chunk_label) != (current_section, current_label) and chunk_buf:
            out.extend(flush_chunk(chunk_buf, section=chunk_section, label=chunk_label))
        chunk_section = current_section
        chunk_label = current_label

        tentative = "\n\n".join([*chunk_buf, paragraph]).strip()
        if len(tentative) > max_chars and chunk_buf:
            out.extend(flush_chunk(chunk_buf, section=chunk_section, label=chunk_label))
        chunk_buf.append(paragraph)

        # Avoid runaway chunk buffers.
        if len(paragraph) >= max_chars:
            out.extend(flush_chunk(chunk_buf, section=chunk_section, label=chunk_label))
        if len("\n\n".join(chunk_buf)) >= max_chars:
            out.extend(flush_chunk(chunk_buf, section=chunk_section, label=chunk_label))

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # MyST label line: (Label)=
        lab = _looks_like_myst_label(stripped)
        if lab:
            pending_label = lab
            i += 1
            continue

        # Markdown heading line: "# Title"
        if stripped.startswith("#"):
            flush_para()
            if chunk_buf:
                out.extend(flush_chunk(chunk_buf, section=chunk_section, label=chunk_label))
            current_section = stripped
            if pending_label:
                current_label = pending_label
                pending_label = None
            i += 1
            continue

        # RST heading: Title \n ======
        if i + 1 < len(lines):
            next_line = lines[i + 1].rstrip()
            if stripped and next_line and set(next_line.strip()) <= {"=", "-", "~", "^", "#", "*", "+"}:
                if len(next_line.strip()) >= max(3, len(stripped) - 1):
                    flush_para()
                    if chunk_buf:
                        out.extend(flush_chunk(chunk_buf, section=chunk_section, label=chunk_label))
                    current_section = stripped
                    if pending_label:
                        current_label = pending_label
                        pending_label = None
                    i += 2
                    continue

        # Paragraph boundaries.
        if not stripped:
            flush_para()
            i += 1
            continue

        para_lines.append(line)
        i += 1

    flush_para()
    if chunk_buf:
        out.extend(flush_chunk(chunk_buf, section=chunk_section, label=chunk_label))
    return out


def build_index(source_dir: Path, index_path: Path) -> None:
    """Build or rebuild the RAG index from the given source directory."""
    print(f"Building index from {source_dir}...")
    docs: list[Document] = []
    chunks: list[str] = []

    if not source_dir.exists():
        print(f"Source directory {source_dir} not found.")
        return

    max_chars = max(_env_int("MOLSYS_AI_RAG_CHUNK_MAX_CHARS", 1600), 200)
    min_chars = max(_env_int("MOLSYS_AI_RAG_CHUNK_MIN_CHARS", 400), 50)

    # Find and parse documentation files.
    exts = (".md", ".rst", ".txt", ".ipynb")
    paths: list[Path] = []
    for ext in exts:
        paths.extend(source_dir.rglob(f"*{ext}"))

    for path in sorted(paths):
        if ".ipynb_checkpoints" in str(path):
            continue
        try:
            if path.suffix.lower() == ".ipynb":
                text = _load_ipynb_markdown(path)
            else:
                text = path.read_text(encoding="utf-8")
            if not text.strip():
                continue
            for chunk_text, meta in _iter_text_chunks(text, max_chars=max_chars, min_chars=min_chars):
                if not chunk_text.strip():
                    continue
                metadata = {"path": str(path), **meta}
                docs.append(Document(content=chunk_text.strip(), metadata=metadata))
                chunks.append(chunk_text.strip())
        except OSError as e:
            print(f"Could not read {path}: {e}")
            continue

    if not docs:
        print("No documents found to index.")
        return

    print(f"Found {len(docs)} chunks to embed.")

    # Embed all chunks
    embedding_model = get_default_embedding_model()
    embeddings = embedding_model.embed(chunks)

    # Add embeddings to Document objects
    for doc, embedding in zip(docs, embeddings):
        # Store normalized vectors to make cosine similarity cheaper at query time.
        try:
            import numpy as np  # local import so build_index stays importable without numpy

            vec = np.asarray(embedding, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            doc.embedding = vec.tolist()
        except Exception:
            doc.embedding = embedding

    # Save the index to disk
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("wb") as f:
        pickle.dump(docs, f)

    print(f"Index built and saved to {index_path}")

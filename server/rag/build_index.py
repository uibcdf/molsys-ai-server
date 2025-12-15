
"""Index building utilities for MolSys-AI RAG."""

from __future__ import annotations

import os
import json
import pickle
import tempfile
from pathlib import Path
from typing import Iterable, Tuple
import multiprocessing as mp

from .embeddings import get_default_embedding_model, get_embedding_model
from .retriever import Document


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)

def _env_bool(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "y", "on"}


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


def _load_ipynb_text(path: Path) -> str:
    """Extract text from a notebook.

    Default behavior includes markdown cells and (optionally) code cells.
    Enable/disable code cells with:

      MOLSYS_AI_RAG_IPYNB_INCLUDE_CODE=1|0
    """
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    cells = nb.get("cells") or []
    parts: list[str] = []
    include_code = _env_bool("MOLSYS_AI_RAG_IPYNB_INCLUDE_CODE", True)
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        ctype = cell.get("cell_type")
        if ctype not in {"markdown", "code"}:
            continue
        if ctype == "code" and not include_code:
            continue
        src = cell.get("source")
        if isinstance(src, list):
            text = "".join(src)
        elif isinstance(src, str):
            text = src
        else:
            continue
        text = text.strip()
        if not text:
            continue
        if ctype == "code":
            parts.append("```python\n" + text + "\n```")
        else:
            parts.append(text)
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


def _iter_source_paths(source_dir: Path) -> list[Path]:
    exts = (".md", ".rst", ".txt", ".ipynb")
    paths: list[Path] = []
    for ext in exts:
        paths.extend(source_dir.rglob(f"*{ext}"))
    out: list[Path] = []
    for p in sorted(paths):
        if ".ipynb_checkpoints" in str(p):
            continue
        out.append(p)
    return out


def _classify_doc(rel_posix: str) -> str:
    p = (rel_posix or "").replace("\\", "/")
    if "/symbol_cards/" in p:
        return "symbol_card"
    if "/api_surface/" in p:
        return "api_surface"
    if "/recipes/" in p:
        return "recipe"
    return "docs"


def _build_docs_and_chunks(
    paths: list[Path],
    *,
    source_dir: Path,
    max_chars: int,
    min_chars: int,
) -> tuple[list[Document], list[str]]:
    docs: list[Document] = []
    chunks: list[str] = []
    for path in paths:
        try:
            if path.suffix.lower() == ".ipynb":
                text = _load_ipynb_text(path)
            else:
                text = path.read_text(encoding="utf-8")
            if not text.strip():
                continue
            for chunk_text, meta in _iter_text_chunks(text, max_chars=max_chars, min_chars=min_chars):
                if not chunk_text.strip():
                    continue
                metadata = {"path": str(path), **meta}
                try:
                    rel = path.resolve().relative_to(source_dir.resolve())
                    rel_posix = rel.as_posix()
                    metadata["relpath"] = rel_posix
                    if rel.parts:
                        metadata["project"] = rel.parts[0]
                    metadata["kind"] = _classify_doc(rel_posix)
                except Exception:
                    pass
                docs.append(Document(content=chunk_text.strip(), metadata=metadata))
                chunks.append(chunk_text.strip())
        except OSError as e:
            print(f"Could not read {path}: {e}")
            continue
    return docs, chunks


def _normalize_embeddings(docs: list[Document], embeddings: list[list[float]]) -> None:
    for doc, embedding in zip(docs, embeddings):
        try:
            import numpy as np  # local import so build_index stays importable without numpy

            vec = np.asarray(embedding, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            doc.embedding = vec.tolist()
        except Exception:
            doc.embedding = embedding


def build_index(source_dir: Path, index_path: Path) -> None:
    """Build or rebuild the RAG index from the given source directory."""
    print(f"Building index from {source_dir}...")

    if not source_dir.exists():
        print(f"Source directory {source_dir} not found.")
        return

    max_chars = max(_env_int("MOLSYS_AI_RAG_CHUNK_MAX_CHARS", 1600), 200)
    min_chars = max(_env_int("MOLSYS_AI_RAG_CHUNK_MIN_CHARS", 400), 50)

    paths = _iter_source_paths(source_dir)
    docs, chunks = _build_docs_and_chunks(paths, source_dir=source_dir, max_chars=max_chars, min_chars=min_chars)
    if not docs:
        print("No documents found to index.")
        return

    print(f"Found {len(docs)} chunks to embed.")

    embedding_model = get_default_embedding_model()
    embeddings = embedding_model.embed(chunks)
    _normalize_embeddings(docs, embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("wb") as f:
        pickle.dump(docs, f)

    print(f"Index built and saved to {index_path}")


def _split_paths_by_size(paths: list[Path], n: int) -> list[list[Path]]:
    if n <= 1:
        return [paths]

    buckets: list[list[Path]] = [[] for _ in range(n)]
    sizes: list[int] = [0 for _ in range(n)]
    for p in paths:
        try:
            sz = int(p.stat().st_size)
        except OSError:
            sz = 0
        i = min(range(n), key=lambda j: sizes[j])
        buckets[i].append(p)
        sizes[i] += sz
    return buckets


def _build_index_part(
    *,
    part_id: int,
    paths: list[Path],
    out_path: Path,
    source_dir: Path,
    device: str | None,
    max_chars: int,
    min_chars: int,
) -> None:
    # IMPORTANT: in a spawned process, set CUDA_VISIBLE_DEVICES before importing
    # torch/sentence-transformers (both may happen inside the embedding model).
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    docs, chunks = _build_docs_and_chunks(paths, source_dir=source_dir, max_chars=max_chars, min_chars=min_chars)
    if not docs:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            pickle.dump([], f)
        return

    embedding_model = get_embedding_model(device=("cuda" if device is not None else None))
    embeddings = embedding_model.embed(chunks)
    _normalize_embeddings(docs, embeddings)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(docs, f)

    print(f"[index:{part_id}] wrote {len(docs)} chunks to {out_path}")


def build_index_parallel(
    source_dir: Path,
    index_path: Path,
    *,
    devices: list[str] | None = None,
    workers: int | None = None,
    parts_dir: Path | None = None,
) -> None:
    """Build a RAG index using multiple worker processes (optionally multi-GPU).

    Strategy:
    - shard the source files (greedy by file size) across workers,
    - each worker reads+chunks its shard and computes embeddings,
    - workers write `rag_index.part-*.pkl`,
    - the parent process merges parts into `index_path`.

    Notes:
    - Use multiprocessing *spawn* for CUDA safety.
    - By default, sentence-transformers embeddings run on CPU in this codebase.
      This function can place workers on GPU if `devices` is provided.
    """

    print(f"Building index from {source_dir} (parallel)...")
    if not source_dir.exists():
        print(f"Source directory {source_dir} not found.")
        return

    max_chars = max(_env_int("MOLSYS_AI_RAG_CHUNK_MAX_CHARS", 1600), 200)
    min_chars = max(_env_int("MOLSYS_AI_RAG_CHUNK_MIN_CHARS", 400), 50)

    paths = _iter_source_paths(source_dir)
    if not paths:
        print("No documents found to index.")
        return

    if devices is None:
        devices = []
    if workers is None or workers <= 0:
        workers = max(len(devices), 1)
    workers = int(workers)
    if devices and workers != len(devices):
        raise ValueError("When `devices` is provided, `workers` must match len(devices).")

    shards = _split_paths_by_size(paths, workers)

    if parts_dir is None:
        parts_dir = Path(tempfile.mkdtemp(prefix="molsys_ai_rag_parts_"))
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_paths = [parts_dir / f"rag_index.part-{i}.pkl" for i in range(workers)]

    ctx = mp.get_context("spawn")
    procs: list[mp.Process] = []
    for i in range(workers):
        dev = devices[i] if devices else None
        p = ctx.Process(
            target=_build_index_part,
            kwargs={
                "part_id": i,
                "paths": shards[i],
                "out_path": part_paths[i],
                "source_dir": source_dir,
                "device": dev,
                "max_chars": max_chars,
                "min_chars": min_chars,
            },
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Index build worker failed with exit code {p.exitcode}.")

    merged: list[Document] = []
    for pp in part_paths:
        if not pp.exists():
            continue
        with pp.open("rb") as f:
            part = pickle.load(f)
        if isinstance(part, list):
            merged.extend(part)

    if not merged:
        print("No documents found to index.")
        return

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("wb") as f:
        pickle.dump(merged, f)

    print(f"Index built (parallel) with {len(merged)} chunks and saved to {index_path}")

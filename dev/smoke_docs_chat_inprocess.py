#!/usr/bin/env python3
"""In-process end-to-end smoke test: RAG retrieve -> prompt -> local model backend.

This avoids running uvicorn servers (useful in restricted environments) while still
validating:

- corpus snapshot exists,
- index loads,
- retrieval returns real sources,
- the configured model backend can answer and produce bracketed citations like [1].

Requires GPU access if `server/model_server/config.yaml` uses the vLLM backend.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    query = " ".join(argv).strip() or (
        "What is MolSysMT? Reply with one short sentence and include at least one citation like [1]."
    )

    os.environ.setdefault("PYTHONPATH", "server:client")
    os.environ.setdefault("MOLSYS_AI_EMBEDDINGS", "hashing")

    docs_dir = Path(os.environ.get("MOLSYS_AI_DOCS_DIR") or (REPO_ROOT / "server/docs_chat/data/docs")).resolve()
    index_path = Path(os.environ.get("MOLSYS_AI_DOCS_INDEX") or (REPO_ROOT / "server/docs_chat/data/rag_index.pkl")).resolve()

    sys.path.insert(0, str(REPO_ROOT / "server"))
    sys.path.insert(0, str(REPO_ROOT / "client"))

    from rag.retriever import load_index, retrieve  # type: ignore
    from model_server.server import Message, get_model_backend  # type: ignore

    if not docs_dir.exists():
        raise SystemExit(f"Docs corpus directory not found: {docs_dir}")
    if not index_path.exists():
        raise SystemExit(f"RAG index not found: {index_path}")

    index = load_index(index_path)
    docs = retrieve(query, index, k=5)

    context_lines: list[str] = []
    for i, doc in enumerate(docs, start=1):
        source = str(doc.metadata.get("path") or "unknown")
        try:
            source = str(Path(source).resolve().relative_to(docs_dir))
        except Exception:
            pass
        context_lines.append(f"[{i}] Source: {source}\n{doc.content}\n")

    context_block = "\n".join(context_lines) if context_lines else "(no documentation snippets found)"

    system_prompt = (
        "You are the MolSys-AI documentation assistant. "
        "Answer user questions using only the provided documentation excerpts. "
        "Cite sources by including bracketed numbers like [1] or [2] in your answer. "
        "If the answer cannot be inferred from the excerpts, say so explicitly."
    )

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=f"Documentation excerpts:\n\n{context_block}\n\nQuestion: {query}"),
    ]

    backend = get_model_backend()
    answer = backend.chat(messages)

    payload = {
        "query": query,
        "sources": [line.splitlines()[0] for line in context_lines],
        "answer": answer,
    }
    print(json.dumps(payload, indent=2))

    if not re.search(r"\[[0-9]+\]", answer):
        raise SystemExit("No [n] citation found in answer.")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

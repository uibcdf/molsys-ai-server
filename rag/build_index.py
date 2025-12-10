
"""Index building utilities for MolSys-AI RAG."""

from __future__ import annotations

import pickle
from pathlib import Path

from .embeddings import get_default_embedding_model
from .retriever import Document


def build_index(source_dir: Path, index_path: Path) -> None:
    """Build or rebuild the RAG index from the given source directory."""
    print(f"Building index from {source_dir}...")
    docs: list[Document] = []
    chunks: list[str] = []

    if not source_dir.exists():
        print(f"Source directory {source_dir} not found.")
        return

    # Find and parse Markdown files
    for path in sorted(source_dir.rglob("*.md")):
        if ".ipynb_checkpoints" in str(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
            # Simple chunking by paragraph
            for paragraph in text.split("\n\n"):
                if paragraph.strip():
                    docs.append(Document(content=paragraph.strip(), metadata={"path": str(path)}))
                    chunks.append(paragraph.strip())
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
        doc.embedding = embedding

    # Save the index to disk
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("wb") as f:
        pickle.dump(docs, f)

    print(f"Index built and saved to {index_path}")


"""Backend for the documentation chatbot (MVP skeleton).

This FastAPI app will eventually:
- receive user questions from the JS widget,
- call RAG + model server,
- return answers tailored to MolSys* documentation.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from agent.model_client import HTTPModelClient
from rag.build_index import build_index
from rag.retriever import Document, retrieve


app = FastAPI(title="MolSys-AI Docs Chat Backend (MVP)")


_DEFAULT_DOCS_DIR = Path(__file__).resolve().parent / "data" / "docs"
DOCS_SOURCE_DIR = Path(os.environ.get("MOLSYS_AI_DOCS_DIR", str(_DEFAULT_DOCS_DIR)))
DOCS_INDEX_PATH = Path(
    os.environ.get(
        "MOLSYS_AI_DOCS_INDEX",
        str(Path(__file__).resolve().parent / "data" / "index.faiss"),
    )
)

_DEFAULT_MODEL_SERVER_URL = "http://127.0.0.1:8000"
MODEL_SERVER_URL = os.environ.get("MOLSYS_AI_MODEL_SERVER_URL", _DEFAULT_MODEL_SERVER_URL)


class DocsChatRequest(BaseModel):
    """Incoming request schema for the docs chatbot."""

    query: str
    k: int = 5


class DocsChatResponse(BaseModel):
    """Response schema for the docs chatbot."""

    answer: str


@app.post("/v1/docs-chat", response_model=DocsChatResponse)
async def docs_chat(req: DocsChatRequest) -> DocsChatResponse:
    """Docs-chat endpoint backed by RAG + model server.

    Current behaviour:
    - retrieves up to ``k`` documents from the in-memory RAG index,
    - builds a prompt that includes those excerpts and the user query,
    - calls the configured model server via :class:`HTTPModelClient`,
    - returns the model's reply.

    If no model server is running, an error is returned.
    """
    docs = retrieve(req.query, k=req.k)

    # Build a simple context string from retrieved documents.
    context_lines: list[str] = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("path", "unknown")
        context_lines.append(f"[{i}] Source: {source}\n{doc.content}\n")

    context_block = "\n".join(context_lines) if context_lines else "(no documentation snippets found)"

    messages = [
        {
            "role": "system",
            "content": (
                "You are the MolSys-AI documentation assistant. "
                "Answer user questions using only the provided documentation excerpts. "
                "If the answer cannot be inferred from the excerpts, say so explicitly."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Documentation excerpts:\n\n{context_block}\n\n"
                f"Question: {req.query}"
            ),
        },
    ]

    client = HTTPModelClient(base_url=MODEL_SERVER_URL)
    answer = client.generate(messages)

    return DocsChatResponse(answer=answer)


@app.on_event("startup")
async def _build_docs_index_on_startup() -> None:
    """Build the in-memory RAG index when the docs-chat backend starts.

    The default source directory is ``docs_chat/data/docs`` relative to this
    file. You can override it via the ``MOLSYS_AI_DOCS_DIR`` environment
    variable. The ``DOCS_INDEX_PATH`` is currently unused and reserved for
    future FAISS-based persistence.
    """
    build_index(DOCS_SOURCE_DIR, DOCS_INDEX_PATH)

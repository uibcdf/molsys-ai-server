"""Backend for the documentation chatbot (MVP).

This FastAPI app:
- receives user messages from the JS widget,
- performs RAG over local Markdown docs,
- calls the model server (`/v1/chat`) to generate an answer.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.model_client import HTTPModelClient
from rag.build_index import build_index
from rag.retriever import Document, load_index, retrieve

app = FastAPI(title="MolSys-AI Docs Chat Backend (MVP)")

_DEFAULT_DOCS_DIR = Path(__file__).resolve().parent / "data" / "docs"
DOCS_SOURCE_DIR = Path(os.environ.get("MOLSYS_AI_DOCS_DIR", str(_DEFAULT_DOCS_DIR)))

DOCS_INDEX_PATH = Path(
    os.environ.get(
        "MOLSYS_AI_DOCS_INDEX",
        str(Path(__file__).resolve().parent / "data" / "rag_index.pkl"),
    )
)

_DEFAULT_MODEL_SERVER_URL = "http://127.0.0.1:8001"
MODEL_SERVER_URL = os.environ.get("MOLSYS_AI_MODEL_SERVER_URL", _DEFAULT_MODEL_SERVER_URL)

_DOCS_INDEX: list[Document] = []


class Message(BaseModel):
    role: str
    content: str


class DocsChatRequest(BaseModel):
    """Incoming request schema for the docs chatbot.

    Use `messages` for a real multi-turn chatbot. `query` is kept for
    backwards-compatible single-turn calls.
    """

    query: str | None = None
    messages: list[Message] | None = None
    k: int = 5


class DocsChatResponse(BaseModel):
    answer: str


@app.post("/v1/docs-chat", response_model=DocsChatResponse)
async def docs_chat(req: DocsChatRequest) -> DocsChatResponse:
    if req.messages:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        query = next((m["content"] for m in reversed(messages) if m["role"] == "user"), None)
        if not query:
            raise HTTPException(status_code=400, detail="No user message found in `messages`.")
    elif req.query:
        query = req.query
        messages = [{"role": "user", "content": query}]
    else:
        raise HTTPException(status_code=400, detail="Provide either `query` or `messages`.")

    docs = retrieve(query, _DOCS_INDEX, k=req.k)

    context_lines: list[str] = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("path", "unknown")
        context_lines.append(f"[{i}] Source: {source}\n{doc.content}\n")
    context_block = "\n".join(context_lines) if context_lines else "(no documentation snippets found)"

    system_msg = {
        "role": "system",
        "content": (
            "You are the MolSys-AI documentation assistant. "
            "Answer user questions using only the provided documentation excerpts. "
            "If the answer cannot be inferred from the excerpts, say so explicitly."
        ),
    }
    if not any(m.get("role") == "system" for m in messages):
        messages.insert(0, system_msg)

    # Inject RAG context only for the current turn (avoid growing the history).
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            original = messages[i].get("content", "")
            messages[i]["content"] = (
                f"Documentation excerpts:\n\n{context_block}\n\n" f"Question: {original}"
            )
            break

    client = HTTPModelClient(base_url=MODEL_SERVER_URL)
    answer = client.generate(messages)
    return DocsChatResponse(answer=answer)


@app.on_event("startup")
async def _load_or_build_docs_index_on_startup() -> None:
    global _DOCS_INDEX
    if DOCS_INDEX_PATH.exists():
        print(f"Loading existing index from {DOCS_INDEX_PATH}")
        _DOCS_INDEX = load_index(DOCS_INDEX_PATH)
    else:
        print(f"Index not found. Building index from {DOCS_SOURCE_DIR}...")
        build_index(DOCS_SOURCE_DIR, DOCS_INDEX_PATH)
        _DOCS_INDEX = load_index(DOCS_INDEX_PATH)
    print(f"Index loaded with {_DOCS_INDEX.__len__()} documents.")


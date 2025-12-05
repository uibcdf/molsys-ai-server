
"""FastAPI-based model server for MolSys-AI (MVP skeleton)."""

from typing import List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="MolSys-AI Model Server (MVP)")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class ChatResponse(BaseModel):
    content: str


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Placeholder chat endpoint.

    For now, this just echoes the last user message. Later it will call
    llama.cpp (or another backend) using the configured model.
    """
    last_user = next((m for m in reversed(req.messages) if m.role == "user"), None)
    content = last_user.content if last_user else "(no user message provided)"
    return ChatResponse(content=f"[MolSys-AI stub reply] You said: {content}")

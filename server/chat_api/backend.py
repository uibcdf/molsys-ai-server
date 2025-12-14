"""Chat API (RAG orchestrator) for MolSys-AI.

This FastAPI app:
- receives user messages from the JS widget,
- optionally performs RAG over local documentation snapshots,
 - calls the model engine server (`/v1/engine/chat`) to generate an answer,
- for CLI calls, can run an LLM router to decide whether to use RAG and show sources.
"""

from __future__ import annotations

import os
import time
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.model_client import HTTPModelClient
from rag.build_index import build_index
from rag.retriever import Document, load_index, retrieve

app = FastAPI(title="MolSys-AI Chat API (RAG Orchestrator, MVP)")

_cors_origins_raw = os.environ.get("MOLSYS_AI_CORS_ORIGINS", "").strip()
if _cors_origins_raw:
    origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

_DEFAULT_DOCS_DIR = Path(__file__).resolve().parent / "data" / "docs"
DOCS_SOURCE_DIR = Path(os.environ.get("MOLSYS_AI_DOCS_DIR", str(_DEFAULT_DOCS_DIR))).expanduser()

DOCS_INDEX_PATH = Path(
    os.environ.get(
        "MOLSYS_AI_DOCS_INDEX",
        str(Path(__file__).resolve().parent / "data" / "rag_index.pkl"),
    )
).expanduser()

_DEFAULT_ENGINE_URL = "http://127.0.0.1:8001"
ENGINE_URL = os.environ.get("MOLSYS_AI_ENGINE_URL", _DEFAULT_ENGINE_URL)
ENGINE_API_KEY = (os.environ.get("MOLSYS_AI_ENGINE_API_KEY") or "").strip() or None

_DEFAULT_ANCHORS_PATH = Path(__file__).resolve().parent / "data" / "anchors.json"
DOCS_ANCHORS_PATH = Path(os.environ.get("MOLSYS_AI_DOCS_ANCHORS", str(_DEFAULT_ANCHORS_PATH))).expanduser()

_CHAT_KEYS_ENV_VAR = "MOLSYS_AI_CHAT_API_KEYS"

_DOCS_INDEX: list[Document] = []
_RATE_STATE: dict[tuple[str, int], int] = {}


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    """Incoming request schema for the chat API.

    Use `messages` for a real multi-turn chatbot. `query` is kept for
    backwards-compatible single-turn calls.
    """

    query: str | None = None
    messages: list[Message] | None = None
    k: int = 5
    client: str | None = None  # "widget" | "cli"
    rag: str | None = None  # "on" | "off" | "auto"
    sources: str | None = None  # "on" | "off" | "auto"


class Source(BaseModel):
    """One retrieved snippet source, aligned with bracketed citations like [1]."""

    id: int
    path: str
    section: str | None = None
    label: str | None = None
    url: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = []


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


@lru_cache()
def get_chat_api_keys() -> set[str]:
    raw = (os.environ.get(_CHAT_KEYS_ENV_VAR) or "").strip()
    return {k.strip() for k in raw.split(",") if k.strip()}


def _extract_api_key(request: Request) -> str | None:
    auth = (request.headers.get("authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        return token or None
    x_api_key = (request.headers.get("x-api-key") or "").strip()
    return x_api_key or None


@lru_cache()
def _load_anchors() -> dict[str, Any] | None:
    if not DOCS_ANCHORS_PATH.exists():
        return None
    try:
        return json.loads(DOCS_ANCHORS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def require_chat_api_key(request: Request) -> None:
    allowed_keys = get_chat_api_keys()
    if not allowed_keys:
        return
    key = _extract_api_key(request)
    if not key or key not in allowed_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def _client_ip(request: Request) -> str:
    # Behind a reverse proxy, prefer the first X-Forwarded-For hop.
    xff = (request.headers.get("x-forwarded-for") or "").strip()
    if xff:
        return xff.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def rate_limit_chat(request: Request) -> None:
    """Very small in-process rate limiter (optional).

    Enable by setting `MOLSYS_AI_CHAT_RATE_LIMIT_PER_MIN` to a positive integer.
    This is not a replacement for reverse-proxy rate limiting, but it adds a minimal
    safety net for public demos.
    """

    limit_per_min = _env_int("MOLSYS_AI_CHAT_RATE_LIMIT_PER_MIN", 0)
    if limit_per_min <= 0:
        return
    ip = _client_ip(request)
    window = int(time.time() // 60)
    key = (ip, window)
    _RATE_STATE[key] = _RATE_STATE.get(key, 0) + 1
    if _RATE_STATE[key] > limit_per_min:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please retry later.")


def _format_source(doc: Document) -> str:
    path = str(doc.metadata.get("path") or "unknown")
    section = str(doc.metadata.get("section") or "").strip()
    label = str(doc.metadata.get("label") or "").strip()
    try:
        rel = Path(path).resolve().relative_to(DOCS_SOURCE_DIR.resolve())
        path_str = str(rel)
    except Exception:
        path_str = path
    if label:
        path_str = f"{path_str}#{label}"
    if section:
        return f"{path_str} ({section})"
    return path_str


def _resolve_docs_url(doc: Document) -> str | None:
    anchors = _load_anchors()
    if not anchors:
        return None

    raw_path = str(doc.metadata.get("path") or "").strip()
    if not raw_path:
        return None

    try:
        rel = Path(raw_path).resolve().relative_to(DOCS_SOURCE_DIR.resolve())
    except Exception:
        return None

    parts = rel.parts
    if len(parts) < 2:
        return None

    project = parts[0]
    rel_in_project = Path(*parts[1:])
    try:
        docs_idx = rel_in_project.parts.index("docs")
    except ValueError:
        return None

    rel_in_docs = Path(*rel_in_project.parts[docs_idx + 1 :])
    if not rel_in_docs.parts:
        return None

    projects = anchors.get("projects")
    if not isinstance(projects, dict):
        return None
    project_info = projects.get(project)
    if not isinstance(project_info, dict) or not project_info.get("ok"):
        return None

    label = str(doc.metadata.get("label") or "").strip() or None
    if label:
        label_index = project_info.get("label_index")
        if isinstance(label_index, dict):
            entry = label_index.get(label)
            if isinstance(entry, dict):
                url = entry.get("docs_url")
                if isinstance(url, str) and url.strip():
                    return url.strip()

    page_key = rel_in_docs.as_posix()
    pages = project_info.get("pages")
    if isinstance(pages, dict):
        entry = pages.get(page_key)
        if isinstance(entry, dict):
            url = entry.get("docs_url")
            if isinstance(url, str) and url.strip():
                url = url.strip()
                if label:
                    return f"{url}#{label}"
                return url

    base = project_info.get("docs_base_url")
    if not isinstance(base, str) or not base.strip():
        return None
    base = base.strip().rstrip("/")
    html_rel = rel_in_docs.with_suffix("").as_posix() + ".html"
    url = f"{base}/{html_rel}"
    if label:
        url = f"{url}#{label}"
    return url


def _source_record(doc: Document, source_id: int) -> Source:
    path = str(doc.metadata.get("path") or "unknown")
    section = str(doc.metadata.get("section") or "").strip() or None
    label = str(doc.metadata.get("label") or "").strip() or None

    try:
        rel = Path(path).resolve().relative_to(DOCS_SOURCE_DIR.resolve())
        path_str = str(rel)
    except Exception:
        path_str = path

    url = _resolve_docs_url(doc)
    return Source(id=int(source_id), path=path_str, section=section, label=label, url=url)


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    s = (text or "").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                block = s[start : i + 1]
                try:
                    obj = json.loads(block)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


def _coerce_client(value: str | None) -> str:
    v = (value or "").strip().lower()
    if v in {"widget", "cli"}:
        return v
    return "widget"


def _coerce_mode(value: str | None, default: str) -> str:
    v = (value or "").strip().lower()
    if v in {"on", "off", "auto"}:
        return v
    return default


def _build_router_prompt(user_text: str) -> list[dict[str, str]]:
    system = (
        "You are a router for MolSys-AI.\n"
        "Decide whether to use documentation retrieval (RAG) and whether to show sources.\n"
        "Return ONLY valid JSON with exactly these keys:\n"
        '  - "use_rag": boolean\n'
        '  - "show_sources": boolean\n'
        "Guidelines:\n"
        "- use_rag=true when the question is about MolSysSuite tools, their APIs, installation, workflows, or documentation.\n"
        "- use_rag=false for unrelated small talk or purely general questions.\n"
        "- show_sources=true when the user explicitly asks for citations/links/sources, or when the answer should be grounded in documentation.\n"
        "- show_sources=false otherwise.\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"User message:\n{user_text}\n"},
    ]


def _default_system_prompt() -> str:
    return (
        "You are MolSys-AI, a specialist assistant for the UIBCDF MolSysSuite ecosystem.\n"
        "Prefer precise, actionable answers. If you are not sure, ask a clarifying question.\n"
        "Do not invent APIs, flags, or behaviors.\n"
    )


def _user_explicitly_wants_sources(text: str) -> bool:
    s = (text or "").lower()
    return bool(re.search(r"\b(cite|citation|citations|source|sources|link|links|reference|references)\b", s))


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "docs_index_chunks": len(_DOCS_INDEX)}


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    _auth: None = Depends(require_chat_api_key),
    _rl: None = Depends(rate_limit_chat),
) -> ChatResponse:
    allowed_roles = {"system", "user", "assistant"}
    max_k = max(_env_int("MOLSYS_AI_CHAT_MAX_K", 8), 1)
    max_messages = max(_env_int("MOLSYS_AI_CHAT_MAX_MESSAGES", 30), 1)
    max_message_chars = max(_env_int("MOLSYS_AI_CHAT_MAX_MESSAGE_CHARS", 4000), 200)
    max_total_chars = max(_env_int("MOLSYS_AI_CHAT_MAX_TOTAL_CHARS", 20000), 1000)

    req_k = int(req.k)
    if req_k <= 0:
        req_k = 1
    if req_k > max_k:
        req_k = max_k

    if req.messages:
        if len(req.messages) > max_messages:
            raise HTTPException(status_code=400, detail=f"Too many messages (max {max_messages}).")
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        for m in messages:
            if m.get("role") not in allowed_roles:
                raise HTTPException(status_code=400, detail=f"Invalid role: {m.get('role')!r}.")
            if not isinstance(m.get("content"), str) or not m["content"].strip():
                raise HTTPException(status_code=400, detail="Empty message content is not allowed.")
            if len(m["content"]) > max_message_chars:
                raise HTTPException(status_code=400, detail=f"Message too large (max {max_message_chars} chars).")

        total = sum(len(m["content"]) for m in messages)
        if total > max_total_chars:
            raise HTTPException(status_code=400, detail=f"Request too large (max {max_total_chars} chars).")

        query = next((m["content"] for m in reversed(messages) if m["role"] == "user"), None)
        if not query:
            raise HTTPException(status_code=400, detail="No user message found in `messages`.")
    elif req.query:
        query = req.query
        messages = [{"role": "user", "content": query}]
    else:
        raise HTTPException(status_code=400, detail="Provide either `query` or `messages`.")

    client_kind = _coerce_client(req.client)
    rag_mode = _coerce_mode(req.rag, "on" if client_kind == "widget" else "auto")
    sources_mode = _coerce_mode(req.sources, "on" if client_kind == "widget" else "auto")

    use_rag = rag_mode == "on"
    show_sources = sources_mode == "on"

    # CLI defaults: specialist-first (RAG likely on), sources only when requested or inferred.
    if client_kind == "cli":
        if rag_mode == "auto":
            use_rag = True
        if sources_mode == "auto":
            show_sources = False

        if rag_mode == "auto" or sources_mode == "auto":
            router_client = HTTPModelClient(base_url=ENGINE_URL, api_key=ENGINE_API_KEY)
            router_text = router_client.generate(
                _build_router_prompt(query),
                generation={"temperature": 0.0, "top_p": 1.0, "max_tokens": 128},
            )
            decision = _extract_first_json_object(router_text) or {}
            if rag_mode == "auto" and isinstance(decision.get("use_rag"), bool):
                use_rag = bool(decision["use_rag"])
            if sources_mode == "auto" and isinstance(decision.get("show_sources"), bool):
                show_sources = bool(decision["show_sources"])

            # Safety net if the router fails badly.
            if not decision:
                if sources_mode == "auto":
                    show_sources = _user_explicitly_wants_sources(query)

    if show_sources and not use_rag:
        use_rag = True

    docs = retrieve(query, _DOCS_INDEX, k=req_k) if use_rag else []

    context_lines: list[str] = []
    sources: list[Source] = []
    for i, doc in enumerate(docs, start=1):
        src = _source_record(doc, i)
        if show_sources:
            sources.append(src)

        if show_sources:
            source_for_prompt = _format_source(doc)
            if src.url:
                source_for_prompt = f"{source_for_prompt} | {src.url}"
            context_lines.append(f"[{i}] Source: {source_for_prompt}\n{doc.content}\n")
        else:
            context_lines.append(doc.content)

    context_block: str
    if use_rag:
        if show_sources:
            context_block = "\n".join(context_lines) if context_lines else "(no documentation snippets found)"
        else:
            context_block = "\n\n---\n\n".join(context_lines) if context_lines else "(no documentation snippets found)"
    else:
        context_block = ""

    system_prompt = os.environ.get("MOLSYS_AI_CHAT_SYSTEM_PROMPT") or _default_system_prompt()
    if use_rag and client_kind == "widget":
        system_prompt = (
            system_prompt.strip()
            + "\nAnswer user questions using ONLY the provided documentation excerpts.\n"
            + "Cite sources by including bracketed numbers like [1] or [2] in your answer.\n"
            + "If the answer cannot be inferred from the excerpts, say so explicitly.\n"
        )
    elif use_rag and show_sources:
        system_prompt = (
            system_prompt.strip()
            + "\nUse the provided documentation excerpts to answer.\n"
            + "Cite sources by including bracketed numbers like [1] or [2] in your answer.\n"
            + "If the answer cannot be inferred from the excerpts, say so explicitly.\n"
        )
    elif use_rag and not show_sources:
        system_prompt = (
            system_prompt.strip()
            + "\nUse the provided documentation excerpts to answer.\n"
            + "Do NOT include bracketed citations like [1]. Do NOT mention sources.\n"
            + "If the answer cannot be inferred from the excerpts, say so explicitly.\n"
        )
    system_msg = {
        "role": "system",
        "content": system_prompt,
    }
    if not any(m.get("role") == "system" for m in messages):
        messages.insert(0, system_msg)

    if use_rag:
        # Inject RAG context only for the current turn (avoid growing the history).
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                original = messages[i].get("content", "")
                prefix = "Documentation excerpts:\n\n" if show_sources else "Documentation excerpts (do not cite):\n\n"
                messages[i]["content"] = f"{prefix}{context_block}\n\nQuestion: {original}"
                break

    client = HTTPModelClient(base_url=ENGINE_URL, api_key=ENGINE_API_KEY)
    answer = client.generate(messages)
    return ChatResponse(answer=answer, sources=sources if show_sources else [])


@app.on_event("startup")
async def _load_or_build_docs_index_on_startup() -> None:
    global _DOCS_INDEX
    force_rebuild = _env_bool("MOLSYS_AI_DOCS_INDEX_REBUILD", False)
    if DOCS_INDEX_PATH.exists() and not force_rebuild:
        print(f"Loading existing index from {DOCS_INDEX_PATH}")
        _DOCS_INDEX = load_index(DOCS_INDEX_PATH)
    else:
        print(f"Building index from {DOCS_SOURCE_DIR}...")
        build_index(DOCS_SOURCE_DIR, DOCS_INDEX_PATH)
        _DOCS_INDEX = load_index(DOCS_INDEX_PATH)
    print(f"Index loaded with {_DOCS_INDEX.__len__()} documents.")

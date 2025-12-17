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
from rag.retriever import Document, ScoredDocument, load_index, retrieve, retrieve_scored

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

PROJECT_INDEX_DIR = Path(
    os.environ.get(
        "MOLSYS_AI_PROJECT_INDEX_DIR",
        str(Path(__file__).resolve().parent / "data" / "indexes"),
    )
).expanduser()

_DEFAULT_ENGINE_URL = "http://127.0.0.1:8001"
ENGINE_URL = os.environ.get("MOLSYS_AI_ENGINE_URL", _DEFAULT_ENGINE_URL)
ENGINE_API_KEY = (os.environ.get("MOLSYS_AI_ENGINE_API_KEY") or "").strip() or None

_DEFAULT_ANCHORS_PATH = Path(__file__).resolve().parent / "data" / "anchors.json"
DOCS_ANCHORS_PATH = Path(os.environ.get("MOLSYS_AI_DOCS_ANCHORS", str(_DEFAULT_ANCHORS_PATH))).expanduser()

_DEFAULT_SYMBOLS_PATH = DOCS_SOURCE_DIR / "_symbols.json"
SYMBOLS_PATH = Path(os.environ.get("MOLSYS_AI_SYMBOLS_PATH", str(_DEFAULT_SYMBOLS_PATH))).expanduser()

_CHAT_KEYS_ENV_VAR = "MOLSYS_AI_CHAT_API_KEYS"

_DOCS_INDEX: list[Document] = []
_PROJECT_INDICES: dict[str, list[Document]] = {}
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


@lru_cache()
def _load_symbol_registry() -> dict[str, Any] | None:
    if not SYMBOLS_PATH.exists():
        return None
    try:
        return json.loads(SYMBOLS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


@lru_cache()
def _symbols_for_project(project: str) -> set[str]:
    reg = _load_symbol_registry()
    if not reg:
        return set()
    projects = reg.get("projects")
    if not isinstance(projects, dict):
        return set()
    entry = projects.get(project)
    if not isinstance(entry, dict):
        return set()
    symbols = entry.get("symbols")
    if not isinstance(symbols, list):
        return set()
    return {str(s) for s in symbols if isinstance(s, str) and s}


def _extract_import_aliases(text: str) -> dict[str, str]:
    """Return a mapping of import aliases to canonical MolSysSuite project names."""
    aliases: dict[str, str] = {}
    for project in ("molsysmt", "molsysviewer", "pyunitwizard", "topomt"):
        m = re.findall(
            rf"^\s*import\s+{project}\s+as\s+([A-Za-z_][A-Za-z0-9_]*)\s*$",
            text,
            flags=re.M,
        )
        for a in m:
            aliases[a] = project
        m2 = re.findall(
            rf"^\s*from\s+{project}\s+import\s+.+?\s+as\s+([A-Za-z_][A-Za-z0-9_]*)\s*$",
            text,
            flags=re.M,
        )
        for a in m2:
            aliases[a] = project
    return aliases


def _extract_candidate_symbols(text: str) -> set[str]:
    """Extract dotted-path API symbol candidates from model output."""
    s = text or ""
    candidates: set[str] = set()
    for t in re.findall(r"`([^`]+)`", s):
        for sym in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b", t):
            candidates.add(sym)
    for sym in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b", s):
        candidates.add(sym)
    return candidates


def _resolve_alias_symbol(sym: str, aliases: dict[str, str]) -> str:
    head, *rest = sym.split(".")
    if head in aliases:
        return ".".join([aliases[head], *rest])
    return sym


def _unknown_tool_symbols(answer: str) -> list[str]:
    """Return a list of unknown MolSysSuite API symbols mentioned in `answer`."""
    aliases = _extract_import_aliases(answer)
    candidates = _extract_candidate_symbols(answer)
    unknown: set[str] = set()

    for sym in candidates:
        resolved = _resolve_alias_symbol(sym, aliases)
        project = resolved.split(".", 1)[0]
        if project not in {"molsysmt", "molsysviewer", "pyunitwizard", "topomt"}:
            continue
        symbols, prefixes = _symbol_sets_for_project(project)
        if not symbols and not prefixes:
            continue
        if resolved in symbols or resolved in prefixes:
            continue
        unknown.add(resolved)

    return sorted(unknown)


@lru_cache()
def _symbol_sets_for_project(project: str) -> tuple[set[str], set[str]]:
    symbols = _symbols_for_project(project)
    prefixes: set[str] = set()
    for s in symbols:
        parts = s.split(".")
        # Add dotted prefixes so mentions like `molsysmt.structure` validate even if the
        # registry contains only deeper symbols (e.g. `molsysmt.structure.get_rmsd`).
        for i in range(1, len(parts)):
            prefixes.add(".".join(parts[:i]))
    return symbols, prefixes


def _is_api_surface_doc(doc: Document) -> bool:
    raw = str(doc.metadata.get("path") or "")
    if not raw:
        return False
    p = raw.replace("\\", "/")
    return "/api_surface/" in p


def _is_symbol_card_doc(doc: Document) -> bool:
    kind = str(doc.metadata.get("kind") or "").strip().lower()
    if kind:
        return kind == "symbol_card"
    raw = str(doc.metadata.get("path") or "")
    p = raw.replace("\\", "/")
    return "/symbol_cards/" in p


def _is_recipe_doc(doc: Document) -> bool:
    kind = str(doc.metadata.get("kind") or "").strip().lower()
    if kind:
        return kind in {"recipe", "recipe_section", "tutorial_recipe", "recipe_card", "tutorial_card"}
    raw = str(doc.metadata.get("path") or "")
    p = raw.replace("\\", "/")
    return "/recipes/" in p


def _is_tutorial_recipe_doc(doc: Document) -> bool:
    kind = str(doc.metadata.get("kind") or "").strip().lower()
    if kind:
        return kind == "tutorial_recipe"
    raw = str(doc.metadata.get("path") or "")
    p = raw.replace("\\", "/")
    return "/recipes/notebooks_tutorials/" in p


def _is_recipe_section_doc(doc: Document) -> bool:
    kind = str(doc.metadata.get("kind") or "").strip().lower()
    if kind:
        return kind == "recipe_section"
    raw = str(doc.metadata.get("path") or "")
    p = raw.replace("\\", "/")
    return "/recipes/notebooks_sections/" in p


def _is_recipe_card_doc(doc: Document) -> bool:
    kind = str(doc.metadata.get("kind") or "").strip().lower()
    if kind:
        return kind == "recipe_card"
    raw = str(doc.metadata.get("path") or "")
    p = raw.replace("\\", "/")
    return "/recipe_cards/" in p


def _is_tutorial_card_doc(doc: Document) -> bool:
    kind = str(doc.metadata.get("kind") or "").strip().lower()
    if kind:
        return kind == "tutorial_card"
    raw = str(doc.metadata.get("path") or "")
    p = raw.replace("\\", "/")
    return "/recipe_cards/notebooks_tutorials/" in p


def _looks_like_api_question(query: str) -> bool:
    s = (query or "").lower()
    if any(p in s for p in ("molsysmt", "molsysviewer", "pyunitwizard", "topomt")):
        return True
    if re.search(r"\b(function|method|class|signature|argument|parameter|keyword|kwarg|import)\b", s):
        return True
    # Identifier-ish query hints.
    if re.search(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\\.[A-Za-z_][A-Za-z0-9_]*)+\\b", s):
        return True
    if "```" in s or "()" in s:
        return True
    return False


def _dedupe_docs(docs: list[Document]) -> list[Document]:
    seen: set[tuple[str, str, str, str]] = set()
    out: list[Document] = []
    for d in docs:
        key = (
            str(d.metadata.get("path") or ""),
            str(d.metadata.get("section") or ""),
            str(d.metadata.get("label") or ""),
            (d.content or "")[:200],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _dedupe_scored_docs(docs: list[ScoredDocument]) -> list[ScoredDocument]:
    best: dict[tuple[str, str, str, str], ScoredDocument] = {}
    for sd in docs:
        d = sd.doc
        key = (
            str(d.metadata.get("path") or ""),
            str(d.metadata.get("section") or ""),
            str(d.metadata.get("label") or ""),
            (d.content or "")[:200],
        )
        prev = best.get(key)
        if prev is None or sd.score > prev.score:
            best[key] = sd
    # Keep stable ordering by score, then by path.
    return sorted(best.values(), key=lambda sd: (sd.score, str(sd.doc.metadata.get("path") or "")), reverse=True)


_NOTEBOOK_RECIPE_PREFIXES = (
    "/recipes/notebooks_tutorials/",
    "/recipes/notebooks_sections/",
    "/recipes/notebooks/",
    "/recipe_cards/notebooks_tutorials/",
    "/recipe_cards/notebooks_sections/",
    "/recipe_cards/notebooks/",
)


def _notebook_key(doc: Document) -> str | None:
    """Return a stable notebook key for notebook-derived recipe docs, else None."""
    rel = str(doc.metadata.get("relpath") or "").replace("\\", "/")
    if not rel:
        return None
    for pref in _NOTEBOOK_RECIPE_PREFIXES:
        if pref not in f"/{rel}":
            continue
        # rel: <project>/<...pref...>/<notebook_rel_without_suffix>/<leaf>.md
        # Example:
        #   molsysmt/recipes/notebooks_sections/docs/content/user/tools/basic/select/section_0001.md
        parts = rel.split(pref.strip("/"), 1)
        if len(parts) != 2:
            continue
        tail = parts[1].lstrip("/")
        if not tail:
            continue
        segs = tail.split("/")
        if len(segs) < 2:
            continue
        notebook_rel = "/".join(segs[:-1])
        if notebook_rel:
            return f"{doc.metadata.get('project') or ''}:{notebook_rel}"
    return None


def _is_notebook_recipe(doc: Document) -> bool:
    rel = str(doc.metadata.get("relpath") or "").replace("\\", "/")
    if not rel:
        return False
    rel2 = f"/{rel}"
    return any(p in rel2 for p in _NOTEBOOK_RECIPE_PREFIXES)


def _safe_read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None


def _doc_from_disk(path: Path, *, kind: str) -> Document | None:
    text = _safe_read_text(path)
    if not text or not text.strip():
        return None
    meta: dict[str, Any] = {"path": str(path), "kind": kind}
    try:
        rel = path.resolve().relative_to(DOCS_SOURCE_DIR.resolve())
        meta["relpath"] = rel.as_posix()
        if rel.parts:
            meta["project"] = rel.parts[0]
    except Exception:
        pass
    return Document(content=text.strip(), metadata=meta)


def _tutorial_path_for_notebook_recipe(doc: Document) -> tuple[Path, str] | None:
    rel = str(doc.metadata.get("relpath") or "").replace("\\", "/")
    if not rel:
        return None

    # Map sections/cells → tutorial within the same notebook directory.
    mapping = [
        ("/recipes/notebooks_sections/", "/recipes/notebooks_tutorials/", "tutorial_recipe"),
        ("/recipes/notebooks/", "/recipes/notebooks_tutorials/", "tutorial_recipe"),
        ("/recipe_cards/notebooks_sections/", "/recipe_cards/notebooks_tutorials/", "tutorial_card"),
        ("/recipe_cards/notebooks/", "/recipe_cards/notebooks_tutorials/", "tutorial_card"),
    ]
    rel2 = f"/{rel}"
    for src_pref, dst_pref, kind in mapping:
        if src_pref not in rel2:
            continue
        # rel: <project><src_pref><notebook_rel>/<leaf>.md
        before, after = rel.split(src_pref.strip("/"), 1)
        tail = after.lstrip("/")
        parts = tail.split("/")
        if len(parts) < 2:
            continue
        notebook_rel = "/".join(parts[:-1])
        out_rel = (before.strip("/") + dst_pref + notebook_rel + "/tutorial.md").lstrip("/")
        return (DOCS_SOURCE_DIR / out_rel, kind)
    return None


def _augment_with_notebook_tutorials(docs: list[Document], *, k: int) -> list[Document]:
    """Ensure notebook-derived answers include the matching tutorial recipe/card when available."""
    if not docs:
        return docs
    have = {str(d.metadata.get("path") or "") for d in docs}
    inserts: list[tuple[int, Document]] = []
    for idx, d in enumerate(docs):
        if not _is_notebook_recipe(d):
            continue
        rec = _tutorial_path_for_notebook_recipe(d)
        if not rec:
            continue
        tpath, kind = rec
        if str(tpath) in have:
            continue
        tdoc = _doc_from_disk(tpath, kind=kind)
        if not tdoc:
            continue
        # Insert tutorials early, right after any leading symbol cards.
        inserts.append((idx, tdoc))
        have.add(str(tpath))

    if not inserts:
        return docs

    # Compute insertion point: after the last symbol card in the current list.
    insert_at = 0
    for i, d in enumerate(docs):
        if str(d.metadata.get("kind") or "").strip().lower() == "symbol_card":
            insert_at = i + 1
        else:
            break

    for _, tdoc in inserts:
        docs.insert(insert_at, tdoc)
        insert_at += 1

    # Keep within k by trimming from the end (least preferred).
    return _dedupe_docs(docs)[:k]


def _log_scored_docs(query: str, scored: list[ScoredDocument], *, label: str) -> None:
    if not _env_bool("MOLSYS_AI_RAG_LOG_SCORES", False):
        return
    top = int((os.environ.get("MOLSYS_AI_RAG_LOG_SCORES_TOP") or "8").strip() or "8")
    top = max(top, 1)
    print(f"[rag] scores ({label}) query={query!r}")
    for i, sd in enumerate(scored[:top], start=1):
        d = sd.doc
        kind = str(d.metadata.get("kind") or "")
        rel = str(d.metadata.get("relpath") or d.metadata.get("path") or "")
        score = f"{sd.score:.4f}"
        sim = f"{sd.sim:.4f}"
        lex = f"{sd.lex_norm:.4f}"
        bm = f"{sd.bm25_norm:.4f}"
        print(f"[rag]  {i:02d} score={score} sim={sim} lex={lex} bm25={bm} kind={kind} rel={rel}")


def _pick_notebook_cluster(scored: list[ScoredDocument], *, want: int) -> list[ScoredDocument]:
    """Pick a coherent tutorial/section/cell set from a single notebook (best-effort)."""
    if want <= 0:
        return []
    by_nb: dict[str, list[ScoredDocument]] = {}
    for sd in scored:
        key = _notebook_key(sd.doc)
        if not key:
            continue
        by_nb.setdefault(key, []).append(sd)
    if not by_nb:
        return []
    # Choose the notebook whose best doc has the highest score.
    best_nb = max(by_nb.items(), key=lambda kv: max(d.score for d in kv[1]))[0]
    docs = sorted(by_nb[best_nb], key=lambda sd: sd.score, reverse=True)

    def kind_of(d: Document) -> str:
        return str(d.metadata.get("kind") or "").strip().lower()

    tutorial = [sd for sd in docs if kind_of(sd.doc) in {"tutorial_card", "tutorial_recipe"}]
    sections = [sd for sd in docs if kind_of(sd.doc) in {"recipe_section"}]
    cells = [sd for sd in docs if kind_of(sd.doc) == "recipe" and "/recipes/notebooks/" in f"/{str(sd.doc.metadata.get('relpath') or '')}"]

    picked: list[ScoredDocument] = []
    if tutorial:
        picked.append(tutorial[0])
    if len(picked) < want and sections:
        picked.append(sections[0])
    if len(picked) < want and cells:
        picked.append(cells[0])
    if len(picked) < want:
        for sd in docs:
            if sd in picked:
                continue
            picked.append(sd)
            if len(picked) >= want:
                break
    return picked[:want]


def _select_docs_for_api_question(candidates: list[Document], *, k: int) -> list[Document]:
    """Select a balanced set of sources for API-heavy questions.

    Preference order:
      1) symbol cards (per-symbol docs)
      2) tutorial recipes (notebook overviews)
      3) recipe sections (multi-cell notebook blocks)
      4) recipes (tests/per-cell snippets/other)
      3) api_surface (per-module docs)
      4) narrative docs
    """

    if not candidates or k <= 0:
        return []

    symbol_cards = [d for d in candidates if _is_symbol_card_doc(d)]
    tutorial_cards = [d for d in candidates if _is_tutorial_card_doc(d)]
    recipe_cards = [d for d in candidates if _is_recipe_card_doc(d) and d not in tutorial_cards]
    tutorial_recipes = [d for d in candidates if _is_tutorial_recipe_doc(d) and d not in tutorial_cards]
    recipe_sections = [d for d in candidates if _is_recipe_section_doc(d)]
    recipes = [d for d in candidates if _is_recipe_doc(d) and d not in tutorial_cards and d not in recipe_cards and d not in tutorial_recipes and d not in recipe_sections]
    api_surface = [d for d in candidates if _is_api_surface_doc(d)]
    rest = [
        d
        for d in candidates
        if d not in symbol_cards
        and d not in tutorial_cards
        and d not in recipe_cards
        and d not in tutorial_recipes
        and d not in recipe_sections
        and d not in recipes
        and d not in api_surface
    ]

    # Quotas tuned for small k (default k=5).
    n_symbol = min(2, k)
    n_tutorial = min(1, max(k - n_symbol, 0))
    n_section = min(1, max(k - n_symbol - n_tutorial, 0))
    n_recipe_card = min(1, max(k - n_symbol - n_tutorial - n_section, 0))
    n_recipe = min(1, max(k - n_symbol - n_tutorial - n_section - n_recipe_card, 0))
    n_api = min(1, max(k - n_symbol - n_tutorial - n_section - n_recipe_card - n_recipe, 0))

    picked: list[Document] = []
    picked.extend(symbol_cards[:n_symbol])
    picked.extend(tutorial_cards[:n_tutorial])
    if len(picked) < (n_symbol + n_tutorial):
        picked.extend(tutorial_recipes[: max(0, n_symbol + n_tutorial - len(picked))])
    picked.extend(recipe_sections[:n_section])
    picked.extend(recipe_cards[:n_recipe_card])
    picked.extend(recipes[:n_recipe])
    picked.extend(api_surface[:n_api])
    picked.extend(rest)
    picked = _dedupe_docs(picked)
    return picked[:k]


def _select_scored_for_api_question(scored: list[ScoredDocument], *, k: int) -> list[Document]:
    """Score-aware selection for API-heavy questions, with notebook hierarchical preference."""
    if not scored or k <= 0:
        return []

    def kind(doc: Document) -> str:
        return str(doc.metadata.get("kind") or "").strip().lower()

    symbol_cards = [sd for sd in scored if kind(sd.doc) == "symbol_card"]
    api_surface = [sd for sd in scored if kind(sd.doc) == "api_surface"]
    recipe_cards = [sd for sd in scored if kind(sd.doc) == "recipe_card"]
    tutorials = [sd for sd in scored if kind(sd.doc) in {"tutorial_recipe", "tutorial_card"}]
    sections = [sd for sd in scored if kind(sd.doc) == "recipe_section"]
    recipes = [sd for sd in scored if kind(sd.doc) == "recipe"]
    rest = [sd for sd in scored if sd not in symbol_cards and sd not in api_surface and sd not in recipe_cards and sd not in tutorials and sd not in sections and sd not in recipes]

    # Pick top symbol cards first.
    picked: list[Document] = []
    picked.extend([sd.doc for sd in symbol_cards[: min(2, k)]])

    # Notebook cluster: prefer pulling a coherent tutorial+section when available.
    remaining_slots = max(k - len(picked), 0)
    notebook_pool = [sd for sd in (tutorials + sections + recipes + recipe_cards) if _is_notebook_recipe(sd.doc)]
    notebook_picked = _pick_notebook_cluster(notebook_pool, want=min(2, remaining_slots))
    picked.extend([sd.doc for sd in notebook_picked])

    remaining_slots = max(k - len(picked), 0)
    if remaining_slots:
        picked.extend([sd.doc for sd in recipe_cards[:remaining_slots]])
    remaining_slots = max(k - len(picked), 0)
    if remaining_slots:
        picked.extend([sd.doc for sd in recipes[:remaining_slots]])
    remaining_slots = max(k - len(picked), 0)
    if remaining_slots:
        picked.extend([sd.doc for sd in api_surface[:remaining_slots]])
    remaining_slots = max(k - len(picked), 0)
    if remaining_slots:
        picked.extend([sd.doc for sd in rest[:remaining_slots]])

    picked = _dedupe_docs(picked)
    return picked[:k]


def _candidate_valid_symbols(answer: str) -> list[str]:
    aliases = _extract_import_aliases(answer)
    out: set[str] = set()
    for sym in _extract_candidate_symbols(answer):
        resolved = _resolve_alias_symbol(sym, aliases)
        project = resolved.split(".", 1)[0]
        if project not in {"molsysmt", "molsysviewer", "pyunitwizard", "topomt"}:
            continue
        symbols, prefixes = _symbol_sets_for_project(project)
        if not symbols and not prefixes:
            continue
        if resolved in symbols or resolved in prefixes:
            out.add(resolved)
    return sorted(out)


def _retrieve_api_surface_snippets(symbol: str, *, k: int) -> list[Document]:
    """Retrieve symbol-focused snippets relevant to an API symbol (best-effort)."""
    if k <= 0:
        return []
    project = symbol.split(".", 1)[0]
    idx = _PROJECT_INDICES.get(project) or _DOCS_INDEX
    # Pull more candidates and filter down to api_surface sources.
    candidates = retrieve(symbol, idx, k=max(k * 6, k))
    symbol_cards = [d for d in candidates if _is_symbol_card_doc(d)]
    api = [d for d in candidates if _is_api_surface_doc(d)]
    return (symbol_cards or api or candidates)[:k]


def _build_reread_prompt(
    *,
    question: str,
    draft_answer: str,
    context_block: str,
    api_block: str,
    show_sources: bool,
) -> list[dict[str, str]]:
    system = (
        "You are MolSys-AI, a specialist assistant for the UIBCDF MolSysSuite ecosystem.\n"
        "Rewrite the answer to ensure API correctness.\n"
        "- Use the term 'MolSysSuite' (never 'MolSys*').\n"
        "- Use the API excerpts to ensure correct symbol names and usage.\n"
        "- If you cannot confirm a symbol/behavior from the excerpts, say `NOT_DOCUMENTED`.\n"
    )
    if show_sources:
        system += "- Keep bracket citations like [1], [2] aligned with the numbered sources.\n"
    system += "Return ONLY the final answer text.\n"

    prefix = "Documentation excerpts:\n\n" if show_sources else "Documentation excerpts (do not cite):\n\n"
    api_prefix = "\n\nAPI excerpts (do not cite; for correctness):\n\n"
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"{prefix}{context_block}\n\n"
                f"Question: {question}\n\n"
                f"Draft answer:\n{draft_answer}\n"
                f"{api_prefix}{api_block}\n"
            ),
        },
    ]


def _build_symbol_fix_prompt(
    *,
    question: str,
    draft_answer: str,
    context_block: str,
    unknown_symbols: list[str],
    show_sources: bool,
) -> list[dict[str, str]]:
    system = (
        "You are MolSys-AI, a specialist assistant for the UIBCDF MolSysSuite ecosystem.\n"
        "You MUST NOT invent API symbols.\n"
        "The following API symbols are NOT VALID and must not appear in your answer:\n"
        + "\n".join(f"- {s}" for s in unknown_symbols)
        + "\n\n"
        "Rewrite the draft answer so that:\n"
        "- you remove or replace invalid symbols with valid alternatives grounded in the provided excerpts,\n"
        "- if you cannot find a valid alternative, say `NOT_DOCUMENTED` and explain what is missing,\n"
        "- you use the term 'MolSysSuite' (never 'MolSys*'),\n"
    )
    if show_sources:
        system += "- keep bracket citations like [1], [2] aligned with the sources list.\n"
    system += "Return ONLY the final answer text.\n"

    prefix = "Documentation excerpts:\n\n" if show_sources else "Documentation excerpts (do not cite):\n\n"
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"{prefix}{context_block}\n\nQuestion: {question}\n\nDraft answer:\n{draft_answer}\n",
        },
    ]


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
        "Use the term 'MolSysSuite'.\n"
        "Never use the term 'MolSys*'.\n"
        "Prefer precise, actionable answers. If you are not sure, ask a clarifying question.\n"
        "Do not invent APIs, flags, or behaviors.\n"
        "When writing Python snippets, use canonical imports.\n"
        "- For MolSysMT, use: `import molsysmt as msm` (never `import msmt`).\n"
    )


def _user_explicitly_wants_sources(text: str) -> bool:
    s = (text or "").lower()
    return bool(re.search(r"\b(cite|citation|citations|source|sources|link|links|reference|references)\b", s))


def _has_bracket_citations(text: str) -> bool:
    return bool(re.search(r"\[\d+\]", text or ""))

_PDB_ID_RE = re.compile(r"\b[0-9][A-Za-z0-9]{3}\b")


def _extract_pdb_ids(text: str) -> list[str]:
    """Extract likely PDB ids from text, preserving original casing."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _PDB_ID_RE.finditer(text or ""):
        s = m.group(0)
        if s.lower() in seen:
            continue
        seen.add(s.lower())
        out.append(s)
    return out


def _has_bad_molsysmt_alias(text: str) -> bool:
    """Detect common hallucinated alias/module names for molsysmt."""
    s = text or ""
    if re.search(r"\bimport\s+msmt\b", s):
        return True
    if re.search(r"\bmsmt\.", s):
        return True
    return False


def _build_identifier_rewrite_prompt(
    *,
    question: str,
    draft_answer: str,
    context_block: str,
    show_sources: bool,
    pdb_ids: list[str],
    forbid_msmt: bool,
) -> list[dict[str, str]]:
    system = (
        "You are MolSys-AI, a specialist assistant for the UIBCDF MolSysSuite ecosystem.\n"
        "Rewrite the draft answer to preserve user-provided identifiers and ensure API correctness.\n"
        "- Use the term 'MolSysSuite' (never 'MolSys*').\n"
        "- Do not invent APIs, flags, or behaviors.\n"
        "- Keep the answer concise.\n"
    )
    if pdb_ids:
        system += "- You MUST use these PDB ids exactly as provided (do not substitute other examples): " + ", ".join(pdb_ids) + "\n"
    if forbid_msmt:
        system += "- You MUST NOT use `msmt` as a module/alias. Use `import molsysmt as msm`.\n"
    if show_sources:
        system += "- Keep bracket citations like [1], [2] aligned with the sources list.\n"
    system += "Return ONLY the final answer text.\n"

    prefix = "Documentation excerpts:\n\n" if show_sources else "Documentation excerpts (do not cite):\n\n"
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"{prefix}{context_block}\n\n"
                f"Question: {question}\n\n"
                "Draft answer:\n"
                f"{draft_answer}\n"
            ),
        },
    ]


def _infer_project_hint(query: str) -> str | None:
    q = (query or "").lower()
    for name in ("molsysmt", "molsysviewer", "pyunitwizard", "topomt"):
        if name in q:
            return name
    return None


def _doc_project(doc: Document) -> str | None:
    raw = str(doc.metadata.get("path") or "").strip()
    if not raw:
        return None
    try:
        rel = Path(raw).resolve().relative_to(DOCS_SOURCE_DIR.resolve())
    except Exception:
        return None
    if not rel.parts:
        return None
    return rel.parts[0]

def _inject_citations(answer: str, sources: list[Source]) -> str:
    """Best-effort: ensure the answer contains at least one bracket citation.

    Some models may omit citations even when asked. As a last resort, append a
    minimal citation marker so the `sources` list is not orphaned.
    """

    if _has_bracket_citations(answer):
        return answer
    if not sources:
        return answer
    suffix = " [1]"
    if answer.rstrip().endswith((".", "!", "?", "]")):
        return answer.rstrip() + suffix
    return answer.rstrip() + suffix


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

    docs: list[Document] = []
    if use_rag:
        # Tool-aware filtering: when the user names a specific MolSysSuite tool
        # (e.g. MolSysMT), prefer sources from that project to reduce cross-project
        # mixing.
        hint = _infer_project_hint(query)
        is_api_q = _looks_like_api_question(query)

        retrieve_k = max(req_k * 8, 40) if is_api_q else max(req_k * 4, req_k)
        if hint and hint in _PROJECT_INDICES:
            idx = _PROJECT_INDICES[hint]
            scored = retrieve_scored(query, idx, k=retrieve_k)
        else:
            scored = retrieve_scored(query, _DOCS_INDEX, k=retrieve_k)
            if hint:
                filtered = [sd for sd in scored if _doc_project(sd.doc) == hint]
                scored = filtered or scored

        scored = _dedupe_scored_docs(scored)
        _log_scored_docs(query, scored, label=("api" if is_api_q else "generic"))
        docs = _select_scored_for_api_question(scored, k=req_k) if is_api_q else [sd.doc for sd in scored[:req_k]]
        if is_api_q:
            docs = _augment_with_notebook_tutorials(docs, k=req_k)

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
            + "If you include code, only use APIs that appear in the excerpts; otherwise provide a template with clear TODO placeholders.\n"
        )
    elif use_rag and show_sources:
        system_prompt = (
            system_prompt.strip()
            + "\nUse the provided documentation excerpts to answer.\n"
            + "Cite sources by including bracketed numbers like [1] or [2] in your answer.\n"
            + "If the answer cannot be inferred from the excerpts, say so explicitly.\n"
            + "Do not guess function/module names. If you cannot find the exact API in the excerpts, say NOT_DOCUMENTED and point to the most relevant doc URLs.\n"
        )
    elif use_rag and not show_sources:
        system_prompt = (
            system_prompt.strip()
            + "\nUse the provided documentation excerpts to answer.\n"
            + "Do NOT include bracketed citations like [1]. Do NOT mention sources.\n"
            + "If the answer cannot be inferred from the excerpts, say so explicitly.\n"
            + "Do not guess function/module names.\n"
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

    # Enforce stable identifiers (best-effort): if the user provided a PDB id, do not
    # replace it with a different example. This also helps benchmark stability.
    pdb_ids = _extract_pdb_ids(query)
    if pdb_ids:
        answer_l = (answer or "").lower()
        missing = [pid for pid in pdb_ids if pid.lower() not in answer_l]
        forbid_msmt = _has_bad_molsysmt_alias(answer)
        if missing or forbid_msmt:
            try:
                rewrite_messages = _build_identifier_rewrite_prompt(
                    question=query,
                    draft_answer=answer,
                    context_block=context_block,
                    show_sources=show_sources,
                    pdb_ids=pdb_ids,
                    forbid_msmt=forbid_msmt,
                )
                rewritten = client.generate(
                    rewrite_messages,
                    generation={"temperature": 0.0, "top_p": 1.0, "max_tokens": 600},
                )
                rewritten = (rewritten or "").strip()
                if rewritten:
                    answer = rewritten
            except Exception:
                pass

    # Best-effort enforcement: when sources are enabled, ensure the answer includes
    # bracket citations like [1]. This avoids confusing UX where a sources list is
    # returned but the answer does not reference it.
    enforce_citations = _env_bool("MOLSYS_AI_CHAT_ENFORCE_CITATIONS", True)
    if enforce_citations and show_sources and sources and not _has_bracket_citations(answer):
        rewrite_system = (
            system_prompt.strip()
            + "\nYou MUST include bracketed citations like [1] referencing the provided excerpts.\n"
            + "If you cannot answer, say so, but still cite the most relevant excerpt(s).\n"
        )
        rewrite_messages = [
            {"role": "system", "content": rewrite_system},
            {
                "role": "user",
                "content": (
                    "Documentation excerpts:\n\n"
                    f"{context_block}\n\n"
                    f"Question: {query}\n\n"
                    "Draft answer (missing citations):\n"
                    f"{answer}\n\n"
                    "Rewrite the answer so it includes citations like [1]. Keep it concise."
                ),
            },
        ]
        rewritten = client.generate(
            rewrite_messages,
            generation={"temperature": 0.0, "top_p": 1.0, "max_tokens": 512},
        )
        if _has_bracket_citations(rewritten):
            answer = rewritten
        else:
            # Fallback: avoid returning a sources list without any citation markers.
            answer = _inject_citations(answer, sources)

    # Symbol verification (best-effort): prevent invented MolSysSuite API names.
    if _env_bool("MOLSYS_AI_CHAT_VERIFY_SYMBOLS", True):
        unknown = _unknown_tool_symbols(answer)
        if unknown:
            try:
                fix_messages = _build_symbol_fix_prompt(
                    question=query,
                    draft_answer=answer,
                    context_block=context_block,
                    unknown_symbols=unknown,
                    show_sources=show_sources,
                )
                fixed = client.generate(
                    fix_messages,
                    generation={"temperature": 0.0, "top_p": 1.0, "max_tokens": 800},
                )
                fixed = (fixed or "").strip()
                if fixed:
                    answer = fixed
            except Exception:
                pass

            still_unknown = _unknown_tool_symbols(answer)
            if still_unknown:
                answer = (
                    answer.rstrip()
                    + "\n\nNOT_DOCUMENTED: I could not verify these API symbols in the available corpus: "
                    + ", ".join(still_unknown)
                )
            # Ensure citations aren't accidentally dropped by the rewrite/append.
            if enforce_citations and show_sources and sources and not _has_bracket_citations(answer):
                answer = _inject_citations(answer, sources)

    # Symbol re-read (best-effort): if the answer mentions valid API symbols, retrieve
    # their API-surface snippets and force a rewrite for accuracy.
    if use_rag and _env_bool("MOLSYS_AI_CHAT_REREAD_SYMBOLS", True):
        max_syms = max(_env_int("MOLSYS_AI_CHAT_REREAD_MAX_SYMBOLS", 2), 0)
        k_per = max(_env_int("MOLSYS_AI_CHAT_REREAD_K_PER_SYMBOL", 2), 0)
        if max_syms > 0 and k_per > 0:
            candidates = _candidate_valid_symbols(answer)[:max_syms]
            api_docs: list[Document] = []
            for s in candidates:
                api_docs.extend(_retrieve_api_surface_snippets(s, k=k_per))
            # De-duplicate by path+section+label+content prefix.
            seen = set()
            uniq: list[Document] = []
            for d in api_docs:
                key = (
                    str(d.metadata.get("path") or ""),
                    str(d.metadata.get("section") or ""),
                    str(d.metadata.get("label") or ""),
                    (d.content or "")[:200],
                )
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(d)
            api_docs = uniq[: max_syms * k_per]

            if api_docs:
                api_lines: list[str] = []
                for d in api_docs:
                    try:
                        rel = Path(str(d.metadata.get("path") or "")).resolve().relative_to(DOCS_SOURCE_DIR.resolve())
                        p = str(rel)
                    except Exception:
                        p = str(d.metadata.get("path") or "unknown")
                    api_lines.append(f"- {p}\n{d.content}\n")
                api_block = "\n".join(api_lines).strip() or "(no API excerpts found)"
                try:
                    reread_messages = _build_reread_prompt(
                        question=query,
                        draft_answer=answer,
                        context_block=context_block,
                        api_block=api_block,
                        show_sources=show_sources,
                    )
                    reread = client.generate(
                        reread_messages,
                        generation={"temperature": 0.0, "top_p": 1.0, "max_tokens": 900},
                    )
                    reread = (reread or "").strip()
                    if reread:
                        answer = reread
                except Exception:
                    pass

                # Re-run symbol verification after the rewrite.
                if _env_bool("MOLSYS_AI_CHAT_VERIFY_SYMBOLS", True):
                    still_unknown = _unknown_tool_symbols(answer)
                    if still_unknown:
                        answer = (
                            answer.rstrip()
                            + "\n\nNOT_DOCUMENTED: I could not verify these API symbols in the available corpus: "
                            + ", ".join(still_unknown)
                        )

                # Re-enforce citations if needed (the rewrite might remove them).
                if enforce_citations and show_sources and sources and not _has_bracket_citations(answer):
                    answer = _inject_citations(answer, sources)

    # Final safety net: if sources are enabled, never return a sources list without citations.
    # (Even when `MOLSYS_AI_CHAT_ENFORCE_CITATIONS=0`, we still inject a minimal marker like
    # "[1]" so the sources list is not orphaned.)
    if show_sources and sources and not _has_bracket_citations(answer):
        answer = _inject_citations(answer, sources)

    return ChatResponse(answer=answer, sources=sources if show_sources else [])


@app.on_event("startup")
async def _load_or_build_docs_index_on_startup() -> None:
    global _DOCS_INDEX, _PROJECT_INDICES
    force_rebuild = _env_bool("MOLSYS_AI_DOCS_INDEX_REBUILD", False)
    if DOCS_INDEX_PATH.exists() and not force_rebuild:
        print(f"Loading existing index from {DOCS_INDEX_PATH}")
        _DOCS_INDEX = load_index(DOCS_INDEX_PATH)
    else:
        print(f"Building index from {DOCS_SOURCE_DIR}...")
        build_index(DOCS_SOURCE_DIR, DOCS_INDEX_PATH)
        _DOCS_INDEX = load_index(DOCS_INDEX_PATH)
    print(f"Index loaded with {_DOCS_INDEX.__len__()} documents.")

    # Optional segmented indices (one per project).
    _PROJECT_INDICES = {}
    if PROJECT_INDEX_DIR.exists():
        for name in ("molsysmt", "molsysviewer", "pyunitwizard", "topomt"):
            p = PROJECT_INDEX_DIR / f"{name}.pkl"
            if p.exists():
                try:
                    _PROJECT_INDICES[name] = load_index(p)
                    print(f"Loaded project index {name}: {len(_PROJECT_INDICES[name])} documents from {p}")
                except Exception:
                    continue

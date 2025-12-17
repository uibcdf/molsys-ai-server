#!/usr/bin/env python3
"""Sync a documentation corpus for MolSys-AI RAG from sibling repos.

This script snapshots selected text files from live, sibling repositories
into `server/chat_api/data/docs/` so that the chat API can build a
local RAG index.

Default sources (relative to this repo):

- ../molsysmt
- ../molsysviewer
- ../pyunitwizard
- ../topomt

The output is deterministic and includes a manifest with commit hashes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Tuple

try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parent.parent
# Allow running this script without installing the repo as a package by ensuring
# `server/` is on sys.path (so `import rag...` works).
SERVER_DIR = REPO_ROOT / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))
DEFAULT_DEST = REPO_ROOT / "server" / "chat_api" / "data" / "docs"
DEFAULT_INDEX = REPO_ROOT / "server" / "chat_api" / "data" / "rag_index.pkl"
DEFAULT_ANCHORS = REPO_ROOT / "server" / "chat_api" / "data" / "anchors.json"

DEFAULT_SOURCES = {
    "molsysmt": REPO_ROOT.parent / "molsysmt",
    "molsysviewer": REPO_ROOT.parent / "molsysviewer",
    "pyunitwizard": REPO_ROOT.parent / "pyunitwizard",
    "topomt": REPO_ROOT.parent / "topomt",
}

DEFAULT_DOCS_BASE_URL = {
    "molsysmt": "https://www.uibcdf.org/molsysmt",
    "molsysviewer": "https://www.uibcdf.org/molsysviewer",
    "pyunitwizard": "https://www.uibcdf.org/pyunitwizard",
    "topomt": "https://www.uibcdf.org/topomt",
}

DEFAULT_INCLUDE_DIRS = ("docs", "doc", "devguide")
DEFAULT_INCLUDE_ROOT_GLOBS = (
    "README*.md",
    "README*.rst",
    "dev_guide.md",
    "README_DEVELOPERS*.md",
    "notes*.md",
    "ideas*.md",
    "CHANGELOG*.md",
)

DEFAULT_TEXT_EXTS = {".md", ".rst", ".txt", ".ipynb"}

DEFAULT_EXCLUDE_DIR_NAMES = {
    ".git",
    ".github",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    ".venv",
    "venv",
    "env",
    "_build",
    "build",
    "dist",
    "node_modules",
    "attic",
    "logos",
    "paper",
}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class SelectionSpec:
    include_dirs: tuple[str, ...]
    include_root_globs: tuple[str, ...]
    exts: set[str]
    exclude_dir_names: set[str]
    docs_base_url: str | None


def _safe_read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None


def _find_primary_python_package(repo_root: Path, project_name: str) -> Path | None:
    """Best-effort discovery of the primary Python package directory for a repo."""

    direct = repo_root / project_name
    if (direct / "__init__.py").exists():
        return direct

    # Common layout: <repo>/<project_name>/<project_name>/__init__.py
    nested = repo_root / project_name / project_name
    if (nested / "__init__.py").exists():
        return nested

    # Fallback: scan one level for a package with __init__.py.
    candidates: list[Path] = []
    for p in repo_root.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith("."):
            continue
        if (p / "__init__.py").exists():
            candidates.append(p)
    if candidates:
        # Prefer the project name if present, otherwise largest by file count.
        for c in candidates:
            if c.name == project_name:
                return c
        candidates.sort(key=lambda d: sum(1 for _ in d.rglob("*.py")), reverse=True)
        return candidates[0]
    return None


def _format_signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = fn.args
    parts: list[str] = []
    for a in args.posonlyargs:
        parts.append(a.arg)
    if args.posonlyargs:
        parts.append("/")
    for a in args.args:
        parts.append(a.arg)
    if args.vararg:
        parts.append("*" + args.vararg.arg)
    elif args.kwonlyargs:
        parts.append("*")
    for a in args.kwonlyargs:
        parts.append(a.arg)
    if args.kwarg:
        parts.append("**" + args.kwarg.arg)
    return f"{fn.name}({', '.join(parts)})"


def _normalize_docstring(text: str) -> str:
    # Keep docstrings compact but readable. Avoid excessive whitespace noise.
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    # Drop leading/trailing blank lines.
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def _doc_excerpt(doc: str, *, max_chars: int) -> str:
    d = _normalize_docstring(doc)
    if not d:
        return ""
    if max_chars <= 0:
        return d
    if len(d) <= max_chars:
        return d
    return d[: max(0, max_chars - 1)].rstrip() + "…"


def _module_qual(pkg_name: str, rel_no_suffix: Path) -> str:
    """Return a canonical Python module path from a package name and a relative path.

    Examples:
      rel_no_suffix="__init__" -> "molsysmt"
      rel_no_suffix="structure/__init__" -> "molsysmt.structure"
      rel_no_suffix="structure/get_rmsd" -> "molsysmt.structure.get_rmsd"
    """
    tail = rel_no_suffix.as_posix().replace("/", ".")
    if tail == "__init__":
        return pkg_name
    if tail.endswith(".__init__"):
        tail = tail[: -len(".__init__")]
    return f"{pkg_name}.{tail}".strip(".")


def _module_rel_dir(rel_py: Path) -> Path:
    """Return a directory-like relative path for a module file.

    Examples:
      structure/get_rmsd.py -> structure/get_rmsd
      structure/__init__.py -> structure
      __init__.py -> _root
    """
    rel_no_suffix = rel_py.with_suffix("")
    parts = list(rel_no_suffix.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return Path("_root")
    return Path(*parts)


def _ast_node_span(node: ast.AST) -> tuple[int | None, int | None]:
    lineno = getattr(node, "lineno", None)
    end_lineno = getattr(node, "end_lineno", None)
    try:
        lineno = int(lineno) if lineno is not None else None
    except Exception:
        lineno = None
    try:
        end_lineno = int(end_lineno) if end_lineno is not None else None
    except Exception:
        end_lineno = None
    return lineno, end_lineno


def _format_signature_typed(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Best-effort signature formatting, including annotations when possible."""

    def fmt_ann(a: ast.AST | None) -> str:
        if a is None:
            return ""
        try:
            return ast.unparse(a)
        except Exception:
            return ""

    args = fn.args
    parts: list[str] = []
    for a in args.posonlyargs:
        ann = fmt_ann(a.annotation)
        parts.append(f"{a.arg}: {ann}" if ann else a.arg)
    if args.posonlyargs:
        parts.append("/")
    for a in args.args:
        ann = fmt_ann(a.annotation)
        parts.append(f"{a.arg}: {ann}" if ann else a.arg)
    if args.vararg:
        ann = fmt_ann(args.vararg.annotation)
        name = "*" + args.vararg.arg
        parts.append(f"{name}: {ann}" if ann else name)
    elif args.kwonlyargs:
        parts.append("*")
    for a in args.kwonlyargs:
        ann = fmt_ann(a.annotation)
        parts.append(f"{a.arg}: {ann}" if ann else a.arg)
    if args.kwarg:
        ann = fmt_ann(args.kwarg.annotation)
        name = "**" + args.kwarg.arg
        parts.append(f"{name}: {ann}" if ann else name)
    ret = fmt_ann(fn.returns)
    suffix = f" -> {ret}" if ret else ""
    return f"{fn.name}({', '.join(parts)}){suffix}"


def _write_symbol_cards_snapshot(
    *,
    spec: SourceSpec,
    repo_root: Path,
    dest: Path,
    max_bytes: int,
    max_modules: int,
    include_private: bool,
    max_doc_chars: int,
) -> dict[str, object]:
    """Generate per-symbol 'symbol cards' from Python source via AST (no imports).

    Output layout:
      <dest>/<project>/symbol_cards/<module_path>/<symbol>.md
    """

    pkg = _find_primary_python_package(repo_root, spec.name)
    if pkg is None:
        return {"ok": False, "error": "No Python package directory found."}

    out_root = dest / spec.name / "symbol_cards"
    out_root.mkdir(parents=True, exist_ok=True)

    modules: list[Path] = []
    discovered = 0
    skipped_private = 0
    skipped_large = 0
    for p in pkg.rglob("*.py"):
        if _should_skip_dir(p, DEFAULT_EXCLUDE_DIR_NAMES):
            continue
        discovered += 1
        if not include_private and p.name.startswith("_") and p.name != "__init__.py":
            skipped_private += 1
            continue
        try:
            if p.stat().st_size > max_bytes:
                skipped_large += 1
                continue
        except OSError:
            continue
        modules.append(p)

    modules.sort()
    limited_by_max_modules = 0
    if max_modules > 0 and len(modules) > max_modules:
        limited_by_max_modules = len(modules) - max_modules
        modules = modules[:max_modules]

    cards_written = 0
    modules_written = 0
    for m in modules:
        text = _safe_read_text(m)
        if not text:
            continue
        try:
            mod = ast.parse(text)
        except SyntaxError:
            continue

        rel = m.resolve().relative_to(pkg.resolve())
        module_qual = _module_qual(pkg.name, rel.with_suffix(""))
        module_dir = _module_rel_dir(rel)

        module_doc = ast.get_docstring(mod) or ""
        module_doc = _doc_excerpt(module_doc, max_chars=max_doc_chars)

        wrote_any = False
        for node in mod.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sym_name = node.name
                sym_qual = f"{module_qual}.{sym_name}"
                sig = _format_signature_typed(node)
                doc = _doc_excerpt(ast.get_docstring(node) or "", max_chars=max_doc_chars)
                lineno, end_lineno = _ast_node_span(node)
                out_path = out_root / module_dir / f"{sym_name}.md"
                _ensure_parent(out_path)
                header = (
                    f"# Symbol card: `{sym_qual}`\n\n"
                    f"Project: {spec.name}\n"
                    f"Module: {module_qual}\n"
                    f"Symbol: {sym_qual}\n"
                    f"Kind: function\n"
                    f"Source: {m}:{lineno}-{end_lineno}\n\n"
                )
                body = (
                    f"## Signature\n\n`{sig}`\n\n"
                    + ("## Module doc\n\n" + module_doc + "\n\n" if module_doc else "")
                    + ("## Docstring\n\n" + doc + "\n\n" if doc else "")
                )
                data = (header + body).encode("utf-8")
                out_path.write_bytes(data[:max_bytes] if len(data) > max_bytes else data)
                cards_written += 1
                wrote_any = True
            elif isinstance(node, ast.ClassDef):
                class_name = node.name
                class_qual = f"{module_qual}.{class_name}"
                class_doc = _doc_excerpt(ast.get_docstring(node) or "", max_chars=max_doc_chars)
                lineno, end_lineno = _ast_node_span(node)
                out_path = out_root / module_dir / f"{class_name}.md"
                _ensure_parent(out_path)
                header = (
                    f"# Symbol card: `{class_qual}`\n\n"
                    f"Project: {spec.name}\n"
                    f"Module: {module_qual}\n"
                    f"Symbol: {class_qual}\n"
                    f"Kind: class\n"
                    f"Source: {m}:{lineno}-{end_lineno}\n\n"
                )
                body = (
                    "## Signature\n\n"
                    f"`class {class_name}`\n\n"
                    + ("## Module doc\n\n" + module_doc + "\n\n" if module_doc else "")
                    + ("## Docstring\n\n" + class_doc + "\n\n" if class_doc else "")
                )
                out_path.write_bytes((header + body).encode("utf-8")[:max_bytes])
                cards_written += 1
                wrote_any = True

                # Methods as separate cards.
                for sub in node.body:
                    if not isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    meth_name = sub.name
                    meth_qual = f"{class_qual}.{meth_name}"
                    sig = _format_signature_typed(sub)
                    doc = _doc_excerpt(ast.get_docstring(sub) or "", max_chars=max_doc_chars)
                    ml1, ml2 = _ast_node_span(sub)
                    meth_path = out_root / module_dir / class_name / f"{meth_name}.md"
                    _ensure_parent(meth_path)
                    mheader = (
                        f"# Symbol card: `{meth_qual}`\n\n"
                        f"Project: {spec.name}\n"
                        f"Module: {module_qual}\n"
                        f"Symbol: {meth_qual}\n"
                        f"Kind: method\n"
                        f"Source: {m}:{ml1}-{ml2}\n\n"
                    )
                    mbody = (
                        f"## Signature\n\n`{sig}`\n\n"
                        f"## Parent class\n\n`{class_qual}`\n\n"
                        + ("## Docstring\n\n" + doc + "\n\n" if doc else "")
                    )
                    meth_path.write_bytes((mheader + mbody).encode("utf-8")[:max_bytes])
                    cards_written += 1
                    wrote_any = True

        if wrote_any:
            modules_written += 1

    return {
        "ok": True,
        "package_dir": str(pkg),
        "symbol_cards_dir": str(out_root),
        "max_modules": int(max_modules),
        "include_private": bool(include_private),
        "max_doc_chars": int(max_doc_chars),
        "modules_discovered": int(discovered),
        "modules_skipped_private": int(skipped_private),
        "modules_skipped_large": int(skipped_large),
        "modules_limited_by_max_modules": int(limited_by_max_modules),
        "modules_written": int(modules_written),
        "cards_written": int(cards_written),
    }


_PY_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_\.]+)", flags=re.M)


def _projects_mentioned_in_python(code: str) -> set[str]:
    out: set[str] = set()
    for m in _PY_IMPORT_RE.findall(code or ""):
        head = m.split(".", 1)[0]
        if head in {"molsysmt", "molsysviewer", "pyunitwizard", "topomt"}:
            out.add(head)
    return out


def _extract_import_aliases_from_python(code: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    s = code or ""
    for project in ("molsysmt", "molsysviewer", "pyunitwizard", "topomt"):
        for a in re.findall(rf"^\s*import\s+{project}\s+as\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", s, flags=re.M):
            aliases[a] = project
        for a in re.findall(
            rf"^\s*from\s+{project}\s+import\s+.+?\s+as\s+([A-Za-z_][A-Za-z0-9_]*)\s*$",
            s,
            flags=re.M,
        ):
            aliases[a] = project
    return aliases


def _extract_dotted_symbols_from_text(text: str) -> set[str]:
    # Keep it conservative: dotted paths only.
    return set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b", text or ""))


def _resolve_alias_symbol(sym: str, aliases: dict[str, str]) -> str:
    head, *rest = sym.split(".")
    if head in aliases:
        return ".".join([aliases[head], *rest])
    return sym


def _extract_symbols_from_python_snippet(code: str, *, extra_aliases: dict[str, str] | None = None) -> dict[str, set[str]]:
    """Best-effort MolSysSuite symbol extraction from a Python snippet.

    This is intentionally conservative and avoids imports/execution.
    `extra_aliases` can provide notebook-level alias mappings so later cells
    can be attributed correctly (e.g. `msmt` -> `molsysmt`).
    """
    aliases = _extract_import_aliases_from_python(code)
    if extra_aliases:
        aliases = {**extra_aliases, **aliases}
    out: dict[str, set[str]] = {p: set() for p in ("molsysmt", "molsysviewer", "pyunitwizard", "topomt")}
    for sym in _extract_dotted_symbols_from_text(code):
        resolved = _resolve_alias_symbol(sym, aliases)
        project = resolved.split(".", 1)[0]
        if project in out:
            out[project].add(resolved)
    # Include imported symbols from `from <project>.<module> import x` (AST-based, no execution).
    try:
        mod = ast.parse(code or "")
    except SyntaxError:
        mod = None
    if mod is not None:
        for node in mod.body:
            if not isinstance(node, ast.ImportFrom):
                continue
            if not isinstance(node.module, str) or not node.module:
                continue
            head = node.module.split(".", 1)[0]
            if head not in out:
                continue
            for alias in node.names or []:
                if not isinstance(alias, ast.alias):
                    continue
                if not alias.name or alias.name == "*":
                    continue
                # If importing from a submodule, use that dotted name.
                # Example: from molsysmt.structure import get_rmsd -> molsysmt.structure.get_rmsd
                out[head].add(f"{node.module}.{alias.name}")
    return {k: v for k, v in out.items() if v}


def _python_defs_and_uses(code: str) -> tuple[set[str], set[str]]:
    """Return (defined_names, used_names) for a Python snippet (AST; no imports/execution).

    This is used to stitch notebook sections by pulling a small "preamble" of
    earlier cells when later code cells depend on earlier definitions.

    We track:
    - simple names (`x`)
    - attribute chains (`obj.attr`, `obj.attr.subattr`)
    - simple subscripts with constant keys (`d['key']`, `arr[0]`)
    """
    try:
        mod = ast.parse(code or "")
    except SyntaxError:
        return set(), set()

    defined: set[str] = set()
    used: set[str] = set()

    def attr_chain(node: ast.AST) -> str | None:
        # Build "a.b.c" if it's a simple Attribute chain rooted in a Name.
        parts: list[str] = []
        cur: ast.AST | None = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
        return None

    def subscript_key(node: ast.Subscript) -> str | None:
        # Best-effort: d['key'] / d[0]
        base = attr_chain(node.value) if isinstance(node.value, ast.Attribute) else (node.value.id if isinstance(node.value, ast.Name) else None)
        if not base:
            return None
        sl = node.slice
        # Python 3.9+: slice is an expression.
        if isinstance(sl, ast.Constant) and isinstance(sl.value, (str, int)):
            val = sl.value
            if isinstance(val, str):
                return f"{base}['{val}']"
            return f"{base}[{val}]"
        return None

    def add_target(t: ast.AST) -> None:
        # Assignment targets.
        if isinstance(t, ast.Name):
            defined.add(t.id)
            return
        if isinstance(t, ast.Attribute):
            chain = attr_chain(t)
            if chain:
                defined.add(chain)
            # Also treat the base object as "used" (e.g. `obj.attr = ...` depends on `obj` existing).
            base = attr_chain(t.value) if isinstance(t.value, ast.Attribute) else (t.value.id if isinstance(t.value, ast.Name) else None)
            if base:
                used.add(base)
            return
        if isinstance(t, ast.Subscript):
            key = subscript_key(t)
            if key:
                defined.add(key)
            # Add base object as used.
            base = attr_chain(t.value) if isinstance(t.value, ast.Attribute) else (t.value.id if isinstance(t.value, ast.Name) else None)
            if base:
                used.add(base)
            return
        if isinstance(t, (ast.Tuple, ast.List)):
            for elt in t.elts:
                add_target(elt)
            return

    for node in ast.walk(mod):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if getattr(node, "name", None):
                defined.add(str(node.name))
        elif isinstance(node, ast.Import):
            for a in node.names or []:
                if not isinstance(a, ast.alias) or not a.name:
                    continue
                defined.add(a.asname or a.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names or []:
                if not isinstance(a, ast.alias) or not a.name or a.name == "*":
                    continue
                defined.add(a.asname or a.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                add_target(t)
        elif isinstance(node, ast.AnnAssign):
            add_target(node.target)
        elif isinstance(node, ast.AugAssign):
            add_target(node.target)
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                defined.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                used.add(node.id)
        elif isinstance(node, ast.Attribute):
            if isinstance(node.ctx, ast.Load):
                chain = attr_chain(node)
                if chain:
                    used.add(chain)
        elif isinstance(node, ast.Subscript):
            if isinstance(node.ctx, ast.Load):
                key = subscript_key(node)
                if key:
                    used.add(key)

    return defined, used


def _builtin_names() -> set[str]:
    b = __builtins__
    if isinstance(b, dict):
        return set(str(k) for k in b.keys())
    return set(dir(b))


_BUILTINS = _builtin_names() | {"True", "False", "None"}


def _normalize_molsysmt_alias(code: str) -> str:
    """Normalize legacy MolSysMT alias usage to the canonical `msm`.

    Some historical notebooks use `import molsysmt as msmt` and then call `msmt.*`.
    This alias is easy for LLMs to over-generalize into hallucinated `msmt` modules.

    We do NOT rewrite the literal docs snapshot; we only normalize derived recipe snippets
    to encourage consistent, modern examples.
    """

    s = code or ""
    # Normalize the import alias itself.
    s = re.sub(r"(^\s*import\s+molsysmt\s+as\s+)msmt(\s*$)", r"\1msm\2", s, flags=re.M)
    # Normalize attribute access (best-effort). Keep it conservative: `msmt.` as a token.
    s = re.sub(r"\bmsmt\.", "msm.", s)
    return s


def _split_markdown_into_heading_segments(text: str) -> list[tuple[str | None, str]]:
    """Split Markdown into segments keyed by the most recent heading line.

    Returns list of (heading_line_or_None, segment_text). A segment beginning with a heading
    includes that heading line in its text. MyST label lines like `(Label)=` are attached to
    the next heading segment when present.
    """
    lines = (text or "").splitlines()
    out: list[tuple[str | None, str]] = []
    pending_label_lines: list[str] = []
    current_heading: str | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        seg = "\n".join(buf).strip()
        buf = []
        if seg:
            out.append((current_heading, seg))

    for ln in lines:
        s = ln.strip()
        if s.startswith("(") and s.endswith(")=") and len(s) >= 4:
            pending_label_lines.append(ln)
            continue
        if s.startswith("#"):
            flush()
            # Start a new segment for this heading.
            current_heading = s
            if pending_label_lines:
                buf.extend(pending_label_lines)
                pending_label_lines = []
            buf.append(ln)
            continue
        buf.append(ln)

    flush()
    return out


def _write_recipes_snapshot_from_ipynb(
    *,
    spec: SourceSpec,
    dest: Path,
    max_bytes: int,
    max_code_chars: int,
    max_markdown_chars: int,
) -> dict[str, object]:
    """Generate 'recipe' docs from code cells in already-snapshotted notebooks."""

    root = dest / spec.name
    if not root.exists():
        return {"ok": False, "error": "Project snapshot not found."}

    out_cells_root = root / "recipes" / "notebooks"
    out_sections_root = root / "recipes" / "notebooks_sections"
    out_tutorials_root = root / "recipes" / "notebooks_tutorials"
    out_cells_root.mkdir(parents=True, exist_ok=True)
    out_sections_root.mkdir(parents=True, exist_ok=True)
    out_tutorials_root.mkdir(parents=True, exist_ok=True)

    notebooks = sorted(root.rglob("*.ipynb"))
    written_cells = 0
    written_sections = 0
    written_tutorials = 0
    for nb_path in notebooks:
        try:
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cells = nb.get("cells") or []
        if not isinstance(cells, list) or not cells:
            continue

        def first_heading(md: str) -> str:
            for ln in (md or "").splitlines():
                s = ln.strip()
                if s.startswith("#"):
                    return s
            return ""

        def cell_tags(cell: dict[str, object]) -> set[str]:
            meta = cell.get("metadata")
            if isinstance(meta, dict):
                tags = meta.get("tags")
                if isinstance(tags, list):
                    return {t for t in tags if isinstance(t, str) and t}
            return set()

        # Collect markdown context per cell (previous markdown cell).
        md_by_idx: dict[int, str] = {}
        last_md = ""
        current_heading = ""
        for i, cell in enumerate(cells):
            if not isinstance(cell, dict):
                continue
            if cell.get("cell_type") != "markdown":
                continue
            src = cell.get("source")
            txt = "".join(src) if isinstance(src, list) else (src if isinstance(src, str) else "")
            txt = (txt or "").strip()
            if txt:
                last_md = txt
                hd = first_heading(txt)
                if hd:
                    current_heading = hd
            md_by_idx[i] = last_md

        # Notebook-level state:
        # - Once we see an import from the project, we keep later code cells as candidates.
        # - Track import aliases across cells so later cells can resolve e.g. `msmt.*`.
        seen_projects: set[str] = set()
        notebook_aliases: dict[str, str] = {}
        project = spec.name

        # Section-level recipe blocks: group multiple code cells under the same markdown heading.
        blocks: list[dict[str, object]] = []
        block_heading = ""
        block_md_parts: list[str] = []
        block_cells: list[tuple[int, str]] = []
        bootstrap_lines: list[str] = []
        # Setup cells: a few early cells before the first "real" section (## ...).
        # These cells may not define variables needed later, but can still be
        # important for runnable snippets (e.g. configuration/display calls).
        setup_cell_indices: list[int] = []
        setup_phase = True
        cell_info: dict[int, dict[str, object]] = {}

        def flush_block() -> None:
            nonlocal block_heading, block_md_parts, block_cells, written_sections
            if not block_cells:
                return
            rel_nb = nb_path.relative_to(root)
            block_id = len(blocks)

            # Determine a stitched preamble: import lines + earlier definitions needed by this section.
            block_idxs = [i for i, _ in block_cells]
            block_start = min(block_idxs)
            # Compute "needed before defined" names in execution order, so we stitch variables
            # that are used at the start of the section even if they are reassigned later.
            needed: set[str] = set()
            defined_so_far: set[str] = set()
            for ci, _ in block_cells:
                info = cell_info.get(ci) or {}
                used = set(info.get("used") or set())
                defined = set(info.get("defined") or set())
                needed |= (used - defined_so_far) - _BUILTINS
                defined_so_far |= defined

            selected_preamble: list[int] = []
            selected_set: set[int] = set()
            rounds = 0
            max_preamble_cells = 6
            while needed and rounds < 3 and len(selected_set) < max_preamble_cells:
                rounds += 1
                changed = False
                for pi in sorted(cell_info.keys(), reverse=True):
                    if pi >= block_start or pi in selected_set:
                        continue
                    info = cell_info.get(pi) or {}
                    defs = set(info.get("defined") or set())
                    uses = set(info.get("used") or set())
                    if defs & needed:
                        selected_set.add(pi)
                        selected_preamble.append(pi)
                        # Remove what we satisfied, and add transitive dependencies.
                        needed -= defs
                        needed |= (uses - defs) - _BUILTINS
                        changed = True
                    if len(selected_set) >= max_preamble_cells:
                        break
                if not changed:
                    break
            selected_preamble = sorted(selected_set)

            # Build combined code with lightweight cell separators.
            code_parts: list[str] = []

            def import_lines_in(snippet: str) -> set[str]:
                out: set[str] = set()
                for ln in (snippet or "").splitlines():
                    if re.match(r"^\s*(?:from|import)\s+", ln):
                        s = ln.strip()
                        if s:
                            out.add(s)
                return out

            present_import_lines: set[str] = set()
            for pi in selected_preamble:
                present_import_lines |= import_lines_in(str((cell_info.get(pi) or {}).get("raw") or ""))
            for _, ccode in block_cells:
                present_import_lines |= import_lines_in(ccode)

            if bootstrap_lines:
                filtered = [ln for ln in bootstrap_lines if (ln or "").strip() and (ln or "").strip() not in present_import_lines]
                if filtered:
                    code_parts.append("\n".join(filtered).strip())
            # Include a small number of early "setup" cells (before the first real section).
            # These can capture important context even when no variables are defined.
            setup_limit = 2
            setup_selected = [i for i in setup_cell_indices if i < block_start and i not in selected_set][:setup_limit]
            if setup_selected:
                for pi in setup_selected:
                    pcode = str((cell_info.get(pi) or {}).get("raw") or "").strip()
                    if pcode:
                        code_parts.append(f"# --- notebook setup cell {pi} ---\n{pcode}")
            if selected_preamble:
                for pi in selected_preamble:
                    pcode = str((cell_info.get(pi) or {}).get("raw") or "").strip()
                    if pcode:
                        code_parts.append(f"# --- notebook preamble cell {pi} ---\n{pcode}")
            for ci, ccode in block_cells:
                ccode = ccode.strip()
                if not ccode:
                    continue
                code_parts.append(f"# --- notebook cell {ci} ---\n{ccode}")
            combined = "\n\n".join(p for p in code_parts if p.strip()).strip()
            if not combined:
                block_cells = []
                return
            if max_code_chars > 0 and len(combined) > max_code_chars:
                combined = combined[: max(0, max_code_chars - 1)].rstrip() + "…"

            # Aggregate symbols across the combined code.
            symbols_by_project = _extract_symbols_from_python_snippet(combined, extra_aliases=notebook_aliases)
            symbols = sorted(symbols_by_project.get(project) or [])

            heading = (block_heading or "").strip() or "(no heading)"
            md_ctx = "\n\n".join([p.strip() for p in block_md_parts if p.strip()]).strip()
            if max_markdown_chars > 0 and len(md_ctx) > max_markdown_chars:
                md_ctx = md_ctx[: max(0, max_markdown_chars - 1)].rstrip() + "…"

            out_path = out_sections_root / rel_nb.with_suffix("") / f"section_{block_id:04d}.md"
            _ensure_parent(out_path)
            tutorial_hint = (out_tutorials_root / rel_nb.with_suffix("") / "tutorial.md").relative_to(root)
            header = (
                "# Recipe (notebook section)\n\n"
                f"Project: {project}\n"
                f"Notebook: {spec.name}/{rel_nb}\n"
                f"Tutorial: {spec.name}/{tutorial_hint.as_posix()}\n"
                f"Section: {heading}\n"
                f"Cells: {', '.join(str(i) for i, _ in block_cells)}\n"
                + (f"Setup cells: {', '.join(str(i) for i in setup_selected)}\n" if setup_selected else "")
                + (f"Preamble cells: {', '.join(str(i) for i in selected_preamble)}\n" if selected_preamble else "")
                + (f"Symbols: {', '.join(symbols)}\n" if symbols else "")
                + "\n"
            )
            body = ""
            if md_ctx:
                body += "## Context (from notebook)\n\n" + md_ctx + "\n\n"
            body += "## Code\n\n```python\n" + combined + "\n```\n"
            data = (header + body).encode("utf-8")
            out_path.write_bytes(data[:max_bytes] if max_bytes > 0 and len(data) > max_bytes else data)
            written_sections += 1
            blocks.append({"heading": heading, "cells": [i for i, _ in block_cells]})
            block_cells = []
            block_heading = ""
            block_md_parts = []

        for i, cell in enumerate(cells):
            if not isinstance(cell, dict):
                continue
            if cell.get("cell_type") == "markdown":
                # If this markdown cell starts a new section, close the previous block.
                src = cell.get("source")
                txt = "".join(src) if isinstance(src, list) else (src if isinstance(src, str) else "")
                txt = (txt or "").strip()
                if txt:
                    for hd, seg in _split_markdown_into_heading_segments(txt):
                        seg = (seg or "").strip()
                        if not seg:
                            continue
                        if hd:
                            current_heading = hd
                            if hd.startswith("##"):
                                setup_phase = False
                            if block_cells:
                                flush_block()
                            block_heading = hd
                        block_md_parts.append(seg)
                continue
            if cell.get("cell_type") != "code":
                continue
            if cell_tags(cell).intersection({"remove-input", "remove-cell"}):
                continue
            src = cell.get("source")
            code = "".join(src) if isinstance(src, list) else (src if isinstance(src, str) else "")
            code = (code or "").strip()
            if not code:
                continue
            if project == "molsysmt":
                code = _normalize_molsysmt_alias(code)

            # Update notebook-level context before deciding to include this cell.
            seen_projects.update(_projects_mentioned_in_python(code))
            notebook_aliases.update(_extract_import_aliases_from_python(code))

            symbols_by_project = _extract_symbols_from_python_snippet(code, extra_aliases=notebook_aliases)

            # Track definitions/uses for *all* visible code cells so section stitching can
            # pull earlier variable definitions even when a cell does not mention MolSysSuite.
            defined, used = _python_defs_and_uses(code)
            cell_info[i] = {"raw": code, "defined": defined, "used": used}
            if setup_phase and len(setup_cell_indices) < 5:
                setup_cell_indices.append(i)

            # If we haven't seen any import for this project in the notebook yet, and this cell
            # doesn't mention the project explicitly, skip (avoid indexing generic boilerplate).
            if project not in seen_projects and project not in symbols_by_project:
                continue

            # Update bootstrap import lines (helps later sections compile standalone).
            for ln in code.splitlines():
                if re.match(rf"^\s*(?:from|import)\s+{re.escape(project)}\b", ln):
                    s = ln.strip()
                    if s and s not in bootstrap_lines:
                        bootstrap_lines.append(s)

            md_ctx = (md_by_idx.get(i) or "").strip()
            if max_markdown_chars > 0 and len(md_ctx) > max_markdown_chars:
                md_ctx = md_ctx[: max(0, max_markdown_chars - 1)].rstrip() + "…"
            code_out = code
            if max_code_chars > 0 and len(code_out) > max_code_chars:
                code_out = code_out[: max(0, max_code_chars - 1)].rstrip() + "…"

            symbols = sorted(symbols_by_project.get(project) or [])
            rel_nb = nb_path.relative_to(root)
            out_path = out_cells_root / rel_nb.with_suffix("") / f"cell_{i:04d}.md"
            _ensure_parent(out_path)
            header = (
                f"# Recipe (notebook)\n\n"
                f"Project: {project}\n"
                f"Source: {spec.name}/{rel_nb} (cell {i})\n"
                + (f"Symbols: {', '.join(symbols)}\n" if symbols else "")
                + "\n"
            )
            body = ""
            if md_ctx:
                body += "## Context (from notebook)\n\n" + md_ctx + "\n\n"
            body += "## Code\n\n```python\n" + code_out + "\n```\n"
            data = (header + body).encode("utf-8")
            out_path.write_bytes(data[:max_bytes] if len(data) > max_bytes else data)
            written_cells += 1

            # Add to current section block.
            if not block_heading:
                # Try to use the closest remembered heading.
                block_heading = current_heading
            block_cells.append((i, code))

        # Flush last block for this notebook.
        flush_block()

        # Tutorial-level recipe: overview + outline + minimal setup snippet.
        rel_nb = nb_path.relative_to(root)
        title = ""
        for cell in cells:
            if not isinstance(cell, dict) or cell.get("cell_type") != "markdown":
                continue
            src = cell.get("source")
            txt = "".join(src) if isinstance(src, list) else (src if isinstance(src, str) else "")
            hd = first_heading((txt or "").strip())
            if hd:
                title = hd
                break
        if not title:
            title = f"# {rel_nb.stem}"

        # Intro: take the first markdown cell under the title (if any).
        intro = ""
        for cell in cells:
            if not isinstance(cell, dict) or cell.get("cell_type") != "markdown":
                continue
            src = cell.get("source")
            txt = "".join(src) if isinstance(src, list) else (src if isinstance(src, str) else "")
            txt = (txt or "").strip()
            if not txt:
                continue
            if first_heading(txt) == title:
                # Keep only the text after the title heading line (skip optional MyST labels).
                lines = txt.splitlines()
                cut = 0
                for j, ln in enumerate(lines):
                    if ln.strip() == title:
                        cut = j + 1
                        break
                intro = "\n".join(lines[cut:]).strip()
                break
        if max_markdown_chars > 0 and len(intro) > max_markdown_chars:
            intro = intro[: max(0, max_markdown_chars - 1)].rstrip() + "…"

        # Setup: include project imports and the first non-import code cell that references the project.
        setup_parts: list[str] = []
        import_cell_raw = ""
        # Prefer the first *visible* import cell that imports the project, because it typically
        # also contains third-party imports (e.g. numpy) needed for later setup cells.
        for ci in sorted(cell_info.keys()):
            raw = str((cell_info.get(ci) or {}).get("raw") or "")
            if any(re.match(rf"^\s*(?:from|import)\s+{re.escape(project)}\b", ln) for ln in raw.splitlines()):
                import_cell_raw = raw.strip()
                break
        if import_cell_raw:
            setup_parts.append(import_cell_raw)
        elif bootstrap_lines:
            setup_parts.append("\n".join(bootstrap_lines).strip())
        entry_cell_code = ""
        for ci in sorted(cell_info.keys()):
            raw = str((cell_info.get(ci) or {}).get("raw") or "")
            if not raw.strip():
                continue
            # Skip pure import cells.
            if all(re.match(r"^\s*(?:from|import)\s+", ln) or not ln.strip() for ln in raw.splitlines()):
                continue
            symbols_by_project = _extract_symbols_from_python_snippet(raw, extra_aliases=notebook_aliases)
            if project in symbols_by_project:
                entry_cell_code = raw.strip()
                break
        if entry_cell_code:
            if not import_cell_raw or entry_cell_code != import_cell_raw:
                setup_parts.append(entry_cell_code)
        setup_code = "\n\n".join(p for p in setup_parts if p.strip()).strip()
        if max_code_chars > 0 and len(setup_code) > max_code_chars:
            setup_code = setup_code[: max(0, max_code_chars - 1)].rstrip() + "…"

        outline = "\n".join(f"- {b.get('heading')}" for b in blocks if str(b.get("heading") or "").strip())

        out_path = out_tutorials_root / rel_nb.with_suffix("") / "tutorial.md"
        _ensure_parent(out_path)
        header = (
            "# Tutorial (notebook)\n\n"
            f"Project: {project}\n"
            f"Notebook: {spec.name}/{rel_nb}\n"
            f"Title: {title}\n\n"
        )
        body = ""
        if intro:
            body += "## Overview\n\n" + intro.strip() + "\n\n"
        if outline.strip():
            body += "## Outline\n\n" + outline.strip() + "\n\n"
        if setup_code.strip():
            body += "## Setup (bootstrap)\n\n```python\n" + setup_code.strip() + "\n```\n\n"
        out_path.write_bytes((header + body).encode("utf-8")[:max_bytes])
        written_tutorials += 1

    return {
        "ok": True,
        "notebooks_seen": int(len(notebooks)),
        "recipes_written_cells": int(written_cells),
        "recipes_written_sections": int(written_sections),
        "recipes_written_tutorials": int(written_tutorials),
        "recipes_dir_cells": str(out_cells_root),
        "recipes_dir_sections": str(out_sections_root),
        "recipes_dir_tutorials": str(out_tutorials_root),
    }


def _write_recipes_snapshot_from_tests(
    *,
    spec: SourceSpec,
    repo_root: Path,
    dest: Path,
    max_bytes: int,
    include_private: bool,
    max_tests: int,
) -> dict[str, object]:
    """Generate 'recipe' docs from upstream tests (AST; no imports; not copied verbatim)."""

    tests_dirs = []
    for d in ("tests", "test"):
        p = repo_root / d
        if p.exists() and p.is_dir():
            tests_dirs.append(p)
    if not tests_dirs:
        return {"ok": True, "tests_seen": 0, "recipes_written": 0, "note": "No tests/ directory found."}

    out_root = dest / spec.name / "recipes" / "tests"
    out_root.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for td in tests_dirs:
        files.extend(sorted(td.rglob("test_*.py")))
        files.extend(sorted(td.rglob("*_test.py")))
    # De-dup.
    uniq: list[Path] = []
    seen = set()
    for f in files:
        if _should_skip_dir(f, DEFAULT_EXCLUDE_DIR_NAMES):
            continue
        if f in seen:
            continue
        seen.add(f)
        uniq.append(f)
    files = uniq

    recipes_written = 0
    tests_seen = 0

    for py in files:
        text = _safe_read_text(py)
        if not text:
            continue
        try:
            mod = ast.parse(text)
        except SyntaxError:
            continue

        # Collect top-of-file imports (up to first non-import statement).
        lines = text.splitlines()
        import_lines: list[str] = []
        for node in mod.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                l1, l2 = _ast_node_span(node)
                if l1 is None:
                    continue
                l2 = l2 or l1
                block = "\n".join(lines[l1 - 1 : l2]).rstrip()
                if block:
                    import_lines.append(block)
                continue
            break
        imports_block = "\n".join(import_lines).strip()

        for node in mod.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if not node.name.startswith("test"):
                continue
            tests_seen += 1
            if max_tests > 0 and tests_seen > max_tests:
                break
            l1, l2 = _ast_node_span(node)
            if l1 is None:
                continue
            l2 = l2 or l1
            body_block = "\n".join(lines[l1 - 1 : l2]).rstrip()
            if not body_block.strip():
                continue

            snippet = (imports_block + "\n\n" + body_block).strip() if imports_block else body_block
            if max_bytes > 0 and len(snippet.encode("utf-8")) > max_bytes:
                # Keep the tail of the function body visible, but clip overall.
                snippet = snippet[: max(0, max_bytes - 64)]

            symbols_by_project = _extract_symbols_from_python_snippet(snippet)
            symbols = symbols_by_project.get(spec.name) or set()
            if not symbols:
                # Skip tests that do not reference MolSysSuite explicitly.
                continue

            rel = py.relative_to(repo_root)
            out_path = out_root / rel.with_suffix("") / f"{node.name}.md"
            _ensure_parent(out_path)
            header = (
                f"# Recipe (test)\n\n"
                f"Project: {spec.name}\n"
                f"Source: {spec.name}/{rel}:{l1}-{l2}\n"
                f"Test: {node.name}\n"
                f"Symbols: {', '.join(sorted(symbols))}\n\n"
            )
            body = "## Code\n\n```python\n" + snippet.strip() + "\n```\n"
            out_path.write_bytes((header + body).encode("utf-8")[:max_bytes])
            recipes_written += 1

    return {
        "ok": True,
        "tests_seen": int(tests_seen),
        "recipes_written": int(recipes_written),
        "recipes_dir": str(out_root),
        "include_private": bool(include_private),
    }


def _extract_fenced_code_blocks(text: str) -> list[tuple[str, str]]:
    """Return a list of (language, code) fenced blocks from Markdown-like text."""
    lines = (text or "").splitlines()
    blocks: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        ln = lines[i].rstrip("\n")
        s = ln.strip()
        if not s.startswith("```"):
            i += 1
            continue
        lang = s[3:].strip().lower()
        i += 1
        buf: list[str] = []
        while i < len(lines):
            end = lines[i].strip()
            if end.startswith("```"):
                break
            buf.append(lines[i].rstrip("\n"))
            i += 1
        # Skip closing fence line if present.
        if i < len(lines) and lines[i].strip().startswith("```"):
            i += 1
        code = "\n".join(buf).rstrip()
        if code.strip():
            blocks.append((lang, code))
    return blocks


def _extract_python_examples_from_docstring(doc: str) -> list[str]:
    """Extract Python examples from a docstring (best-effort, stdlib-only).

    Supported patterns:
    - doctest prompts (>>> / ...),
    - fenced code blocks (```python ...```).
    """
    d = _normalize_docstring(doc or "")
    if not d:
        return []

    examples: list[str] = []

    # 1) Fenced blocks inside docstrings (rare but possible).
    for lang, code in _extract_fenced_code_blocks(d):
        if lang in {"python", "py", ""}:
            examples.append(code.strip())

    # 2) Doctest prompts.
    lines = d.splitlines()
    buf: list[str] = []
    in_block = False
    for ln in lines:
        s = ln.lstrip()
        if s.startswith(">>>"):
            in_block = True
            buf.append(s[4:] if s.startswith(">>> ") else s[3:].lstrip())
            continue
        if in_block and s.startswith("..."):
            buf.append(s[4:] if s.startswith("... ") else s[3:].lstrip())
            continue
        if in_block:
            # Stop at the first non-prompt line (likely output or a new section).
            if buf:
                code = "\n".join(buf).strip()
                if code:
                    examples.append(code)
            buf = []
            in_block = False
    if in_block and buf:
        code = "\n".join(buf).strip()
        if code:
            examples.append(code)

    # De-dup while preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for e in examples:
        e2 = e.strip()
        if not e2 or e2 in seen:
            continue
        seen.add(e2)
        uniq.append(e2)
    return uniq


def _write_recipes_snapshot_from_docstrings(
    *,
    spec: SourceSpec,
    repo_root: Path,
    dest: Path,
    max_bytes: int,
    include_private: bool,
) -> dict[str, object]:
    """Generate 'recipe' docs from docstring examples (AST; no imports)."""

    pkg = _find_primary_python_package(repo_root, spec.name)
    if pkg is None:
        return {"ok": False, "error": "No Python package directory found."}

    out_root = dest / spec.name / "recipes" / "docstrings"
    out_root.mkdir(parents=True, exist_ok=True)

    modules: list[Path] = []
    for p in pkg.rglob("*.py"):
        if _should_skip_dir(p, DEFAULT_EXCLUDE_DIR_NAMES):
            continue
        if not include_private and p.name.startswith("_") and p.name != "__init__.py":
            continue
        try:
            if max_bytes > 0 and p.stat().st_size > max_bytes:
                continue
        except OSError:
            continue
        modules.append(p)
    modules.sort()

    written = 0
    modules_seen = 0
    symbols_seen = 0

    for m in modules:
        text = _safe_read_text(m)
        if not text:
            continue
        try:
            mod = ast.parse(text)
        except SyntaxError:
            continue
        modules_seen += 1

        rel = m.resolve().relative_to(pkg.resolve())
        module_qual = _module_qual(pkg.name, rel.with_suffix(""))
        module_dir = _module_rel_dir(rel)

        for node in mod.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols_seen += 1
                sym_name = node.name
                sym_qual = f"{module_qual}.{sym_name}"
                doc = ast.get_docstring(node) or ""
                examples = _extract_python_examples_from_docstring(doc)
                if not examples:
                    continue
                l1, l2 = _ast_node_span(node)
                out_path = out_root / module_dir / f"{sym_name}.md"
                _ensure_parent(out_path)
                header = (
                    "# Recipe (docstring examples)\n\n"
                    f"Project: {spec.name}\n"
                    f"Symbol: {sym_qual}\n"
                    f"Module: {module_qual}\n"
                    f"Source: {m}:{l1}-{l2}\n"
                    f"Symbols: {sym_qual}\n\n"
                )
                body_parts: list[str] = []
                for j, code in enumerate(examples, start=1):
                    code = _normalize_molsysmt_alias(code) if spec.name == "molsysmt" else code
                    body_parts.append(f"## Example {j}\n\n```python\n{code.strip()}\n```")
                data = (header + "\n\n".join(body_parts).strip() + "\n").encode("utf-8")
                out_path.write_bytes(data[:max_bytes] if max_bytes > 0 and len(data) > max_bytes else data)
                written += 1
            elif isinstance(node, ast.ClassDef):
                class_name = node.name
                class_qual = f"{module_qual}.{class_name}"
                # Class docstring examples.
                symbols_seen += 1
                cdoc = ast.get_docstring(node) or ""
                cexamples = _extract_python_examples_from_docstring(cdoc)
                if cexamples:
                    l1, l2 = _ast_node_span(node)
                    out_path = out_root / module_dir / f"{class_name}.md"
                    _ensure_parent(out_path)
                    header = (
                        "# Recipe (docstring examples)\n\n"
                        f"Project: {spec.name}\n"
                        f"Symbol: {class_qual}\n"
                        f"Module: {module_qual}\n"
                        f"Source: {m}:{l1}-{l2}\n"
                        f"Symbols: {class_qual}\n\n"
                    )
                    parts: list[str] = []
                    for j, code in enumerate(cexamples, start=1):
                        code = _normalize_molsysmt_alias(code) if spec.name == "molsysmt" else code
                        parts.append(f"## Example {j}\n\n```python\n{code.strip()}\n```")
                    data = (header + "\n\n".join(parts).strip() + "\n").encode("utf-8")
                    out_path.write_bytes(data[:max_bytes] if max_bytes > 0 and len(data) > max_bytes else data)
                    written += 1

                # Method docstring examples.
                for mnode in node.body:
                    if not isinstance(mnode, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    symbols_seen += 1
                    meth_name = mnode.name
                    meth_qual = f"{class_qual}.{meth_name}"
                    mdoc = ast.get_docstring(mnode) or ""
                    mexamples = _extract_python_examples_from_docstring(mdoc)
                    if not mexamples:
                        continue
                    ml1, ml2 = _ast_node_span(mnode)
                    out_path = out_root / module_dir / class_name / f"{meth_name}.md"
                    _ensure_parent(out_path)
                    header = (
                        "# Recipe (docstring examples)\n\n"
                        f"Project: {spec.name}\n"
                        f"Symbol: {meth_qual}\n"
                        f"Module: {module_qual}\n"
                        f"Source: {m}:{ml1}-{ml2}\n"
                        f"Symbols: {meth_qual}\n\n"
                    )
                    parts: list[str] = []
                    for j, code in enumerate(mexamples, start=1):
                        code = _normalize_molsysmt_alias(code) if spec.name == "molsysmt" else code
                        parts.append(f"## Example {j}\n\n```python\n{code.strip()}\n```")
                    data = (header + "\n\n".join(parts).strip() + "\n").encode("utf-8")
                    out_path.write_bytes(data[:max_bytes] if max_bytes > 0 and len(data) > max_bytes else data)
                    written += 1

    return {
        "ok": True,
        "modules_seen": int(modules_seen),
        "symbols_seen": int(symbols_seen),
        "recipes_written": int(written),
        "recipes_dir": str(out_root),
        "include_private": bool(include_private),
    }


def _write_recipes_snapshot_from_markdown_snippets(
    *,
    spec: SourceSpec,
    dest: Path,
    max_bytes: int,
    max_blocks_per_file: int,
) -> dict[str, object]:
    """Generate 'recipe' docs from fenced code blocks in Markdown sources (snapshot)."""

    root = dest / spec.name
    if not root.exists():
        return {"ok": False, "error": "Project snapshot not found."}

    out_root = root / "recipes" / "markdown_snippets"
    out_root.mkdir(parents=True, exist_ok=True)

    md_files = sorted(root.rglob("*.md"))
    written = 0
    files_seen = 0
    blocks_seen = 0

    def is_generated(p: Path) -> bool:
        rel = p.relative_to(root).as_posix()
        return any(
            f"/{part}/" in f"/{rel}/"
            for part in ("recipes", "recipe_cards", "symbol_cards", "api_surface")
        )

    for md in md_files:
        if is_generated(md):
            continue
        text = _safe_read_text(md)
        if not text:
            continue
        files_seen += 1
        rel_md = md.relative_to(root)

        segments = _split_markdown_into_heading_segments(text)
        seg_idx = 0
        kept_in_file = 0
        for heading, seg in segments:
            seg_idx += 1
            for block_idx, (lang, code) in enumerate(_extract_fenced_code_blocks(seg), start=1):
                blocks_seen += 1
                if max_blocks_per_file > 0 and kept_in_file >= max_blocks_per_file:
                    continue
                if lang not in {"python", "py", "bash", "sh", "shell", "console"}:
                    continue
                kept_in_file += 1
                code2 = code
                if spec.name == "molsysmt" and lang in {"python", "py", ""}:
                    code2 = _normalize_molsysmt_alias(code2)

                symbols = []
                if lang in {"python", "py", ""}:
                    syms = _extract_symbols_from_python_snippet(code2).get(spec.name) or set()
                    symbols = sorted(syms)

                out_path = out_root / rel_md.with_suffix("") / f"seg_{seg_idx:04d}_block_{block_idx:04d}.md"
                _ensure_parent(out_path)
                header = (
                    "# Recipe (Markdown snippet)\n\n"
                    f"Project: {spec.name}\n"
                    f"Source: {spec.name}/{rel_md}\n"
                    + (f"Section: {heading}\n" if heading else "")
                    + (f"Language: {lang}\n" if lang else "")
                    + (f"Symbols: {', '.join(symbols)}\n" if symbols else "")
                    + "\n"
                )
                body = "## Code\n\n```" + (lang or "") + "\n" + code2.strip() + "\n```\n"
                data = (header + body).encode("utf-8")
                out_path.write_bytes(data[:max_bytes] if max_bytes > 0 and len(data) > max_bytes else data)
                written += 1

    return {
        "ok": True,
        "files_seen": int(files_seen),
        "blocks_seen": int(blocks_seen),
        "recipes_written": int(written),
        "recipes_dir": str(out_root),
        "max_blocks_per_file": int(max_blocks_per_file),
    }


def _extract_api_surface_from_module(
    py_path: Path,
    *,
    max_symbols: int,
    max_doc_chars: int,
    max_method_doc_chars: int,
) -> tuple[str, int, bool, set[str]]:
    """Return a compact Markdown 'API surface' summary for a Python module.

    Returns:
      (markdown, emitted_symbols, truncated_by_limit)

    Notes:
      - `max_symbols <= 0` means "no limit".
      - This uses AST parsing only (no imports).
    """

    text = _safe_read_text(py_path)
    if not text:
        return "", 0, False, set()
    try:
        mod = ast.parse(text)
    except SyntaxError:
        return "", 0, False, set()

    module_doc = ast.get_docstring(mod) or ""
    module_doc_1 = module_doc.strip().splitlines()[0].strip() if module_doc.strip() else ""

    lines: list[str] = []
    symbols: set[str] = set()
    lines.append(f"# API surface: `{py_path.name}`")
    if module_doc_1:
        lines.append("")
        lines.append(module_doc_1)

    lines.append("")
    lines.append("## Top-level symbols")

    n = 0
    truncated = False
    limit = max_symbols if max_symbols > 0 else None
    for node in mod.body:
        if limit is not None and n >= limit:
            truncated = True
            break
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node) or ""
            doc_1 = doc.strip().splitlines()[0].strip() if doc.strip() else ""
            sig = _format_signature(node)
            symbols.add(node.name)
            lines.append("")
            lines.append(f"### `{sig}`")
            excerpt = _doc_excerpt(doc, max_chars=max_doc_chars)
            if excerpt:
                lines.append(excerpt)
            n += 1
        elif isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node) or ""
            doc_1 = doc.strip().splitlines()[0].strip() if doc.strip() else ""
            symbols.add(node.name)
            lines.append("")
            lines.append(f"### `class {node.name}`")
            excerpt = _doc_excerpt(doc, max_chars=max_doc_chars)
            if excerpt:
                lines.append(excerpt)
            methods = [m for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
            # Record all methods in the symbol registry (best-effort), even if the
            # human-readable API surface only lists a subset.
            for m in methods:
                symbols.add(f"{node.name}.{m.name}")

            # Include a few methods (names + doc first line) to expose API names.
            for m in methods[:10]:
                if limit is not None and n >= limit:
                    truncated = True
                    break
                mdoc = ast.get_docstring(m) or ""
                mdoc_1 = mdoc.strip().splitlines()[0].strip() if mdoc.strip() else ""
                msig = _format_signature(m)
                mex = _doc_excerpt(mdoc, max_chars=max_method_doc_chars)
                if mex:
                    mex_1 = mex.splitlines()[0].strip() if mex.strip() else ""
                else:
                    mex_1 = ""
                lines.append(f"- `{node.name}.{msig}`" + (f" — {mex_1}" if mex_1 else ""))
                n += 1

    return "\n".join(lines).strip() + "\n", n, truncated, symbols


def _write_api_surface_snapshot(
    *,
    spec: SourceSpec,
    repo_root: Path,
    dest: Path,
    max_bytes: int,
    max_modules: int,
    max_symbols_per_module: int,
    include_private: bool,
    max_doc_chars: int,
    max_method_doc_chars: int,
) -> dict[str, object]:
    """Generate API-surface markdown files under the corpus snapshot.

    Output layout:
      <dest>/<project>/api_surface/<module_path>.md
    """

    pkg = _find_primary_python_package(repo_root, spec.name)
    if pkg is None:
        return {"ok": False, "error": "No Python package directory found."}

    out_root = dest / spec.name / "api_surface"
    out_root.mkdir(parents=True, exist_ok=True)

    modules_discovered = 0
    modules_skipped_private = 0
    modules_skipped_large = 0
    modules: list[Path] = []
    for p in pkg.rglob("*.py"):
        if _should_skip_dir(p, DEFAULT_EXCLUDE_DIR_NAMES):
            continue
        modules_discovered += 1
        if not include_private and p.name.startswith("_") and p.name != "__init__.py":
            modules_skipped_private += 1
            continue
        try:
            if p.stat().st_size > max_bytes:
                modules_skipped_large += 1
                continue
        except OSError:
            continue
        modules.append(p)

    modules.sort()
    modules_limited_by_max_modules = 0
    if max_modules > 0 and len(modules) > max_modules:
        modules_limited_by_max_modules = len(modules) - max_modules
    if max_modules > 0:
        modules = modules[:max_modules]

    written = 0
    symbols_written_total = 0
    modules_truncated_by_symbol_limit = 0
    symbol_registry: set[str] = set()
    for m in modules:
        rel = m.resolve().relative_to(pkg.resolve())
        md_rel = rel.with_suffix(".md")
        out_path = out_root / md_rel
        content, emitted_symbols, truncated, symbols = _extract_api_surface_from_module(
            m,
            max_symbols=max_symbols_per_module,
            max_doc_chars=max_doc_chars,
            max_method_doc_chars=max_method_doc_chars,
        )
        if not content.strip():
            continue
        # Header with fully-qualified module path for better retrieval.
        module_qual = _module_qual(pkg.name, rel.with_suffix(""))
        symbol_registry.add(module_qual)
        for s in symbols:
            symbol_registry.add(f"{module_qual}.{s}")
        pref = f"Project: {spec.name}\nModule: {module_qual}\nSource: {m}\n\n"
        data = (pref + content).encode("utf-8")
        if len(data) > max_bytes:
            # Truncate aggressively: keep header + first N chars.
            head = (pref + content[: max(0, max_bytes - len(pref) - 64)] + "\n").encode("utf-8")
            data = head[:max_bytes]
        _ensure_parent(out_path)
        out_path.write_bytes(data)
        written += 1
        symbols_written_total += int(emitted_symbols)
        if truncated:
            modules_truncated_by_symbol_limit += 1

    return {
        "ok": True,
        "package_dir": str(pkg),
        "api_surface_dir": str(out_root),
        "max_modules": int(max_modules),
        "max_symbols_per_module": int(max_symbols_per_module),
        "include_private": bool(include_private),
        "max_doc_chars": int(max_doc_chars),
        "max_method_doc_chars": int(max_method_doc_chars),
        "modules_discovered": int(modules_discovered),
        "modules_skipped_private": int(modules_skipped_private),
        "modules_skipped_large": int(modules_skipped_large),
        "modules_limited_by_max_modules": int(modules_limited_by_max_modules),
        "modules_written": written,
        "symbols_written_total": int(symbols_written_total),
        "modules_truncated_by_symbol_limit": int(modules_truncated_by_symbol_limit),
        "symbol_registry": sorted(symbol_registry),
    }


def _run_git_rev_parse(repo_dir: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return out.strip() or None


def _is_probably_text(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(8192)
    except OSError:
        return False
    if b"\x00" in head:
        return False
    return True


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_files_under(base: Path) -> Iterator[Path]:
    for p in base.rglob("*"):
        if p.is_file():
            yield p


def _should_skip_dir(path: Path, exclude_names: set[str]) -> bool:
    for part in path.parts:
        if part in exclude_names:
            return True
    return False


def _collect_source_files(
    source_root: Path,
    *,
    include_dirs: tuple[str, ...],
    include_root_globs: tuple[str, ...],
    exts: set[str],
    exclude_dir_names: set[str],
    max_bytes: int,
    max_bytes_ipynb: int,
    include_large_text: str,
) -> list[Path]:
    out: list[Path] = []

    # Root-level globs (README, etc.)
    for pat in include_root_globs:
        for p in sorted(source_root.glob(pat)):
            if not p.is_file():
                continue
            if p.suffix.lower() not in exts:
                continue
            limit = max_bytes_ipynb if p.suffix.lower() == ".ipynb" else max_bytes
            if p.stat().st_size > limit and include_large_text == "skip":
                continue
            if not _is_probably_text(p):
                continue
            out.append(p)

    # Selected directories.
    for dname in include_dirs:
        d = source_root / dname
        if not d.exists():
            continue
        for p in sorted(_iter_files_under(d)):
            if _should_skip_dir(p, exclude_dir_names):
                continue
            if p.suffix.lower() not in exts:
                continue
            try:
                limit = max_bytes_ipynb if p.suffix.lower() == ".ipynb" else max_bytes
                if p.stat().st_size > limit and include_large_text == "skip":
                    continue
            except OSError:
                continue
            if not _is_probably_text(p):
                continue
            out.append(p)

    # De-duplicate while preserving order.
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def _collect_source_files_with_stats(
    source_root: Path,
    *,
    include_dirs: tuple[str, ...],
    include_root_globs: tuple[str, ...],
    exts: set[str],
    exclude_dir_names: set[str],
    max_bytes: int,
    max_bytes_ipynb: int,
    include_large_text: str,
) -> tuple[list[Path], dict[str, object]]:
    """Like `_collect_source_files`, but also returns selection stats.

    The returned `files` list is de-duplicated and ordered, matching the snapshot selection.
    """

    stats: dict[str, object] = {
        "candidates_seen": 0,
        "selected": 0,
        "selected_by_ext": {},
        "selected_by_action": {"copy": 0, "truncate": 0, "skip_large": 0},
        "skipped_by_reason": {"excluded_dir": 0, "ext": 0, "binary": 0, "stat_error": 0},
    }
    selected: list[Path] = []
    seen: set[Path] = set()

    def add_path(p: Path) -> None:
        if not p.is_file():
            return
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp in seen:
            return
        seen.add(rp)

        stats["candidates_seen"] = int(stats.get("candidates_seen", 0)) + 1

        if _should_skip_dir(p, exclude_dir_names):
            skipped = stats["skipped_by_reason"]
            assert isinstance(skipped, dict)
            skipped["excluded_dir"] = int(skipped.get("excluded_dir", 0)) + 1
            return

        suf = p.suffix.lower()
        if suf not in exts:
            skipped = stats["skipped_by_reason"]
            assert isinstance(skipped, dict)
            skipped["ext"] = int(skipped.get("ext", 0)) + 1
            return

        if not _is_probably_text(p):
            skipped = stats["skipped_by_reason"]
            assert isinstance(skipped, dict)
            skipped["binary"] = int(skipped.get("binary", 0)) + 1
            return

        try:
            limit = max_bytes_ipynb if suf == ".ipynb" else max_bytes
            size = int(p.stat().st_size)
        except OSError:
            skipped = stats["skipped_by_reason"]
            assert isinstance(skipped, dict)
            skipped["stat_error"] = int(skipped.get("stat_error", 0)) + 1
            return

        action = "copy"
        if size > limit:
            action = "skip_large" if include_large_text == "skip" else "truncate"
        action_counts = stats["selected_by_action"]
        assert isinstance(action_counts, dict)
        action_counts[action] = int(action_counts.get(action, 0)) + 1
        if action == "skip_large":
            return

        selected.append(p)
        stats["selected"] = int(stats.get("selected", 0)) + 1
        by_ext = stats["selected_by_ext"]
        assert isinstance(by_ext, dict)
        by_ext[suf] = int(by_ext.get(suf, 0)) + 1

    # Root-level globs.
    for pat in include_root_globs:
        for p in sorted(source_root.glob(pat)):
            add_path(p)

    # Selected directories.
    for dname in include_dirs:
        d = source_root / dname
        if not d.exists():
            continue
        for p in sorted(_iter_files_under(d)):
            add_path(p)

    # Preserve original selection ordering but ensure stable sort across runs.
    try:
        selected.sort(key=lambda p: str(_rel_to_repo_root(source_root, p)))
    except Exception:
        selected.sort()

    return selected, stats


def _rel_to_repo_root(source_root: Path, p: Path) -> Path:
    try:
        return p.resolve().relative_to(source_root.resolve())
    except Exception:
        # Fallback: best effort.
        return Path(p.name)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: object) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_truncated_text_snapshot(src: Path, dest: Path, *, max_bytes: int) -> int:
    """Write a truncated UTF-8 snapshot of a text file.

    Truncation is done at the byte level, with invalid UTF-8 dropped to keep the
    snapshot readable by downstream tools.
    """
    with src.open("rb") as f:
        head = f.read(max_bytes)
    text = head.decode("utf-8", errors="ignore")
    _ensure_parent(dest)
    dest.write_text(text, encoding="utf-8")
    try:
        return int(dest.stat().st_size)
    except OSError:
        return len(text.encode("utf-8"))


def _write_ipynb_snapshot(
    src: Path,
    dest: Path,
    *,
    max_bytes: int,
    strip_outputs: bool,
    max_cells: int,
) -> dict[str, object]:
    """Write a compact notebook snapshot for indexing.

    - Drops outputs and metadata by default (reduces size drastically).
    - If still larger than `max_bytes`, truncates by keeping the first N cells.
    """
    try:
        raw = src.read_text(encoding="utf-8")
    except Exception:
        return {"ok": False, "error": "Could not read as utf-8 text."}
    try:
        nb = json.loads(raw)
    except Exception:
        return {"ok": False, "error": "Invalid JSON notebook."}

    cells = nb.get("cells") or []
    if not isinstance(cells, list):
        cells = []

    compact_cells: list[dict[str, object]] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        ctype = cell.get("cell_type")
        if ctype not in {"markdown", "code"}:
            continue
        src_field = cell.get("source")
        if not isinstance(src_field, (list, str)):
            continue
        rec: dict[str, object] = {"cell_type": ctype, "source": src_field}
        # Preserve minimal metadata needed for downstream extraction (e.g. tags like
        # "remove-input"). Keep only `metadata.tags` to avoid size blow-ups.
        meta = cell.get("metadata")
        if isinstance(meta, dict):
            tags = meta.get("tags")
            if isinstance(tags, list) and tags:
                rec["metadata"] = {"tags": [t for t in tags if isinstance(t, str) and t]}
        if not strip_outputs and ctype == "code":
            if "execution_count" in cell:
                rec["execution_count"] = cell.get("execution_count")
            if "outputs" in cell:
                rec["outputs"] = cell.get("outputs")
        compact_cells.append(rec)

    total_cells = len(compact_cells)
    if max_cells > 0:
        compact_cells = compact_cells[:max_cells]

    nb_min = {
        "cells": compact_cells,
        "metadata": {},
        "nbformat": int(nb.get("nbformat") or 4),
        "nbformat_minor": int(nb.get("nbformat_minor") or 0),
    }

    def dumps_with_cells(n: int) -> bytes:
        nb_min["cells"] = compact_cells[:n]
        return json.dumps(nb_min, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    data = dumps_with_cells(len(compact_cells))
    truncated_by_size = False
    cells_kept = len(compact_cells)

    if len(data) > max_bytes:
        # Find the maximum number of cells that fits.
        lo, hi = 1, len(compact_cells)
        best = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            blob = dumps_with_cells(mid)
            if len(blob) <= max_bytes:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        data = dumps_with_cells(best)
        truncated_by_size = True
        cells_kept = best

    _ensure_parent(dest)
    dest.write_bytes(data)

    return {
        "ok": True,
        "snapshot_bytes": int(len(data)),
        "strip_outputs": bool(strip_outputs),
        "cells_total": int(total_cells),
        "cells_kept": int(cells_kept),
        "truncated_by_size": bool(truncated_by_size),
    }


def _looks_like_myst_label(line: str) -> str | None:
    s = line.strip()
    if not (s.startswith("(") and s.endswith(")=")):
        return None
    inner = s[1:-2].strip()
    if not inner:
        return None
    # Be conservative: allow common MyST label characters used in MolSysSuite.
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.:")
    if any(ch not in allowed for ch in inner):
        return None
    return inner


def _extract_myst_labels_from_markdown(text: str) -> list[dict[str, object]]:
    labels: list[dict[str, object]] = []
    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        lab = _looks_like_myst_label(line)
        if lab:
            labels.append({"label": lab, "line": i})
    return labels


def _extract_myst_labels_from_ipynb(path: Path) -> list[dict[str, object]]:
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    cells = nb.get("cells") or []
    out: list[dict[str, object]] = []
    for ci, cell in enumerate(cells):
        if not isinstance(cell, dict):
            continue
        if cell.get("cell_type") != "markdown":
            continue
        src = cell.get("source")
        if isinstance(src, list):
            text = "".join(src)
        elif isinstance(src, str):
            text = src
        else:
            continue
        for rec in _extract_myst_labels_from_markdown(text):
            rec = dict(rec)
            rec["cell"] = ci
            out.append(rec)
    return out


def _build_docs_anchor_map(
    *,
    spec: SourceSpec,
    dest_docs_dir: Path,
    docs_base_url: str | None,
    max_bytes: int,
    max_bytes_ipynb: int,
) -> dict[str, object]:
    """Build an anchor map for one project by extracting explicit MyST labels.

    MolSysSuite documentation uses explicit MyST labels of the form:

      (Some_Label)=
      # Heading

    These labels are the stable anchor contract, so we can build deep links without
    compiling Sphinx HTML or executing `conf.py`.
    """

    if docs_base_url:
        docs_base_url = docs_base_url.rstrip("/")

    # Only anchor-map pages that come from the docs tree.
    snap_docs_root = dest_docs_dir / spec.name / "docs"
    if not snap_docs_root.exists():
        return {"ok": False, "error": "No docs/ directory in corpus snapshot."}

    pages: dict[str, object] = {}
    label_to_page: dict[str, object] = {}

    for src in sorted(snap_docs_root.rglob("*")):
        if not src.is_file():
            continue
        suf = src.suffix.lower()
        if suf not in {".md", ".ipynb"}:
            continue
        try:
            limit = max_bytes_ipynb if suf == ".ipynb" else max_bytes
            if src.stat().st_size > limit:
                continue
        except OSError:
            continue

        rel_in_docs = src.relative_to(snap_docs_root)
        doc_rel = rel_in_docs.with_suffix("")  # strip extension
        html_rel = doc_rel.as_posix() + ".html"
        docs_url = (f"{docs_base_url}/{html_rel}" if docs_base_url else None)

        if suf == ".md":
            try:
                text = src.read_text(encoding="utf-8")
            except OSError:
                continue
            labels = _extract_myst_labels_from_markdown(text)
        else:
            labels = _extract_myst_labels_from_ipynb(src)

        if not labels:
            continue

        pages[rel_in_docs.as_posix()] = {
            "html_rel": html_rel,
            "docs_url": docs_url,
            "labels": labels,
        }

        for rec in labels:
            lab = rec.get("label")
            if not isinstance(lab, str) or not lab:
                continue
            label_to_page[lab] = {
                "page": rel_in_docs.as_posix(),
                "docs_url": (f"{docs_url}#{lab}" if docs_url else None),
            }

    return {"ok": True, "docs_base_url": docs_base_url, "pages": pages, "label_index": label_to_page}


def _parse_sources(args: argparse.Namespace) -> list[SourceSpec]:
    if args.source:
        specs: list[SourceSpec] = []
        for item in args.source:
            if "=" not in item:
                raise SystemExit(f"Invalid --source value {item!r}. Expected name=/path/to/repo.")
            name, raw_path = item.split("=", 1)
            name = name.strip()
            if not name:
                raise SystemExit("Empty source name.")
            p = Path(raw_path).expanduser()
            specs.append(SourceSpec(name=name, path=p))
        return specs
    return [SourceSpec(name=k, path=v) for k, v in DEFAULT_SOURCES.items()]


def _load_corpus_config(path: Path) -> dict[str, object]:
    if not path.exists():
        raise SystemExit(f"Corpus config not found: {path}")
    suf = path.suffix.lower()
    suffixes = [s.lower() for s in path.suffixes]
    is_json = suf == ".json" or suffixes[-2:] == [".json", ".example"]
    is_toml = suf in {".toml", ".tml"} or suffixes[-2:] in ([".toml", ".example"], [".tml", ".example"])

    if is_json:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SystemExit(f"Invalid JSON corpus config: {path}: {exc}") from exc
    if is_toml:
        if tomllib is None:
            raise SystemExit("TOML corpus config requires Python 3.11+ (tomllib).")
        try:
            return tomllib.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SystemExit(f"Invalid TOML corpus config: {path}: {exc}") from exc
    raise SystemExit(f"Unsupported corpus config format: {path} (use .toml or .json)")


def _parse_str_list(value: object, *, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    raise SystemExit(f"Invalid corpus config field {field!r}: expected a string list.")


def _selection_defaults_from_config(cfg: dict[str, object]) -> SelectionSpec:
    defaults = cfg.get("defaults") or {}
    if not isinstance(defaults, dict):
        defaults = {}

    include_dirs = tuple(_parse_str_list(defaults.get("include_dirs"), field="defaults.include_dirs") or list(DEFAULT_INCLUDE_DIRS))
    include_root_globs = tuple(
        _parse_str_list(defaults.get("include_root_globs"), field="defaults.include_root_globs") or list(DEFAULT_INCLUDE_ROOT_GLOBS)
    )
    exts_raw = _parse_str_list(defaults.get("text_exts"), field="defaults.text_exts")
    exts = set(exts_raw) if exts_raw else set(DEFAULT_TEXT_EXTS)
    # Normalize extensions.
    exts = {("." + e.lstrip(".")) for e in exts}

    exclude_raw = _parse_str_list(defaults.get("exclude_dir_names"), field="defaults.exclude_dir_names")
    exclude = set(exclude_raw) if exclude_raw else set(DEFAULT_EXCLUDE_DIR_NAMES)

    return SelectionSpec(
        include_dirs=include_dirs,
        include_root_globs=include_root_globs,
        exts=exts,
        exclude_dir_names=exclude,
        docs_base_url=None,
    )


def _sources_from_config(
    cfg: dict[str, object],
    *,
    base: list[SourceSpec],
) -> list[SourceSpec]:
    sources_cfg = cfg.get("sources") or {}
    if not isinstance(sources_cfg, dict):
        return base

    by_name: dict[str, SourceSpec] = {s.name: s for s in base}

    for name, entry in sources_cfg.items():
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(entry, dict):
            continue
        enabled = entry.get("enabled")
        if enabled is False:
            by_name.pop(name, None)
            continue
        raw_path = entry.get("path")
        if raw_path is None:
            # No path override: keep existing SourceSpec (if present).
            continue
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise SystemExit(f"Invalid corpus config sources.{name}.path: expected a non-empty string.")
        by_name[name] = SourceSpec(name=name, path=Path(raw_path).expanduser())

    # Keep deterministic order: prefer DEFAULT_SOURCES order, then any extras.
    ordered: list[SourceSpec] = []
    for k in DEFAULT_SOURCES.keys():
        if k in by_name:
            ordered.append(by_name.pop(k))
    for k in sorted(by_name.keys()):
        ordered.append(by_name[k])
    return ordered


def _selection_for_source(
    *,
    cfg: dict[str, object],
    defaults: SelectionSpec,
    name: str,
    include_examples_cli: bool,
) -> SelectionSpec:
    sources_cfg = cfg.get("sources") or {}
    entry = sources_cfg.get(name) if isinstance(sources_cfg, dict) else None
    entry = entry if isinstance(entry, dict) else {}

    include_dirs = list(defaults.include_dirs)
    include_root_globs = list(defaults.include_root_globs)
    exts = set(defaults.exts)
    exclude = set(defaults.exclude_dir_names)
    docs_base_url = DEFAULT_DOCS_BASE_URL.get(name)

    # Per-source overrides.
    if "include_dirs" in entry:
        include_dirs = _parse_str_list(entry.get("include_dirs"), field=f"sources.{name}.include_dirs")
    if "include_root_globs" in entry:
        include_root_globs = _parse_str_list(entry.get("include_root_globs"), field=f"sources.{name}.include_root_globs")
    if "text_exts" in entry:
        raw = _parse_str_list(entry.get("text_exts"), field=f"sources.{name}.text_exts")
        exts = {("." + e.lstrip(".")) for e in raw} if raw else set()
    if "exclude_dir_names" in entry:
        raw = _parse_str_list(entry.get("exclude_dir_names"), field=f"sources.{name}.exclude_dir_names")
        exclude = set(raw)
    if isinstance(entry.get("docs_base_url"), str) and str(entry.get("docs_base_url")).strip():
        docs_base_url = str(entry.get("docs_base_url")).strip()

    include_examples_cfg = bool(entry.get("include_examples") is True)
    if include_examples_cli or include_examples_cfg:
        if "examples" not in include_dirs:
            include_dirs.append("examples")

    # De-duplicate while preserving order.
    uniq_dirs: list[str] = []
    seen_dirs: set[str] = set()
    for d in include_dirs:
        if not isinstance(d, str) or not d.strip():
            continue
        if d in seen_dirs:
            continue
        seen_dirs.add(d)
        uniq_dirs.append(d)

    uniq_globs: list[str] = []
    seen_globs: set[str] = set()
    for g in include_root_globs:
        if not isinstance(g, str) or not g.strip():
            continue
        if g in seen_globs:
            continue
        seen_globs.add(g)
        uniq_globs.append(g)

    return SelectionSpec(
        include_dirs=tuple(uniq_dirs),
        include_root_globs=tuple(uniq_globs),
        exts=exts,
        exclude_dir_names=exclude,
        docs_base_url=docs_base_url,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync MolSys-AI RAG corpus from sibling repos.")
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_DEST),
        help=f"Destination corpus directory (default: {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--source",
        action="append",
        help="Override sources (repeatable): name=/abs/or/rel/path",
    )
    parser.add_argument(
        "--corpus-config",
        default="",
        help=(
            "Optional corpus selection config (.toml or .json). "
            "Use this to control, per project, which directories and files are snapshotted."
        ),
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build the RAG index after syncing (requires RAG dependencies).",
    )
    parser.add_argument(
        "--build-bm25",
        action="store_true",
        help="Also build a BM25 sidecar for each index (writes `<index>.bm25.pkl`).",
    )
    parser.add_argument(
        "--build-index-parallel",
        action="store_true",
        help="Build the RAG index using multiple worker processes (optionally multi-GPU).",
    )
    parser.add_argument(
        "--build-project-indices",
        action="store_true",
        help="Also build one index per project under `--index-dir` (recommended).",
    )
    parser.add_argument(
        "--index-dir",
        default="",
        help="Directory for per-project indices (default: <repo>/server/chat_api/data/indexes).",
    )
    parser.add_argument(
        "--index-workers",
        type=int,
        default=0,
        help="Number of workers for --build-index-parallel (default: auto).",
    )
    parser.add_argument(
        "--index-devices",
        default="",
        help=(
            "Comma-separated GPU ids for --build-index-parallel (e.g. '0,1,2'). "
            "If set, workers default to len(devices) and each worker is pinned to one GPU."
        ),
    )
    parser.add_argument(
        "--index",
        default=str(DEFAULT_INDEX),
        help=f"Index output path used with --build-index (default: {DEFAULT_INDEX})",
    )
    parser.add_argument(
        "--build-anchors",
        action="store_true",
        help="Build a docs anchor map by extracting explicit MyST labels `(Label)=` from `.md` and `.ipynb` sources.",
    )
    parser.add_argument(
        "--anchors-out",
        default=str(DEFAULT_ANCHORS),
        help=f"Anchor map JSON output path used with --build-anchors (default: {DEFAULT_ANCHORS})",
    )
    parser.add_argument(
        "--build-api-surface",
        action="store_true",
        help="Generate a compact API-surface snapshot from Python docstrings/signatures (no imports).",
    )
    parser.add_argument(
        "--build-symbol-cards",
        action="store_true",
        help="Generate per-symbol 'symbol cards' from Python sources (AST; no imports).",
    )
    parser.add_argument(
        "--symbol-cards-max-modules",
        type=int,
        default=0,
        help="Max number of Python modules to scan per project for symbol cards (default: 0 = no limit).",
    )
    parser.add_argument(
        "--symbol-cards-include-private",
        action="store_true",
        help="Include private modules (leading underscore) in symbol cards.",
    )
    parser.add_argument(
        "--symbol-cards-max-doc-chars",
        type=int,
        default=0,
        help="Max docstring chars per symbol card (default: 0 = no limit).",
    )
    parser.add_argument(
        "--build-recipes",
        action="store_true",
        help="Generate derived 'recipes' from notebooks and tests (offline; no imports).",
    )
    parser.add_argument(
        "--recipes-max-code-chars",
        type=int,
        default=4000,
        help="Max code chars per notebook recipe cell (default: 4000).",
    )
    parser.add_argument(
        "--recipes-max-markdown-chars",
        type=int,
        default=2000,
        help="Max markdown context chars per notebook recipe cell (default: 2000).",
    )
    parser.add_argument(
        "--recipes-max-tests",
        type=int,
        default=0,
        help="Optional max number of tests to scan per project (default: 0 = no limit).",
    )
    parser.add_argument(
        "--recipes-max-md-blocks",
        type=int,
        default=50,
        help="Max fenced code blocks extracted per Markdown file (default: 50).",
    )
    parser.add_argument(
        "--api-surface-max-modules",
        type=int,
        default=0,
        help="Max number of Python modules to include per project (default: 0 = no limit).",
    )
    parser.add_argument(
        "--api-surface-max-symbols",
        type=int,
        default=0,
        help="Max number of symbols per module (default: 0 = no limit).",
    )
    parser.add_argument(
        "--api-surface-include-private",
        action="store_true",
        help="Include private modules (leading underscore) in the API-surface snapshot.",
    )
    parser.add_argument(
        "--api-surface-max-doc-chars",
        type=int,
        default=0,
        help="Max docstring chars per top-level symbol (default: 0 = no limit).",
    )
    parser.add_argument(
        "--api-surface-max-method-doc-chars",
        type=int,
        default=0,
        help="Max docstring chars for methods listed under classes (default: 0 = no limit).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing per-source subdirectories in the destination before syncing.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=512 * 1024,
        help="Max bytes for text files in the snapshot (default: 524288).",
    )
    parser.add_argument(
        "--max-bytes-ipynb",
        type=int,
        default=10 * 1024 * 1024,
        help="Max bytes for notebook files in the snapshot (default: 10485760).",
    )
    parser.add_argument(
        "--include-large-text",
        choices=("skip", "truncate"),
        default="truncate",
        help="What to do with files larger than the max-bytes limit (default: truncate).",
    )
    parser.add_argument(
        "--ipynb-keep-outputs",
        action="store_true",
        help="Keep notebook outputs/metadata in the snapshot (default: strip outputs).",
    )
    parser.add_argument(
        "--ipynb-max-cells",
        type=int,
        default=0,
        help="If a notebook still exceeds --max-bytes-ipynb, keep only the first N cells (default: 0 = auto-fit by size).",
    )
    parser.add_argument(
        "--include-examples",
        action="store_true",
        help="Also snapshot files under upstream `examples/` directories (default: off; these can be stale and may reinforce legacy APIs/aliases).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without writing files.",
    )
    args = parser.parse_args(argv)

    dest = Path(args.dest).expanduser()
    cfg_path = Path(str(args.corpus_config)).expanduser() if (args.corpus_config or "").strip() else None
    cfg: dict[str, object] = _load_corpus_config(cfg_path) if cfg_path else {}
    selection_defaults = _selection_defaults_from_config(cfg)

    sources_base = _parse_sources(args)
    sources = _sources_from_config(cfg, base=sources_base)
    max_bytes = int(args.max_bytes)
    max_bytes_ipynb = int(args.max_bytes_ipynb)
    include_large_text = str(args.include_large_text)
    strip_ipynb_outputs = not bool(args.ipynb_keep_outputs)
    ipynb_max_cells = int(args.ipynb_max_cells)
    index_path = Path(args.index).expanduser()
    index_dir = Path(args.index_dir).expanduser() if (args.index_dir or "").strip() else (REPO_ROOT / "server" / "chat_api" / "data" / "indexes")
    anchors_out = Path(args.anchors_out).expanduser()

    selection_by_source: dict[str, SelectionSpec] = {}
    for spec in sources:
        selection_by_source[spec.name] = _selection_for_source(
            cfg=cfg,
            defaults=selection_defaults,
            name=spec.name,
            include_examples_cli=bool(args.include_examples),
        )

    if args.clean and not args.dry_run:
        for spec in sources:
            sub = dest / spec.name
            if sub.exists():
                shutil.rmtree(sub)

    manifest: dict[str, object] = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dest": str(dest),
        "corpus_config": str(cfg_path) if cfg_path else None,
        "selection_defaults": {
            "include_dirs": list(selection_defaults.include_dirs),
            "include_root_globs": list(selection_defaults.include_root_globs),
            "text_exts": sorted(selection_defaults.exts),
            "exclude_dir_names": sorted(selection_defaults.exclude_dir_names),
        },
        "include_examples": bool(args.include_examples),
        "max_bytes": max_bytes,
        "max_bytes_ipynb": max_bytes_ipynb,
        "include_large_text": include_large_text,
        "ipynb_strip_outputs": strip_ipynb_outputs,
        "ipynb_max_cells": ipynb_max_cells,
        "sources": {},
    }

    coverage_report: dict[str, object] = {
        "generated_at_utc": manifest["generated_at_utc"],
        "dest": str(dest),
        "corpus_config": manifest["corpus_config"],
        "selection_defaults": manifest["selection_defaults"],
        "include_examples": bool(args.include_examples),
        "max_bytes": max_bytes,
        "max_bytes_ipynb": max_bytes_ipynb,
        "include_large_text": include_large_text,
        "projects": {},
    }

    copied_count = 0
    try:
        for spec in sources:
            src = spec.path
            if not src.exists():
                raise SystemExit(f"Source repo not found: {spec.name} -> {src}")

            commit = _run_git_rev_parse(src)
            sel = selection_by_source.get(spec.name) or selection_defaults
            files, selection_stats = _collect_source_files_with_stats(
                src,
                include_dirs=sel.include_dirs,
                include_root_globs=sel.include_root_globs,
                exts=sel.exts,
                exclude_dir_names=sel.exclude_dir_names,
                max_bytes=max_bytes,
                max_bytes_ipynb=max_bytes_ipynb,
                include_large_text=include_large_text,
            )

            source_info: dict[str, object] = {
                "path": str(src),
                "git_head": commit,
                "selected_file_count": len(files),
                "selection_stats": selection_stats,
                "selection": {
                    "include_dirs": list(sel.include_dirs),
                    "include_root_globs": list(sel.include_root_globs),
                    "text_exts": sorted(sel.exts),
                    "exclude_dir_names": sorted(sel.exclude_dir_names),
                    "docs_base_url": sel.docs_base_url,
                },
                "files": [],
                "stats": {
                    "written": 0,
                    "truncated_text": 0,
                    "ipynb_compacted": 0,
                    "ipynb_truncated_by_size": 0,
                    "skipped_write": 0,
                },
            }

            snapshot_bytes_total = 0
            snapshot_bytes_by_ext: dict[str, int] = {}
            for p in files:
                rel = _rel_to_repo_root(src, p)
                out_path = dest / spec.name / rel
                orig_bytes = int(p.stat().st_size)
                limit = max_bytes_ipynb if p.suffix.lower() == ".ipynb" else max_bytes
                rec = {
                    "source_relpath": str(rel),
                    "bytes": orig_bytes,
                    "sha256": _sha256(p),
                    "snapshot_bytes_limit": int(limit),
                }
                cast_files = source_info["files"]
                assert isinstance(cast_files, list)
                cast_files.append(rec)

                if args.dry_run:
                    print(f"[dry-run] {spec.name}: {rel} -> {out_path}")
                    continue

                _ensure_parent(out_path)
                if p.suffix.lower() == ".ipynb":
                    info = _write_ipynb_snapshot(
                        p,
                        out_path,
                        max_bytes=max_bytes_ipynb,
                        strip_outputs=strip_ipynb_outputs,
                        max_cells=ipynb_max_cells,
                    )
                    if not info.get("ok"):
                        # Fall back to raw copy if the notebook is small enough.
                        if orig_bytes <= max_bytes_ipynb and include_large_text != "skip":
                            shutil.copy2(p, out_path)
                            rec["snapshot_bytes"] = orig_bytes
                            rec["ipynb_snapshot_mode"] = "raw_copy"
                            snapshot_bytes_total += int(orig_bytes)
                            ext = p.suffix.lower()
                            snapshot_bytes_by_ext[ext] = int(snapshot_bytes_by_ext.get(ext, 0)) + int(orig_bytes)
                            copied_count += 1
                            stats = source_info["stats"]
                            assert isinstance(stats, dict)
                            stats["written"] = int(stats.get("written", 0)) + 1
                        else:
                            rec["skipped_write"] = True
                            rec["skip_reason"] = str(info.get("error") or "ipynb_write_failed")
                            stats = source_info["stats"]
                            assert isinstance(stats, dict)
                            stats["skipped_write"] = int(stats.get("skipped_write", 0)) + 1
                        continue
                    rec.update(info)
                    rec["ipynb_snapshot_mode"] = "compacted"
                    snap_bytes = int(info.get("snapshot_bytes") or 0)
                    snapshot_bytes_total += snap_bytes
                    ext = p.suffix.lower()
                    snapshot_bytes_by_ext[ext] = int(snapshot_bytes_by_ext.get(ext, 0)) + snap_bytes
                    copied_count += 1
                    stats = source_info["stats"]
                    assert isinstance(stats, dict)
                    stats["written"] = int(stats.get("written", 0)) + 1
                    stats["ipynb_compacted"] = int(stats.get("ipynb_compacted", 0)) + 1
                    if info.get("truncated_by_size"):
                        stats["ipynb_truncated_by_size"] = int(stats.get("ipynb_truncated_by_size", 0)) + 1
                else:
                    if orig_bytes > max_bytes and include_large_text == "truncate":
                        snap_bytes = _write_truncated_text_snapshot(p, out_path, max_bytes=max_bytes)
                        rec["snapshot_bytes"] = int(snap_bytes)
                        rec["truncated"] = True
                        snapshot_bytes_total += int(snap_bytes)
                        ext = p.suffix.lower()
                        snapshot_bytes_by_ext[ext] = int(snapshot_bytes_by_ext.get(ext, 0)) + int(snap_bytes)
                        copied_count += 1
                        stats = source_info["stats"]
                        assert isinstance(stats, dict)
                        stats["written"] = int(stats.get("written", 0)) + 1
                        stats["truncated_text"] = int(stats.get("truncated_text", 0)) + 1
                    else:
                        # Either small enough, or policy is "skip" which should have filtered it out already.
                        shutil.copy2(p, out_path)
                        rec["snapshot_bytes"] = orig_bytes
                        snapshot_bytes_total += int(orig_bytes)
                        ext = p.suffix.lower()
                        snapshot_bytes_by_ext[ext] = int(snapshot_bytes_by_ext.get(ext, 0)) + int(orig_bytes)
                        copied_count += 1
                        stats = source_info["stats"]
                        assert isinstance(stats, dict)
                        stats["written"] = int(stats.get("written", 0)) + 1

            sources_map = manifest["sources"]
            assert isinstance(sources_map, dict)
            sources_map[spec.name] = source_info
            projects = coverage_report["projects"]
            assert isinstance(projects, dict)
            projects[spec.name] = {
                "source_repo": str(src),
                "git_head": commit,
                "selected_file_count": int(len(files)),
                "selection_stats": selection_stats,
                "write_stats": source_info.get("stats"),
                "snapshot_bytes_total": int(snapshot_bytes_total),
                "snapshot_bytes_by_ext": {k: int(v) for k, v in sorted(snapshot_bytes_by_ext.items())},
            }

        if args.dry_run:
            print(f"[dry-run] would copy {copied_count} files total.")
            return 0
    except BrokenPipeError:
        # Allow piping to `head`/`less` without stack traces.
        try:
            sys.stdout.close()
        except Exception:
            pass
        return 0

    _write_json(dest / "_manifest.json", manifest)
    print(f"Copied {copied_count} files into {dest}")
    print(f"Wrote manifest: {dest / '_manifest.json'}")
    _write_json(dest / "_coverage.json", coverage_report)
    print(f"Wrote coverage report: {dest / '_coverage.json'}")
    try:
        projects = coverage_report.get("projects") or {}
        if isinstance(projects, dict) and projects:
            print("Corpus coverage summary (docs snapshot):")
            for name, info in sorted(projects.items()):
                if not isinstance(info, dict):
                    continue
                sel = (info.get("selection_stats") or {})
                if not isinstance(sel, dict):
                    continue
                selected = int(sel.get("selected") or 0)
                by_action = sel.get("selected_by_action") or {}
                if not isinstance(by_action, dict):
                    by_action = {}
                skipped_large = int(by_action.get("skip_large") or 0)
                eligible = selected + skipped_large
                pct = (100.0 * selected / eligible) if eligible else 0.0
                print(f"- {name}: selected={selected}, skipped_large={skipped_large}, coverage={pct:.1f}%")
    except Exception:
        pass

    if args.build_anchors:
        anchor_map: dict[str, object] = {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dest": str(dest),
            "projects": {},
        }
        for spec in sources:
            sel = selection_by_source.get(spec.name)
            info = _build_docs_anchor_map(
                spec=spec,
                dest_docs_dir=dest,
                docs_base_url=(sel.docs_base_url if sel else DEFAULT_DOCS_BASE_URL.get(spec.name)),
                max_bytes=max_bytes,
                max_bytes_ipynb=max_bytes_ipynb,
            )
            projects = anchor_map["projects"]
            assert isinstance(projects, dict)
            projects[spec.name] = info

            # Also record in the manifest for traceability.
            sources_map = manifest["sources"]
            assert isinstance(sources_map, dict)
            entry = sources_map.get(spec.name)
            if isinstance(entry, dict):
                entry["anchors_out"] = str(anchors_out)
                entry["docs_base_url"] = (sel.docs_base_url if sel else DEFAULT_DOCS_BASE_URL.get(spec.name))

        _write_json(anchors_out, anchor_map)
        print(f"Wrote anchors: {anchors_out}")

    if args.build_api_surface:
        api_info: dict[str, object] = {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dest": str(dest),
            "projects": {},
        }
        symbols_info: dict[str, object] = {
            "generated_at_utc": api_info["generated_at_utc"],
            "dest": str(dest),
            "projects": {},
        }
        for spec in sources:
            info = _write_api_surface_snapshot(
                spec=spec,
                repo_root=spec.path,
                dest=dest,
                max_bytes=max_bytes,
                max_modules=int(args.api_surface_max_modules),
                max_symbols_per_module=int(args.api_surface_max_symbols),
                include_private=bool(args.api_surface_include_private),
                max_doc_chars=int(args.api_surface_max_doc_chars),
                max_method_doc_chars=int(args.api_surface_max_method_doc_chars),
            )
            sym = info.pop("symbol_registry", None)
            sym_list = sym if isinstance(sym, list) else []
            info["symbol_registry_count"] = int(len(sym_list))
            projects = api_info["projects"]
            assert isinstance(projects, dict)
            projects[spec.name] = info
            sym_projects = symbols_info["projects"]
            assert isinstance(sym_projects, dict)
            sym_projects[spec.name] = {
                "ok": bool(info.get("ok")),
                "count": int(len(sym_list)),
                "symbols": sym_list,
            }
        _write_json(dest / "_api_surface.json", api_info)
        print(f"Wrote API surface manifest: {dest / '_api_surface.json'}")
        _write_json(dest / "_symbols.json", symbols_info)
        print(f"Wrote symbol registry: {dest / '_symbols.json'}")
        try:
            projects = api_info.get("projects") or {}
            if isinstance(projects, dict) and projects:
                print("Corpus coverage summary (API surface):")
                for name, info in sorted(projects.items()):
                    if not isinstance(info, dict) or not info.get("ok"):
                        continue
                    discovered = int(info.get("modules_discovered") or 0)
                    skipped_large = int(info.get("modules_skipped_large") or 0)
                    skipped_private = int(info.get("modules_skipped_private") or 0)
                    include_private = bool(info.get("include_private"))
                    written = int(info.get("modules_written") or 0)
                    eligible = discovered - skipped_large - (0 if include_private else skipped_private)
                    if eligible < 0:
                        eligible = 0
                    pct = (100.0 * written / eligible) if eligible else 0.0
                    limited = int(info.get("modules_limited_by_max_modules") or 0)
                    trunc_sym = int(info.get("modules_truncated_by_symbol_limit") or 0)
                    print(
                        f"- {name}: eligible_modules={eligible}, written={written}, "
                        f"coverage={pct:.1f}%, limited_by_max_modules={limited}, "
                        f"truncated_by_symbol_limit={trunc_sym}"
                    )
        except Exception:
            pass

    if args.build_symbol_cards:
        cards_info: dict[str, object] = {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dest": str(dest),
            "projects": {},
        }
        for spec in sources:
            info = _write_symbol_cards_snapshot(
                spec=spec,
                repo_root=spec.path,
                dest=dest,
                max_bytes=max_bytes,
                max_modules=int(args.symbol_cards_max_modules),
                include_private=bool(args.symbol_cards_include_private),
                max_doc_chars=int(args.symbol_cards_max_doc_chars),
            )
            projects = cards_info["projects"]
            assert isinstance(projects, dict)
            projects[spec.name] = info
        _write_json(dest / "_symbol_cards.json", cards_info)
        print(f"Wrote symbol cards manifest: {dest / '_symbol_cards.json'}")

    if args.build_recipes:
        recipes_info: dict[str, object] = {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dest": str(dest),
            "projects": {},
        }
        for spec in sources:
            per: dict[str, object] = {}
            per["notebooks"] = _write_recipes_snapshot_from_ipynb(
                spec=spec,
                dest=dest,
                max_bytes=max_bytes,
                max_code_chars=int(args.recipes_max_code_chars),
                max_markdown_chars=int(args.recipes_max_markdown_chars),
            )
            per["tests"] = _write_recipes_snapshot_from_tests(
                spec=spec,
                repo_root=spec.path,
                dest=dest,
                max_bytes=max_bytes,
                include_private=False,
                max_tests=int(args.recipes_max_tests),
            )
            per["docstrings"] = _write_recipes_snapshot_from_docstrings(
                spec=spec,
                repo_root=spec.path,
                dest=dest,
                max_bytes=max_bytes,
                include_private=False,
            )
            per["markdown_snippets"] = _write_recipes_snapshot_from_markdown_snippets(
                spec=spec,
                dest=dest,
                max_bytes=max_bytes,
                max_blocks_per_file=int(args.recipes_max_md_blocks),
            )
            projects = recipes_info["projects"]
            assert isinstance(projects, dict)
            projects[spec.name] = per
        _write_json(dest / "_recipes.json", recipes_info)
        print(f"Wrote recipes manifest: {dest / '_recipes.json'}")

    if args.build_index:
        # Import lazily so that the sync step is usable in minimal environments.
        try:
            from rag.build_index import build_index, build_index_parallel  # type: ignore
        except Exception as exc:
            raise SystemExit(f"Failed to import RAG index builder. Install `molsys-ai[rag]`. Error: {exc}") from exc

        if bool(args.build_bm25):
            os.environ["MOLSYS_AI_RAG_BUILD_BM25"] = "1"

        print(f"Building RAG index: {index_path}")
        if args.build_index_parallel:
            raw_devices = (args.index_devices or "").strip()
            devices = [d.strip() for d in raw_devices.split(",") if d.strip()] if raw_devices else []
            workers = int(args.index_workers or 0)
            if devices and workers <= 0:
                workers = len(devices)
            if workers <= 0:
                workers = 1
            build_index_parallel(dest, index_path, devices=devices or None, workers=workers)
        else:
            build_index(dest, index_path)

        if args.build_project_indices:
            index_dir.mkdir(parents=True, exist_ok=True)
            for spec in sources:
                proj_dir = dest / spec.name
                if not proj_dir.exists():
                    continue
                out_pkl = index_dir / f"{spec.name}.pkl"
                print(f"Building project index: {spec.name} -> {out_pkl}")
                if args.build_index_parallel and (args.index_devices or "").strip():
                    # Reuse the same sharding approach over the project subtree.
                    build_index_parallel(proj_dir, out_pkl, devices=devices or None, workers=workers)
                else:
                    build_index(proj_dir, out_pkl)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

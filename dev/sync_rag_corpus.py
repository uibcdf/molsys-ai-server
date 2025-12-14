#!/usr/bin/env python3
"""Sync a documentation corpus for MolSys-AI RAG from sibling repos.

This script snapshots selected text files from live, sibling repositories
into `server/docs_chat/data/docs/` so that the docs-chat backend can build a
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
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = REPO_ROOT / "server" / "docs_chat" / "data" / "docs"
DEFAULT_INDEX = REPO_ROOT / "server" / "docs_chat" / "data" / "rag_index.pkl"
DEFAULT_ANCHORS = REPO_ROOT / "server" / "docs_chat" / "data" / "anchors.json"

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

DEFAULT_INCLUDE_DIRS = ("docs", "doc", "examples", "devguide")
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
) -> list[Path]:
    out: list[Path] = []

    # Root-level globs (README, etc.)
    for pat in include_root_globs:
        for p in sorted(source_root.glob(pat)):
            if not p.is_file():
                continue
            if p.suffix.lower() not in exts:
                continue
            if p.stat().st_size > max_bytes:
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
                if p.stat().st_size > max_bytes:
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
            if src.stat().st_size > max_bytes:
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
        "--build-index",
        action="store_true",
        help="Build the RAG index after syncing (requires RAG dependencies).",
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
        "--clean",
        action="store_true",
        help="Remove existing per-source subdirectories in the destination before syncing.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=512 * 1024,
        help="Skip files larger than this many bytes (default: 524288).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without writing files.",
    )
    args = parser.parse_args(argv)

    dest = Path(args.dest).expanduser()
    sources = _parse_sources(args)
    max_bytes = int(args.max_bytes)
    index_path = Path(args.index).expanduser()
    anchors_out = Path(args.anchors_out).expanduser()

    if args.clean and not args.dry_run:
        for spec in sources:
            sub = dest / spec.name
            if sub.exists():
                shutil.rmtree(sub)

    manifest: dict[str, object] = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dest": str(dest),
        "max_bytes": max_bytes,
        "sources": {},
    }

    copied_count = 0
    try:
        for spec in sources:
            src = spec.path
            if not src.exists():
                raise SystemExit(f"Source repo not found: {spec.name} -> {src}")

            commit = _run_git_rev_parse(src)
            files = _collect_source_files(
                src,
                include_dirs=DEFAULT_INCLUDE_DIRS,
                include_root_globs=DEFAULT_INCLUDE_ROOT_GLOBS,
                exts=DEFAULT_TEXT_EXTS,
                exclude_dir_names=DEFAULT_EXCLUDE_DIR_NAMES,
                max_bytes=max_bytes,
            )

            source_info: dict[str, object] = {
                "path": str(src),
                "git_head": commit,
                "file_count": len(files),
                "files": [],
            }

            for p in files:
                rel = _rel_to_repo_root(src, p)
                out_path = dest / spec.name / rel
                rec = {
                    "source_relpath": str(rel),
                    "bytes": int(p.stat().st_size),
                    "sha256": _sha256(p),
                }
                cast_files = source_info["files"]
                assert isinstance(cast_files, list)
                cast_files.append(rec)

                if args.dry_run:
                    print(f"[dry-run] {spec.name}: {rel} -> {out_path}")
                    continue

                _ensure_parent(out_path)
                shutil.copy2(p, out_path)
                copied_count += 1

            sources_map = manifest["sources"]
            assert isinstance(sources_map, dict)
            sources_map[spec.name] = source_info

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

    if args.build_anchors:
        anchor_map: dict[str, object] = {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dest": str(dest),
            "projects": {},
        }
        for spec in sources:
            info = _build_docs_anchor_map(
                spec=spec,
                dest_docs_dir=dest,
                docs_base_url=DEFAULT_DOCS_BASE_URL.get(spec.name),
                max_bytes=max_bytes,
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
                entry["docs_base_url"] = DEFAULT_DOCS_BASE_URL.get(spec.name)

        _write_json(anchors_out, anchor_map)
        print(f"Wrote anchors: {anchors_out}")

    if args.build_index:
        # Import lazily so that the sync step is usable in minimal environments.
        try:
            from rag.build_index import build_index  # type: ignore
        except Exception as exc:
            raise SystemExit(f"Failed to import RAG index builder. Install `molsys-ai[rag]`. Error: {exc}") from exc

        print(f"Building RAG index: {index_path}")
        build_index(dest, index_path)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

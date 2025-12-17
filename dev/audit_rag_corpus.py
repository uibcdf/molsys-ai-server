#!/usr/bin/env python3
"""Audit MolSys-AI RAG corpus coverage.

This script is intended to answer questions like:

- How many docs/tutorial files were eligible vs included?
- How many were truncated due to size limits?
- Did API-surface extraction hit max_modules/max_symbols limits?

It reads the snapshot manifest produced by `dev/sync_rag_corpus.py` and, optionally,
re-scans the live sibling repos to estimate coverage with the same include rules.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_sync_module() -> Any:
    sync_path = REPO_ROOT / "dev" / "sync_rag_corpus.py"
    spec = importlib.util.spec_from_file_location("sync_rag_corpus", sync_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {sync_path}")
    mod = importlib.util.module_from_spec(spec)
    # dataclasses expects the module to exist in sys.modules when decorators run.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[misc]
    return mod


@dataclass(frozen=True)
class CandidateFile:
    rel: str
    ext: str
    bytes: int


def _iter_candidate_files(
    sync_mod: Any,
    source_root: Path,
    *,
    include_dirs: tuple[str, ...],
    include_root_globs: tuple[str, ...],
    exts: set[str],
    exclude_dir_names: set[str],
) -> Iterable[CandidateFile]:
    seen: set[Path] = set()

    def consider(path: Path) -> None:
        nonlocal seen
        if not path.is_file():
            return
        if path in seen:
            return
        seen.add(path)
        if path.suffix.lower() not in exts:
            return
        if sync_mod._should_skip_dir(path, exclude_dir_names):
            return
        if not sync_mod._is_probably_text(path):
            return
        try:
            size = int(path.stat().st_size)
        except OSError:
            return
        rel = str(sync_mod._rel_to_repo_root(source_root, path))
        yield CandidateFile(rel=rel, ext=path.suffix.lower(), bytes=size)

    # Root globs.
    for pat in include_root_globs:
        for p in sorted(source_root.glob(pat)):
            yield from consider(p)

    # Selected directories.
    for dname in include_dirs:
        d = source_root / dname
        if not d.exists():
            continue
        for p in sorted(sync_mod._iter_files_under(d)):
            yield from consider(p)


def _policy_action(
    c: CandidateFile,
    *,
    max_bytes: int,
    max_bytes_ipynb: int,
    include_large_text: str,
) -> str:
    limit = max_bytes_ipynb if c.ext == ".ipynb" else max_bytes
    if c.bytes <= limit:
        return "copy"
    if include_large_text == "skip":
        return "skip_large"
    return "truncate"


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return tuple(out)
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    return tuple()


def _as_str_set(value: Any) -> set[str]:
    return set(_as_str_tuple(value))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit MolSys-AI RAG corpus coverage.")
    parser.add_argument(
        "--dest",
        default=str(REPO_ROOT / "server" / "chat_api" / "data" / "docs"),
        help="Corpus snapshot directory (default: server/chat_api/data/docs).",
    )
    parser.add_argument(
        "--rescan-sources",
        action="store_true",
        help="Re-scan source repos to estimate eligible vs included coverage.",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write a JSON report.",
    )
    args = parser.parse_args(argv)

    dest = Path(args.dest).expanduser()
    manifest_path = dest / "_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path} (run dev/sync_rag_corpus.py first)")
    manifest = _load_json(manifest_path)

    max_bytes = int(manifest.get("max_bytes") or 0) or 512 * 1024
    max_bytes_ipynb = int(manifest.get("max_bytes_ipynb") or 0) or 10 * 1024 * 1024
    include_large_text = str(manifest.get("include_large_text") or "truncate")

    report: dict[str, Any] = {
        "manifest": {
            "path": str(manifest_path),
            "generated_at_utc": manifest.get("generated_at_utc"),
            "max_bytes": max_bytes,
            "max_bytes_ipynb": max_bytes_ipynb,
            "include_large_text": include_large_text,
        },
        "projects": {},
    }

    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise SystemExit("Manifest has no 'sources' map.")

    # Snapshot-side stats.
    for name, info in sources.items():
        if not isinstance(info, dict):
            continue
        files = info.get("files") or []
        if not isinstance(files, list):
            continue
        ext_counter: Counter[str] = Counter()
        trunc_text = 0
        ipynb_trunc = 0
        for rec in files:
            if not isinstance(rec, dict):
                continue
            rel = str(rec.get("source_relpath") or "")
            ext = Path(rel).suffix.lower()
            ext_counter[ext] += 1
            if rec.get("truncated") is True:
                trunc_text += 1
            if rec.get("truncated_by_size") is True:
                ipynb_trunc += 1

        proj = report["projects"].setdefault(name, {})
        proj["snapshot"] = {
            "source_repo": info.get("path"),
            "git_head": info.get("git_head"),
            "files": int(len(files)),
            "by_ext": dict(ext_counter),
            "truncated_text": int(trunc_text),
            "ipynb_truncated_by_size": int(ipynb_trunc),
            "stats": info.get("stats"),
        }

    # API-surface stats (if available).
    api_manifest_path = dest / "_api_surface.json"
    if api_manifest_path.exists():
        api_manifest = _load_json(api_manifest_path)
        api_projects = api_manifest.get("projects")
        if isinstance(api_projects, dict):
            for name, info in api_projects.items():
                proj = report["projects"].setdefault(name, {})
                proj["api_surface"] = info

    if args.rescan_sources:
        sync_mod = _load_sync_module()
        sel_defaults = manifest.get("selection_defaults") or {}
        if not isinstance(sel_defaults, dict):
            sel_defaults = {}
        for name, info in sources.items():
            if not isinstance(info, dict):
                continue
            src = Path(str(info.get("path") or "")).expanduser()
            if not src.exists():
                continue

            sel = info.get("selection") or {}
            if not isinstance(sel, dict):
                sel = {}

            include_dirs = _as_str_tuple(sel.get("include_dirs")) or _as_str_tuple(sel_defaults.get("include_dirs")) or sync_mod.DEFAULT_INCLUDE_DIRS
            include_root_globs = (
                _as_str_tuple(sel.get("include_root_globs")) or _as_str_tuple(sel_defaults.get("include_root_globs")) or sync_mod.DEFAULT_INCLUDE_ROOT_GLOBS
            )
            exts = _as_str_set(sel.get("text_exts")) or _as_str_set(sel_defaults.get("text_exts")) or sync_mod.DEFAULT_TEXT_EXTS
            exclude_dir_names = (
                _as_str_set(sel.get("exclude_dir_names")) or _as_str_set(sel_defaults.get("exclude_dir_names")) or sync_mod.DEFAULT_EXCLUDE_DIR_NAMES
            )

            candidates = list(
                _iter_candidate_files(
                    sync_mod,
                    src,
                    include_dirs=include_dirs,
                    include_root_globs=include_root_globs,
                    exts=exts,
                    exclude_dir_names=exclude_dir_names,
                )
            )
            by_ext = Counter(c.ext for c in candidates)
            by_action = Counter(
                _policy_action(
                    c,
                    max_bytes=max_bytes,
                    max_bytes_ipynb=max_bytes_ipynb,
                    include_large_text=include_large_text,
                )
                for c in candidates
            )

            snap_files = info.get("files") or []
            snap_rel = {str(r.get("source_relpath") or "") for r in snap_files if isinstance(r, dict)}
            cand_rel = {c.rel for c in candidates}
            missing_in_snapshot = sorted(cand_rel - snap_rel)

            proj = report["projects"].setdefault(name, {})
            proj["source_rescan"] = {
                "eligible_candidates": int(len(candidates)),
                "by_ext": dict(by_ext),
                "by_policy_action": dict(by_action),
                "missing_in_snapshot": {
                    "count": int(len(missing_in_snapshot)),
                    "examples": missing_in_snapshot[:25],
                },
            }

    # Human-readable output.
    print("MolSys-AI corpus audit")
    print(f"- manifest: {manifest_path}")
    print(f"- max_bytes: {max_bytes}")
    print(f"- max_bytes_ipynb: {max_bytes_ipynb}")
    print(f"- include_large_text: {include_large_text}")
    print("")

    for name, proj in sorted(report["projects"].items()):
        snap = proj.get("snapshot") or {}
        print(f"[{name}]")
        print(f"  snapshot files: {snap.get('files', 0)}")
        print(f"  snapshot by_ext: {snap.get('by_ext', {})}")
        if snap.get("truncated_text"):
            print(f"  truncated_text: {snap.get('truncated_text')}")
        if snap.get("ipynb_truncated_by_size"):
            print(f"  ipynb_truncated_by_size: {snap.get('ipynb_truncated_by_size')}")
        api = proj.get("api_surface")
        if isinstance(api, dict) and api.get("ok"):
            discovered = int(api.get("modules_discovered") or 0)
            skipped_large = int(api.get("modules_skipped_large") or 0)
            skipped_private = int(api.get("modules_skipped_private") or 0)
            include_private = bool(api.get("include_private"))
            written = int(api.get("modules_written") or 0)
            eligible = discovered - skipped_large - (0 if include_private else skipped_private)
            if eligible < 0:
                eligible = 0
            pct = (100.0 * written / eligible) if eligible else 0.0
            print(
                "  api_surface: "
                f"modules_written={api.get('modules_written')} "
                f"modules_discovered={api.get('modules_discovered')} "
                f"skipped_private={api.get('modules_skipped_private')} "
                f"skipped_large={api.get('modules_skipped_large')} "
                f"limited_by_max_modules={api.get('modules_limited_by_max_modules')} "
                f"symbol_limit_trunc={api.get('modules_truncated_by_symbol_limit')} "
                f"coverage={pct:.1f}%"
            )
        rescan = proj.get("source_rescan")
        if isinstance(rescan, dict):
            print(f"  rescan eligible: {rescan.get('eligible_candidates')}")
            print(f"  rescan by_policy_action: {rescan.get('by_policy_action')}")
            miss = rescan.get("missing_in_snapshot") or {}
            if isinstance(miss, dict) and miss.get("count"):
                print(f"  WARNING missing in snapshot: {miss.get('count')} (examples: {miss.get('examples')})")
        print("")

    if args.json_out:
        out_path = Path(args.json_out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote JSON report: {out_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

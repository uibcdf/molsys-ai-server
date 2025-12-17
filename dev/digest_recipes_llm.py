#!/usr/bin/env python3
"""Digest extracted recipes into compact "recipe cards" using a local LLM engine.

This is an *offline/batch* quality tool. It is intentionally NOT part of the default
serving path.

Typical workflow:
  1) Build the corpus snapshot + recipes:
     python dev/sync_rag_corpus.py --clean --build-recipes ...
  2) Start the engine (`server/model_server`) locally.
  3) Run this script to generate derived recipe cards:
     python dev/digest_recipes_llm.py --docs-dir server/chat_api/data/docs

Output is written under:
  <docs-dir>/<project>/recipe_cards/...

These derived files are designed to be indexed alongside raw docs/recipes.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
CLIENT_DIR = REPO_ROOT / "client"
if str(CLIENT_DIR) not in sys.path:
    sys.path.insert(0, str(CLIENT_DIR))

from agent.model_client import HTTPModelClient  # noqa: E402


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_symbol_registry(path: Path) -> dict[str, tuple[set[str], set[str]]]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    projects = obj.get("projects")
    if not isinstance(projects, dict):
        return {}
    out: dict[str, tuple[set[str], set[str]]] = {}
    for name, info in projects.items():
        if not isinstance(info, dict):
            continue
        syms = info.get("symbols")
        if not isinstance(syms, list):
            continue
        symbols = {str(s) for s in syms if isinstance(s, str) and s}
        prefixes: set[str] = set()
        for s in symbols:
            parts = s.split(".")
            for i in range(1, len(parts)):
                prefixes.add(".".join(parts[:i]))
        out[str(name)] = (symbols, prefixes)
    return out


def _extract_import_aliases(text: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for project in ("molsysmt", "molsysviewer", "pyunitwizard", "topomt"):
        for a in re.findall(rf"^\s*import\s+{project}\s+as\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", text, flags=re.M):
            aliases[a] = project
    return aliases


def _extract_candidate_symbols(text: str) -> set[str]:
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


def _unknown_symbols(text: str, *, registry: dict[str, tuple[set[str], set[str]]]) -> list[str]:
    aliases = _extract_import_aliases(text)
    bad: set[str] = set()
    for sym in _extract_candidate_symbols(text):
        resolved = _resolve_alias_symbol(sym, aliases)
        project = resolved.split(".", 1)[0]
        if project not in registry:
            continue
        symbols, prefixes = registry.get(project) or (set(), set())
        if resolved in symbols or resolved in prefixes:
            continue
        bad.add(resolved)
    return sorted(bad)


def _build_prompt(*, raw_recipe: str, project: str, want_sources: bool) -> list[dict[str, str]]:
    system = (
        "You are MolSys-AI.\n"
        "Rewrite the raw recipe into a compact 'recipe card' for the MolSysSuite ecosystem.\n"
        "- Use the term 'MolSysSuite' (never 'MolSys*').\n"
        "- Do NOT invent API names.\n"
        "- Prefer short, runnable snippets.\n"
        "- Keep it concise.\n"
        "- When writing MolSysMT Python snippets, use: `import molsysmt as msm`.\n"
        "Return ONLY Markdown.\n"
    )
    if want_sources:
        system += "- Keep bracket citations like [1] if present.\n"
    user = (
        f"Project hint: {project}\n\n"
        "Raw extracted recipe:\n\n"
        f"{raw_recipe.strip()}\n\n"
        "Rewrite as a recipe card with sections:\n"
        "## Goal\n"
        "## Minimal snippet\n"
        "## Notes\n"
        "## Related symbols\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Digest extracted recipes into compact recipe cards using the local engine.")
    p.add_argument("--docs-dir", default="server/chat_api/data/docs", help="Docs snapshot dir (default: server/chat_api/data/docs).")
    p.add_argument("--symbols", default="", help="Path to _symbols.json (default: <docs-dir>/_symbols.json).")
    p.add_argument("--engine-url", default="http://127.0.0.1:8001", help="Engine base URL (default: http://127.0.0.1:8001).")
    p.add_argument("--engine-api-key", default="", help="Optional engine API key (Authorization: Bearer ...).")
    p.add_argument("--limit", type=int, default=0, help="Optional max number of recipes to digest.")
    p.add_argument(
        "--include-cell-recipes",
        action="store_true",
        help="Also digest per-cell notebook recipes under recipes/notebooks/ (default: only notebooks_sections + tests).",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N processed recipes (default: 10; set 0 to disable).",
    )
    p.add_argument(
        "--no-purge",
        action="store_true",
        help="Do not delete existing <project>/recipe_cards/ before generating new ones (default: purge).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing recipe_cards outputs (only relevant with --no-purge).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print what would be generated without writing.")
    args = p.parse_args(argv)

    docs_dir = Path(args.docs_dir).resolve()
    if not docs_dir.exists():
        raise SystemExit(f"Docs dir not found: {docs_dir}")

    if not args.no_purge:
        # Purge by default to avoid stale recipe cards lingering after upstream docs/recipes move or disappear.
        for project in ("molsysmt", "molsysviewer", "pyunitwizard", "topomt"):
            out_root = (docs_dir / project / "recipe_cards").resolve()
            try:
                out_root.relative_to(docs_dir)
            except Exception:
                continue
            if out_root.name != "recipe_cards":
                continue
            if out_root.exists():
                if args.dry_run:
                    print(f"[dry-run] purge: {out_root}")
                else:
                    shutil.rmtree(out_root)
        if not args.dry_run:
            print("Purged existing recipe_cards/ (default behavior).")
    symbols_path = Path(args.symbols).resolve() if (args.symbols or "").strip() else (docs_dir / "_symbols.json")
    registry = _load_symbol_registry(symbols_path)

    engine_url = str(args.engine_url).rstrip("/")
    api_key = (args.engine_api_key or "").strip() or None
    client = HTTPModelClient(base_url=engine_url, api_key=api_key)

    recipe_paths: list[Path] = []
    for project in ("molsysmt", "molsysviewer", "pyunitwizard", "topomt"):
        recipe_paths.extend(sorted((docs_dir / project / "recipes" / "notebooks_tutorials").rglob("*.md")))
        recipe_paths.extend(sorted((docs_dir / project / "recipes" / "notebooks_sections").rglob("*.md")))
        recipe_paths.extend(sorted((docs_dir / project / "recipes" / "tests").rglob("*.md")))
        if args.include_cell_recipes:
            recipe_paths.extend(sorted((docs_dir / project / "recipes" / "notebooks").rglob("*.md")))

    # Filter out already-generated outputs unless overwrite is requested.
    planned: list[tuple[Path, Path]] = []
    for src in recipe_paths:
        try:
            rel = src.relative_to(docs_dir)
        except Exception:
            continue
        if not rel.parts:
            continue
        project = rel.parts[0]
        out = docs_dir / project / "recipe_cards" / Path(*rel.parts[2:])  # drop "<project>/recipes/"
        if out.exists() and not args.overwrite:
            continue
        planned.append((src, out))

    if args.limit and args.limit > 0:
        planned = planned[: int(args.limit)]

    total = len(planned)
    started = time.perf_counter()
    progress_every = max(int(args.progress_every), 0)
    print(f"Planned recipe digests: {total}")

    n_written = 0
    processed = 0
    for src, out in planned:
        rel = src.relative_to(docs_dir)
        project = rel.parts[0]

        raw = src.read_text(encoding="utf-8", errors="replace")
        messages = _build_prompt(raw_recipe=raw, project=project, want_sources=False)
        card = (client.generate(messages, generation={"temperature": 0.0, "top_p": 1.0, "max_tokens": 900}) or "").strip()
        if not card:
            processed += 1
            continue

        bad = _unknown_symbols(card, registry=registry)
        if bad:
            # One retry with a strict constraint.
            fix_system = (
                "You are MolSys-AI.\n"
                "Rewrite the recipe card to REMOVE or REPLACE invalid API symbols.\n"
                "- Use the term 'MolSysSuite' (never 'MolSys*').\n"
                "The following API symbols MUST NOT appear:\n"
                + "\n".join(f"- {s}" for s in bad)
                + "\n\nReturn ONLY Markdown.\n"
            )
            fix_user = "Draft recipe card:\n\n" + card
            card2 = (client.generate([{"role": "system", "content": fix_system}, {"role": "user", "content": fix_user}], generation={"temperature": 0.0, "top_p": 1.0, "max_tokens": 900}) or "").strip()
            if card2:
                card = card2

        header = (
            f"<!-- Generated at {_now_utc()} from {rel.as_posix()} using engine {engine_url} -->\n\n"
        )
        final = header + card.strip() + "\n"

        if args.dry_run:
            print(f"[dry-run] {rel.as_posix()} -> {out.relative_to(docs_dir).as_posix()}")
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(final, encoding="utf-8")
            n_written += 1
        processed += 1

        if progress_every and (processed % progress_every == 0 or processed == total):
            elapsed = time.perf_counter() - started
            rate = (processed / elapsed) if elapsed > 0 else 0.0
            remaining = total - processed
            eta_s = (remaining / rate) if rate > 0 else 0.0
            pct = (100.0 * processed / total) if total else 100.0
            print(
                f"Progress: {processed}/{total} ({pct:.1f}%) | "
                f"written={n_written} | rate={rate:.2f}/s | ETA={eta_s/60:.1f} min"
            )

    print(f"Wrote {n_written} recipe cards under {docs_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit notebook-derived recipes for a given notebook path.

This script inspects the *generated* artifacts under the docs snapshot:

  <docs-dir>/<project>/recipes/notebooks_tutorials/<nb_rel_without_ext>/tutorial.md
  <docs-dir>/<project>/recipes/notebooks_sections/<nb_rel_without_ext>/section_*.md
  <docs-dir>/<project>/recipes/notebooks/<nb_rel_without_ext>/cell_*.md

It is designed as a quick sanity check for the tutorial→section→cell extraction
and the section stitching strategy (imports/defs/setup cells).

No third-party dependencies (stdlib-only).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def _rel_no_ext(p: Path) -> Path:
    return p.with_suffix("")


def _read_head(path: Path, n_lines: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[:n_lines]).rstrip()


def _extract_header_field(text: str, key: str) -> str | None:
    for ln in (text or "").splitlines():
        if ln.startswith(key):
            return ln[len(key) :].strip()
    return None


def _has_import_in_code(text: str) -> bool:
    in_code = False
    for ln in (text or "").splitlines():
        s = ln.rstrip()
        if s.strip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code:
            continue
        if re.search(r"^\s*(import|from)\s+\w+", s):
            return True
    return False


def _list_sorted(glob_root: Path, pattern: str) -> list[Path]:
    return sorted(glob_root.glob(pattern))


def main() -> int:
    p = argparse.ArgumentParser(description="Audit notebook-derived recipes for tutorial/section/cell extraction.")
    p.add_argument(
        "--docs-dir",
        default="server/chat_api/data/docs",
        help="Docs snapshot dir (default: server/chat_api/data/docs).",
    )
    p.add_argument(
        "--notebook",
        action="append",
        default=[],
        help="Notebook path relative to docs-dir (repeatable), e.g. molsysmt/docs/content/.../select.ipynb",
    )
    p.add_argument("--head", type=int, default=80, help="How many lines to show from tutorial/section headers (default: 80).")
    args = p.parse_args()

    docs_dir = Path(args.docs_dir).resolve()
    if not docs_dir.exists():
        raise SystemExit(f"Docs dir not found: {docs_dir}")

    notebooks = [Path(nb) for nb in args.notebook]
    if not notebooks:
        raise SystemExit("Provide at least one --notebook path (relative to --docs-dir).")

    for nb_rel in notebooks:
        nb_path = (docs_dir / nb_rel).resolve()
        print()
        print("#" * 80)
        print()
        print(f"Notebook: {nb_rel.as_posix()}")
        print(f"Exists:   {nb_path.exists()}")
        if not nb_path.exists():
            continue
        try:
            nb_rel_to_docs = nb_path.relative_to(docs_dir)
        except Exception:
            nb_rel_to_docs = nb_rel
        if len(nb_rel_to_docs.parts) < 2:
            print("ERROR: notebook path must include <project>/... under docs-dir.")
            continue
        project = nb_rel_to_docs.parts[0]
        nb_without_ext = _rel_no_ext(nb_rel_to_docs)

        tut = docs_dir / project / "recipes" / "notebooks_tutorials" / nb_without_ext.relative_to(project) / "tutorial.md"
        sec_root = docs_dir / project / "recipes" / "notebooks_sections" / nb_without_ext.relative_to(project)
        cell_root = docs_dir / project / "recipes" / "notebooks" / nb_without_ext.relative_to(project)

        print(f"Project:  {project}")
        print(f"Tutorial: {tut.relative_to(docs_dir) if tut.exists() else '(missing)'}")
        if tut.exists():
            print()
            print("[tutorial head]")
            print(_read_head(tut, int(args.head)) or "(empty)")

        sections = _list_sorted(sec_root, "section_*.md") if sec_root.exists() else []
        cells = _list_sorted(cell_root, "cell_*.md") if cell_root.exists() else []
        print()
        print(f"Sections: {len(sections)} ({sec_root.relative_to(docs_dir) if sec_root.exists() else sec_root})")
        print(f"Cells:    {len(cells)} ({cell_root.relative_to(docs_dir) if cell_root.exists() else cell_root})")

        if sections:
            missing_import = 0
            for s in sections[: min(5, len(sections))]:
                txt = _read_head(s, 400)
                cells_field = _extract_header_field(txt, "Cells:")
                setup_field = _extract_header_field(txt, "Setup cells:")
                symbols_field = _extract_header_field(txt, "Symbols:")
                has_import = _has_import_in_code(txt)
                if not has_import:
                    missing_import += 1
                rel_s = s.relative_to(docs_dir)
                print()
                print(f"- {rel_s} (cells={cells_field or '?'}, setup={setup_field or '-'}, has_import={has_import})")
                if symbols_field:
                    print(f"  symbols: {symbols_field}")
            if missing_import:
                print()
                print(f"WARNING: {missing_import}/{min(5, len(sections))} sampled sections contain no import statements in their code blocks.")

        if cells:
            print()
            print("Sample cell recipes:")
            for c in cells[: min(3, len(cells))]:
                rel_c = c.relative_to(docs_dir)
                print(f"- {rel_c}")

    print()
    print("#" * 80)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


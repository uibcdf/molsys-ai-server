#!/usr/bin/env python3
"""Pretty viewer for benchmark run JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON at {path}:{i}: {exc}") from exc
        if not isinstance(obj, dict):
            continue
        rows.append(obj)
    return rows


def _short(text: str, width: int) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= width:
        return t
    return t[: max(0, width - 1)] + "…"


def _print_row(row: dict[str, Any], *, full: bool, width: int) -> None:
    rid = row.get("id")
    status = (row.get("http") or {}).get("status")
    elapsed = (row.get("http") or {}).get("elapsed_s")
    ok = ((row.get("eval") or {}).get("ok") is True)
    tag = "OK" if ok else "FAIL"
    print(f"{tag} {rid} (HTTP {status}, {elapsed}s)")

    query = row.get("query") or ""
    print("Q:", query)

    ans = ((row.get("response") or {}).get("answer") or "")
    if full:
        print("\nA:\n" + ans.rstrip() + "\n")
    else:
        print("A:", _short(ans, width))

    sources = (row.get("response") or {}).get("sources")
    if isinstance(sources, list):
        print(f"Sources: {len(sources)}")
        for s in sources[:5]:
            if not isinstance(s, dict):
                continue
            sid = s.get("id")
            path = s.get("path")
            url = s.get("url")
            if full:
                print(f"  [{sid}] {path}")
                if url:
                    print(f"      {url}")
            else:
                print(f"  [{sid}] {_short(str(path or ''), width)}")
        if len(sources) > 5 and not full:
            print("  …")
    else:
        print("Sources: -")

    if not ok:
        checks = (row.get("eval") or {}).get("checks") or {}
        bad = []
        for name, v in checks.items():
            if isinstance(v, dict) and v.get("ok") is False:
                bad.append(f"{name}: {v.get('detail')}")
        if bad:
            print("Failed checks:")
            for b in bad:
                print("  -", b)

    print()
    print("#" * 80)
    print("#" * 80)
    print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pretty-print a MolSys-AI chat benchmark run JSONL.")
    p.add_argument("run", help="Path to a run JSONL under dev/benchmarks/runs/")
    p.add_argument("--only-fail", action="store_true", help="Show only failed questions.")
    p.add_argument("--ids", default="", help="Comma-separated list of question ids to show.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--short", action="store_true", help="Show a single-line answer preview and short source paths.")
    mode.add_argument("--full", action="store_true", help="Show full answers and source URLs (default).")
    p.add_argument("--width", type=int, default=140, help="Line width for --short mode.")
    args = p.parse_args(argv)

    run_path = Path(args.run)
    rows = _read_jsonl(run_path)
    # First row may be meta.
    data = [r for r in rows if "id" in r]

    ids = [s.strip() for s in (args.ids or "").split(",") if s.strip()]
    if ids:
        data = [r for r in data if str(r.get("id")) in set(ids)]

    if args.only_fail:
        data = [r for r in data if not ((r.get("eval") or {}).get("ok") is True)]

    if not data:
        print("No matching rows.")
        return 0

    for r in data:
        _print_row(r, full=(not bool(args.short)), width=int(args.width))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

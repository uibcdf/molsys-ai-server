#!/usr/bin/env python3
"""Sweep retrieval weights (BM25/hybrid) using per-request rag_config overrides.

This requires the chat API to be started with:
  MOLSYS_AI_CHAT_ALLOW_RAG_CONFIG=1

Optionally enable response debug:
  MOLSYS_AI_CHAT_ALLOW_DEBUG=1

It runs the benchmark question set multiple times, changing only rag_config,
and prints a small summary table.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        items.append(json.loads(s))
    return items


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_one(
    *,
    base_url: str,
    in_path: Path,
    api_key: str | None,
    bm25_weight: float,
    hybrid_weight: float,
    strict_symbols: bool,
    timeout_s: int,
) -> tuple[Path, int, int, float]:
    base_items = _read_jsonl(in_path)
    # Copy items, inject rag_config for all actual questions (skip _meta if present).
    rows: list[dict[str, Any]] = []
    for obj in base_items:
        if "_meta" in obj:
            rows.append(obj)
            continue
        o2 = dict(obj)
        o2["rag_config"] = {"bm25_weight": bm25_weight, "hybrid_weight": hybrid_weight}
        o2["debug"] = True
        rows.append(o2)

    with tempfile.TemporaryDirectory(prefix="molsys_ai_sweep_") as td:
        tmp_in = Path(td) / "questions.jsonl"
        _write_jsonl(tmp_in, rows)
        out_dir = REPO_ROOT / "dev" / "benchmarks" / "runs"
        out_path = out_dir / f"sweep_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_bm25{bm25_weight:.2f}_hyb{hybrid_weight:.2f}.jsonl"

        cmd = [
            sys.executable,
            str(REPO_ROOT / "dev" / "benchmarks" / "run_chat_bench.py"),
            "--base-url",
            base_url,
            "--in",
            str(tmp_in),
            "--out",
            str(out_path),
            "--timeout",
            str(timeout_s),
            "--check-symbols",
        ]
        if strict_symbols:
            cmd.append("--strict-symbols")
        if api_key:
            cmd.extend(["--api-key", api_key])

        subprocess.check_call(cmd)

    # Parse summary from output file.
    passed = 0
    total = 0
    elapsed_total = 0.0
    for row in _read_jsonl(out_path):
        if "_meta" in row:
            continue
        total += 1
        http = row.get("http") or {}
        if isinstance(http, dict):
            elapsed_total += float(http.get("elapsed_s") or 0.0)
        ev = row.get("eval") or {}
        if isinstance(ev, dict) and ev.get("ok") is True:
            passed += 1
    mean_s = (elapsed_total / total) if total else 0.0
    return out_path, passed, total, mean_s


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Sweep retrieval weights via rag_config overrides.")
    p.add_argument("--base-url", default="http://127.0.0.1:8000", help="Chat API base URL.")
    p.add_argument("--in", dest="in_path", default="dev/benchmarks/questions_v0.jsonl", help="Input questions JSONL.")
    p.add_argument("--api-key", default="", help="Optional /v1/chat API key.")
    p.add_argument("--timeout", type=int, default=1800, help="Per-request timeout seconds.")
    p.add_argument("--strict-symbols", action="store_true", help="Fail unknown symbols even if NOT_DOCUMENTED is present.")
    p.add_argument(
        "--bm25",
        default="0,0.15,0.25,0.35",
        help="Comma-separated bm25_weight values to test (default: 0,0.15,0.25,0.35).",
    )
    p.add_argument(
        "--hybrid",
        default="0.15",
        help="Comma-separated hybrid_weight values to test (default: 0.15).",
    )
    args = p.parse_args(argv)

    in_path = (REPO_ROOT / args.in_path).resolve() if not Path(args.in_path).is_absolute() else Path(args.in_path)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    bm25_vals = [float(x.strip()) for x in str(args.bm25).split(",") if x.strip()]
    hyb_vals = [float(x.strip()) for x in str(args.hybrid).split(",") if x.strip()]
    api_key = (args.api_key or "").strip() or None

    print("bm25_weight  hybrid_weight  passed  total  mean_s  run_file")
    for hyb in hyb_vals:
        for bm in bm25_vals:
            out_path, passed, total, mean_s = _run_one(
                base_url=str(args.base_url),
                in_path=in_path,
                api_key=api_key,
                bm25_weight=bm,
                hybrid_weight=hyb,
                strict_symbols=bool(args.strict_symbols),
                timeout_s=int(args.timeout),
            )
            print(f"{bm:9.2f}  {hyb:12.2f}  {passed:6d}  {total:5d}  {mean_s:6.2f}  {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


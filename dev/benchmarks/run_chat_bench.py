#!/usr/bin/env python3
"""Minimal chat quality benchmark runner for MolSys-AI.

This runner executes a JSONL question set against the public chat API:

  POST /v1/chat

and stores responses + optional lightweight checks (citations/sources/keywords)
as JSONL so results can be compared across corpus/prompt/model iterations.

Design goals:
- zero non-stdlib dependencies,
- explicit, reproducible inputs/outputs,
- small enough to run ad-hoc on the GPU host.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _now_utc_ts() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSONL at {path}:{i}: {exc}") from exc
        if not isinstance(obj, dict):
            raise SystemExit(f"Invalid JSONL at {path}:{i}: expected object/dict.")
        items.append(obj)
    return items


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _coerce_list_of_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return value
    return []

def _load_symbol_registry(path: Path) -> dict[str, tuple[set[str], set[str]]]:
    """Load `_symbols.json` into {project: (symbols_set, prefixes_set)}.

    Prefixes allow validating mentions like `molsysmt.structure` even when the
    registry contains only deeper symbols such as `molsysmt.structure.get_rmsd`.
    """
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
        if isinstance(syms, list):
            symbols = {str(s) for s in syms if isinstance(s, str) and s}
            prefixes: set[str] = set()
            for sym in symbols:
                parts = sym.split(".")
                for i in range(1, len(parts)):
                    prefixes.add(".".join(parts[:i]))
            out[str(name)] = (symbols, prefixes)
    return out


def _extract_import_aliases(text: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for project in ("molsysmt", "molsysviewer", "pyunitwizard", "topomt"):
        for a in re.findall(rf"^\s*import\s+{project}\s+as\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", text, flags=re.M):
            aliases[a] = project
        for a in re.findall(
            rf"^\s*from\s+{project}\s+import\s+.+?\s+as\s+([A-Za-z_][A-Za-z0-9_]*)\s*$",
            text,
            flags=re.M,
        ):
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


def _http_post_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str | None,
    timeout_s: int,
) -> tuple[int, dict[str, Any] | None, str]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                obj = None
            return int(resp.status), obj, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            obj = None
        return int(e.code), obj, raw
    except urllib.error.URLError as e:
        return 0, None, f"URLError: {e}"


@dataclass(frozen=True)
class Checks:
    want_sources: bool | None = None
    contains: list[str] = None  # type: ignore[assignment]
    contains_any: list[str] = None  # type: ignore[assignment]
    not_contains: list[str] = None  # type: ignore[assignment]
    symbols_must_exist: bool | None = None
    symbols_strict: bool | None = None


def _evaluate(
    answer: str,
    sources: Any,
    checks: Checks,
    *,
    symbols: dict[str, tuple[set[str], set[str]]],
) -> dict[str, Any]:
    ans = answer or ""
    ans_lower = ans.lower()
    out: dict[str, Any] = {"ok": True, "checks": {}}

    def fail(name: str, detail: str) -> None:
        out["ok"] = False
        out["checks"][name] = {"ok": False, "detail": detail}

    def ok(name: str, detail: str = "") -> None:
        out["checks"][name] = {"ok": True, "detail": detail}

    if checks.want_sources is True:
        if not re.search(r"\[\d+\]", ans):
            fail("citations", "Expected bracketed citations like [1].")
        else:
            ok("citations")
        if not isinstance(sources, list) or len(sources) < 1:
            fail("sources", "Expected non-empty sources list.")
        else:
            ok("sources", f"{len(sources)} sources")
    elif checks.want_sources is False:
        if re.search(r"\[\d+\]", ans):
            fail("citations", "Did not expect bracketed citations like [1].")
        else:
            ok("citations")
        if isinstance(sources, list) and len(sources) > 0:
            fail("sources", "Did not expect sources.")
        else:
            ok("sources")

    if checks.contains:
        for needle in checks.contains:
            if needle.lower() not in ans_lower:
                fail("contains", f"Missing expected substring: {needle!r}")
                break
        else:
            ok("contains")

    if checks.contains_any:
        if not any(needle.lower() in ans_lower for needle in checks.contains_any):
            fail("contains_any", f"Missing any of: {checks.contains_any!r}")
        else:
            ok("contains_any")

    for needle in checks.not_contains or []:
        if needle.lower() in ans_lower:
            fail("not_contains", f"Found forbidden substring: {needle!r}")
            break
    else:
        if checks.not_contains:
            ok("not_contains")

    if checks.symbols_must_exist is True and symbols:
        aliases = _extract_import_aliases(ans)
        bad: list[str] = []
        for sym in sorted(_extract_candidate_symbols(ans)):
            resolved = _resolve_alias_symbol(sym, aliases)
            project = resolved.split(".", 1)[0]
            if project not in {"molsysmt", "molsysviewer", "pyunitwizard", "topomt"}:
                continue
            reg_syms, reg_prefixes = symbols.get(project) or (set(), set())
            if (reg_syms or reg_prefixes) and (resolved not in reg_syms and resolved not in reg_prefixes):
                bad.append(resolved)
        strict = (checks.symbols_strict is True)
        if bad and (strict or "NOT_DOCUMENTED" not in ans):
            fail("symbols", f"Unknown tool symbols: {bad[:10]!r}" + (" (truncated)" if len(bad) > 10 else ""))
        else:
            ok("symbols")

    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run a minimal chat quality benchmark against /v1/chat.")
    p.add_argument("--base-url", default="http://127.0.0.1:8000", help="Chat API base URL (default: http://127.0.0.1:8000)")
    p.add_argument("--api-key", default="", help="Optional API key for POST /v1/chat (Authorization: Bearer ...)")
    p.add_argument("--in", dest="in_path", required=True, help="Input JSONL file with questions.")
    p.add_argument("--out", default="", help="Output JSONL path. Default: dev/benchmarks/runs/<ts>.jsonl")
    p.add_argument("--timeout", type=int, default=1800, help="Per-request timeout seconds (default: 1800)")
    p.add_argument("--k", type=int, default=5, help="Default k for retrieval (default: 5)")
    p.add_argument("--client", default="cli", choices=["cli", "widget"], help="Default client kind (default: cli)")
    p.add_argument("--rag", default="auto", choices=["on", "off", "auto"], help="Default rag mode (default: auto)")
    p.add_argument(
        "--sources",
        default="auto",
        choices=["on", "off", "auto"],
        help="Default sources mode (default: auto; widget usually wants on)",
    )
    p.add_argument("--limit", type=int, default=0, help="Optional max number of questions to run.")
    p.add_argument(
        "--symbols",
        default="server/chat_api/data/docs/_symbols.json",
        help="Path to symbol registry JSON (default: server/chat_api/data/docs/_symbols.json).",
    )
    p.add_argument(
        "--check-symbols",
        action="store_true",
        help="Enable semantic check: fail when answers mention unknown MolSysSuite API symbols (best-effort).",
    )
    p.add_argument(
        "--strict-symbols",
        action="store_true",
        help="When used with --check-symbols, also fail if the answer contains `NOT_DOCUMENTED` but still mentions unknown symbols.",
    )
    args = p.parse_args(argv)

    base_url = args.base_url.rstrip("/")
    url = f"{base_url}/v1/chat"
    api_key = (args.api_key or "").strip() or None

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")
    items = _read_jsonl(in_path)
    if args.limit and args.limit > 0:
        items = items[: int(args.limit)]

    out_path = Path(args.out) if args.out else (Path(__file__).resolve().parent / "runs" / f"run_{_now_utc_ts()}.jsonl")
    symbols = _load_symbol_registry(Path(args.symbols)) if args.check_symbols else {}

    run_meta = {
        "type": "molsys-ai-chat-bench",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "url": url,
        "defaults": {
            "k": int(args.k),
            "client": args.client,
            "rag": args.rag,
            "sources": args.sources,
        },
        "input": str(in_path),
        "symbols_check_enabled": bool(args.check_symbols),
        "symbols_strict": bool(args.strict_symbols),
        "symbols_path": str(args.symbols),
    }

    rows: list[dict[str, Any]] = [{"_meta": run_meta}]
    ok_count = 0
    total = 0

    for idx, item in enumerate(items, start=1):
        qid = str(item.get("id") or f"q{idx}")
        query = item.get("query")
        if not isinstance(query, str) or not query.strip():
            raise SystemExit(f"Missing/invalid 'query' for item id={qid!r}")

        payload: dict[str, Any] = {
            "k": int(item.get("k") or args.k),
            "client": str(item.get("client") or args.client),
            "rag": str(item.get("rag") or args.rag),
            "sources": str(item.get("sources") or args.sources),
        }

        # Allow either explicit `messages` or a single-turn `query`.
        messages = item.get("messages")
        if isinstance(messages, list) and all(isinstance(m, dict) for m in messages):
            payload["messages"] = messages
        else:
            payload["messages"] = [{"role": "user", "content": query}]

        want_sources = item.get("want_sources")
        symbols_must_exist = item.get("symbols_must_exist")
        symbols_strict = item.get("symbols_strict")
        checks = Checks(
            want_sources=want_sources if isinstance(want_sources, bool) else None,
            contains=_coerce_list_of_str(item.get("contains")),
            contains_any=_coerce_list_of_str(item.get("contains_any")),
            not_contains=_coerce_list_of_str(item.get("not_contains")),
            symbols_must_exist=(
                bool(symbols_must_exist)
                if isinstance(symbols_must_exist, bool)
                else (True if args.check_symbols else None)
            ),
            symbols_strict=(
                bool(symbols_strict)
                if isinstance(symbols_strict, bool)
                else (True if args.strict_symbols else None)
            ),
        )

        started = time.perf_counter()
        status, obj, raw = _http_post_json(url, payload, api_key=api_key, timeout_s=int(args.timeout))
        elapsed = time.perf_counter() - started

        answer = ""
        sources: Any = None
        if isinstance(obj, dict):
            answer = str(obj.get("answer") or obj.get("content") or "")
            sources = obj.get("sources")

        evaluation = _evaluate(answer, sources, checks, symbols=symbols)
        ok = bool(evaluation.get("ok"))
        total += 1
        ok_count += 1 if ok else 0

        rows.append(
            {
                "id": qid,
                "query": query,
                "http": {"status": status, "elapsed_s": round(elapsed, 4)},
                "request": payload,
                "response": {"answer": answer, "sources": sources},
                "eval": evaluation,
                "raw": raw if status != 200 else None,
            }
        )

        tag = "OK" if ok else "FAIL"
        print(f"[{idx}/{len(items)}] {tag} {qid} (HTTP {status}, {elapsed:.2f}s)")

    _write_jsonl(out_path, rows)
    print(f"Wrote results: {out_path}")
    print(f"Summary: {ok_count}/{total} passed checks")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

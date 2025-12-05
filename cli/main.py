
"""Minimal CLI entrypoint for MolSys-AI.

The CLI currently supports:
- printing version information, and
- sending a single prompt to the agent using a stub model client.

Later it will open an interactive chat loop and expose more commands.
"""

import argparse
import sys
from typing import Dict, List

from agent.core import MolSysAIAgent
from agent.model_client import EchoModelClient, HTTPModelClient


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="molsys-ai", description="MolSys-AI CLI (MVP)")
    parser.add_argument(
        "--version", action="store_true", help="Print version information and exit"
    )
    parser.add_argument(
        "-m",
        "--message",
        help="Send a single user message to the MolSys-AI agent (uses a stub model client).",
    )
    parser.add_argument(
        "--server-url",
        help=(
            "Base URL of a running MolSys-AI model server. "
            "If provided, the CLI will call this server instead of using the local echo stub."
        ),
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print("MolSys-AI CLI (MVP)")
        return 0

    # Decide which model client to use.
    if args.server_url:
        model_client = HTTPModelClient(base_url=args.server_url)
    else:
        model_client = EchoModelClient()

    if args.message:
        agent = MolSysAIAgent(model_client=model_client)
        messages: List[Dict[str, str]] = [
            {"role": "user", "content": args.message},
        ]
        try:
            reply = agent.chat(messages)
        except RuntimeError as exc:
            print(f"[MolSys-AI CLI] Error: {exc}", file=sys.stderr)
            return 1
        else:
            print(reply)
            return 0

    print(
        "[MolSys-AI] CLI skeleton is in place.\n"
        "Use `--message` to send a prompt. Optionally provide "
        "`--server-url http://127.0.0.1:8000` to talk to a running model server."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

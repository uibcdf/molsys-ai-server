
"""Minimal CLI entrypoint for MolSys-AI.

For now this just prints a welcome message and exits.
Later it will open an interactive chat loop with the agent.
"""

import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="molsys-ai", description="MolSys-AI CLI (MVP)")
    parser.add_argument(
        "--version", action="store_true", help="Print version information and exit"
    )
    args = parser.parse_args(argv)

    if args.version:
        print("MolSys-AI CLI (MVP)")
        return 0

    print("[MolSys-AI] CLI skeleton is in place. Agent wiring will come next.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

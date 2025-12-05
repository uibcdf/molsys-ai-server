"""Skeleton script for training a LoRA/QLoRA variant of MolSys-AI models.

This file does not implement the actual training loop yet. It is meant to
serve as an entrypoint when the first experiments are added.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="molsys-ai-train-lora",
        description="Train a LoRA/QLoRA adapter for MolSys-AI (skeleton).",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a training configuration file (e.g. YAML).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    parser.print_help()
    print(
        "\n[MolSys-AI train] LoRA/QLoRA training is not implemented yet.\n"
        f"Configuration file provided: {args.config}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


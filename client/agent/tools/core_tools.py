from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass


def sys_info() -> str:
    return "\n".join(
        [
            f"platform: {platform.platform()}",
            f"python: {sys.version.split()[0]}",
            f"cwd: {os.getcwd()}",
        ]
    )


def shell(command: str, timeout_s: int = 60) -> str:
    """Run a shell command locally.

    This is a local-agent tool and is intended to be used only with explicit
    user approval in the CLI.
    """

    timeout_s = int(timeout_s)
    proc = subprocess.run(
        ["bash", "-lc", command],
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )

    out = (proc.stdout or "") + (proc.stderr or "")
    out = out.strip()
    if not out:
        out = "(no output)"

    # Truncate to avoid dumping huge logs into the prompt.
    max_chars = 20_000
    if len(out) > max_chars:
        out = out[:max_chars] + "\n...(truncated)..."

    return f"exit_code: {proc.returncode}\noutput:\n{out}"


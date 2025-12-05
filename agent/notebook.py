"""Helpers for using MolSys-AI from inside Jupyter notebooks.

These utilities provide a small, convenient interface to:

- create a `MolSysAIAgent` bound to an HTTP model server, and
- keep a simple chat session with message history.

The goal is to offer a clean starting point that can later grow into a richer
notebook experience (widgets, rich display, etc.) without changing the public
API defined here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import json

from .core import MolSysAIAgent
from .model_client import HTTPModelClient


DEFAULT_MODEL_SERVER_URL = os.environ.get(
    "MOLSYS_AI_MODEL_SERVER_URL", "http://127.0.0.1:8000"
)


def create_notebook_agent(
    server_url: Optional[str] = None,
    use_planner: bool = True,
) -> MolSysAIAgent:
    """Create a `MolSysAIAgent` configured for notebook usage.

    Parameters
    ----------
    server_url:
        Base URL of the MolSys-AI model server. If omitted, the value from
        ``MOLSYS_AI_MODEL_SERVER_URL`` is used, falling back to
        ``http://127.0.0.1:8000``.
    use_planner:
        Currently unused; kept for future expansion when alternative planners
        may be plugged in. For now, the default `SimplePlanner` is always used.
    """

    base_url = (server_url or DEFAULT_MODEL_SERVER_URL).rstrip("/")
    client = HTTPModelClient(base_url=base_url)
    agent = MolSysAIAgent(model_client=client)
    return agent


@dataclass
class NotebookChatSession:
    """Simple chat session helper for notebooks.

    This class keeps a conversation history and exposes an `ask()` method that:
    - appends the user message,
    - calls the agent with planning enabled,
    - appends the assistant reply,
    - returns the reply as a string.
    """

    agent: MolSysAIAgent
    messages: List[Dict[str, str]] = field(default_factory=list)

    def ask(self, question: str, force_rag: bool = False) -> str:
        """Send a question to the agent and return the reply.

        Parameters
        ----------
        question:
            The user question as a plain string.
        force_rag:
            If True, the planner will be instructed to use RAG even if its
            heuristic decision would not.
        """

        self.messages.append({"role": "user", "content": question})
        reply = self.agent.chat_with_planning(self.messages, force_rag=force_rag)
        self.messages.append({"role": "assistant", "content": reply})
        return reply


@dataclass
class NotebookCellSpec:
    """Specification for a notebook cell in a generated workflow.

    Parameters
    ----------
    cell_type:
        Either ``\"markdown\"`` or ``\"code\"``.
    source:
        The cell contents as a single string. Newlines are preserved as-is.
    """

    cell_type: str
    source: str


def create_workflow_notebook(
    cells: Sequence[NotebookCellSpec],
    output_path: Union[str, Path],
) -> Path:
    """Create a new Jupyter notebook implementing a workflow.

    This is the **safe** path for notebook integration: the current notebook
    is not modified. Instead, a new ``.ipynb`` file is written to ``output_path``.

    Parameters
    ----------
    cells:
        Sequence of :class:`NotebookCellSpec` objects describing the notebook
        cells (markdown or code).
    output_path:
        Path where the notebook should be written. If the suffix is missing,
        ``.ipynb`` will be appended.
    """

    path = Path(output_path)
    if path.suffix != ".ipynb":
        path = path.with_suffix(".ipynb")

    nb_cells: List[Dict[str, object]] = []
    for spec in cells:
        if spec.cell_type not in {"markdown", "code"}:
            raise ValueError(f"Unsupported cell_type: {spec.cell_type!r}")
        cell: Dict[str, object] = {
            "cell_type": spec.cell_type,
            "metadata": {},
            "source": spec.source.splitlines(keepends=True),
        }
        if spec.cell_type == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        nb_cells.append(cell)

    notebook = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "name": "python3",
                "language": "python",
                "display_name": "Python 3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    return path


def inject_workflow_into_current_notebook(*args: object, **kwargs: object) -> None:
    """Placeholder for in-place notebook editing (JupyterLab/VS Code).

    This function is intentionally **not** implemented yet. The long-term idea
    is to support:

    - inserting new cells into the current notebook (e.g. via the JupyterLab
      front-end APIs or a thin JavaScript bridge),
    - doing so only with explicit user confirmation and clear visual feedback.

    For safety reasons, and to keep the initial design simple, this function
    currently raises :class:`NotImplementedError`. The safe alternative is
    :func:`create_workflow_notebook`, which writes a new notebook file instead
    of modifying the active one.
    """

    raise NotImplementedError(
        "In-place notebook editing is not implemented yet. "
        "Use `create_workflow_notebook` to generate a new notebook instead."
    )


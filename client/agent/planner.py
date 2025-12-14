
"""Planning utilities for MolSys-AI.

This module defines a very small, MVP-level planner that:

- inspects the latest user message,
- (optionally) decides to use RAG for documentation-style questions,
- builds the set of messages that should be sent to the model.

The goal is to stabilise a simple interface that can later grow into a more
capable planner with multi-step reasoning and tool selection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Plan:
    """Minimal representation of a planning decision.

    Attributes
    ----------
    use_rag:
        Whether the planner decided to include RAG context in the model call.
    use_tools:
        Whether the planner decided that tools should be invoked.
    tool_name:
        The name of the tool to be invoked.
    tool_args:
        The arguments to be passed to the tool.
    """

    use_rag: bool = False
    use_tools: bool = False
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None


class SimplePlanner:
    """Very small heuristic planner.

    Current behaviour:
    - If `force_rag` is True, always use RAG.
    - Otherwise, decide to use RAG if the last user message looks like a
      documentation question (very naive heuristic).
    - Tools are not yet invoked, but the `Plan` structure already accounts
      for them.
    """

    def __init__(self, index_path: Path = Path("data/rag_index.pkl")):
        self.index_path = index_path

    def decide(self, messages: List[Dict[str, str]], force_rag: bool = False) -> Plan:
        """Return a simple plan given the current conversation."""

        if not messages:
            return Plan()

        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        if last_user is None:
            return Plan()

        content = last_user.get("content", "")

        if force_rag:
            return Plan(use_rag=True)

        # Explicit local shell mode: if the user message starts with "!".
        stripped = content.strip()
        if stripped.startswith("!") and len(stripped) > 1:
            return Plan(
                use_tools=True,
                tool_name="local.shell",
                tool_args={"command": stripped[1:].strip()},
            )

        # Heuristic 2: check for PDB ID and use the tool.
        match = re.search(r"\b([0-9][a-zA-Z0-9]{3})\b", content)
        if match:
            pdb_id = match.group(1)
            return Plan(
                use_tools=True,
                tool_name="molsysmt.get_info",
                tool_args={"pdb_id": pdb_id},
            )

        # Very naive heuristic: treat "what is", "how to" questions as
        # documentation-style and prefer RAG.
        lowered = content.lower()
        if lowered.startswith("what is ") or lowered.startswith("how to ") or "docs" in lowered:
            return Plan(use_rag=True)

        return Plan()

    def build_model_messages(
        self,
        messages: List[Dict[str, str]],
        plan: Plan,
        rag_k: int = 5,
    ) -> List[Dict[str, str]]:
        """Build the messages that should be sent to the model.

        If `plan.use_rag` is True, this method:
        - extracts the latest user question,
        - retrieves up to `rag_k` documents from the RAG layer,
        - builds a new set of messages including the documentation excerpts.

        Otherwise it returns the original messages unmodified.
        """

        if not plan.use_rag:
            return messages

        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        if last_user is None:
            return messages

        query = last_user.get("content", "")

        try:
            from rag.retriever import load_index, retrieve  # type: ignore
        except Exception:
            # If RAG deps are not installed, fall back to the raw conversation.
            return messages

        index = load_index(self.index_path)
        docs = retrieve(query, index, k=rag_k)

        context_lines: List[str] = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("path", "unknown")
            context_lines.append(f"[{i}] Source: {source}\n{doc.content}\n")

        context_block = "\n".join(context_lines) if context_lines else "(no documentation snippets found)"

        return [
            {
                "role": "system",
                "content": (
                    "You are the MolSys-AI assistant. When RAG context is provided, "
                    "answer the user question using only that context. If the answer "
                    "cannot be inferred from the context, say so explicitly."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Documentation excerpts:\n\n{context_block}\n\n"
                    f"Question: {query}"
                ),
            },
        ]

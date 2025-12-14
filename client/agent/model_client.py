"""Model client abstractions for MolSys-AI.

The agent talks to language models through this layer so that:
- the core logic does not depend on a specific backend,
- switching from an echo stub to llama.cpp, vLLM, etc. does not require
  changes in the agent code.

For the MVP we provide a minimal `EchoModelClient` placeholder. A future
`HTTPModelClient` will call the FastAPI model server defined in
`model_server/server.py`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests


class ModelClient(ABC):
    """Abstract base class for model clients.

    Concrete implementations may talk to:
    - a local stub,
    - a FastAPI server (see `model_server.server`),
    - or other backends (e.g. vLLM, remote services).
    """

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], *, generation: Optional[Dict[str, Any]] = None) -> str:
        """Generate a reply given a ChatML-like list of messages."""


class EchoModelClient(ModelClient):
    """Very small placeholder model client.

    This implementation ignores all messages and returns a fixed reply.
    It is useful for wiring tests and early CLI/agent experiments
    without requiring a running model server.
    """

    def generate(self, messages: List[Dict[str, str]], *, generation: Optional[Dict[str, Any]] = None) -> str:
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            None,
        )
        if last_user is None:
            return "[MolSys-AI stub reply] No user message found."
        return f"[MolSys-AI stub reply] You said: {last_user}"


class HTTPModelClient(ModelClient):
    """HTTP-based model client.

    This implementation calls the FastAPI model server defined in
    :mod:`model_server.server` using its ``/v1/chat`` endpoint.
    """

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        # Normalise base URL to avoid double slashes.
        self.base_url = base_url.rstrip("/")
        self.api_key = (api_key or "").strip() or None

    def generate(self, messages: List[Dict[str, str]], *, generation: Optional[Dict[str, Any]] = None) -> str:
        url = f"{self.base_url}/v1/chat"
        payload: Dict[str, Any] = {"messages": messages}
        if generation:
            payload["generation"] = generation
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=300)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network
            raise RuntimeError(f"Failed to call model server at {url}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("Model server returned invalid JSON.") from exc

        content = data.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Model server response does not contain a string 'content' field.")

        return content

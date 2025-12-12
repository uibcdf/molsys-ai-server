
"""FastAPI-based model server for MolSys-AI (MVP skeleton).

The server exposes a `/v1/chat` endpoint and can be configured to use:

- a simple stub backend that echoes the last user message, or
- (in the future) a real backend powered by llama.cpp or other engines.

The configuration is read from a YAML file whose path is given by the
``MOLSYS_AI_MODEL_CONFIG`` environment variable, or defaults to
``model_server/config.yaml``. If no configuration file is found, the
server falls back to the stub backend.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:  # Optional dependency; only needed if a YAML config exists.
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


app = FastAPI(title="MolSys-AI Model Server (MVP)")

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
_CONFIG_ENV_VAR = "MOLSYS_AI_MODEL_CONFIG"


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class ChatResponse(BaseModel):
    content: str


class ModelBackend:
    """Base interface for model backends."""

    def chat(self, messages: List[Message]) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class StubBackend(ModelBackend):
    """Echo backend used when no model is configured."""

    def chat(self, messages: List[Message]) -> str:
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        content = last_user.content if last_user else "(no user message provided)"
        return f"[MolSys-AI stub reply] You said: {content}"


class LlamaCppBackend(ModelBackend):
    """Backend powered by llama.cpp-python (placeholder skeleton).

    This implementation expects a `local_path` field under the `model`
    section of the configuration, pointing to a local GGUF model file.
    """

    def __init__(self, model_cfg: Dict[str, Any]) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on external package
            raise RuntimeError(
                "llama_cpp-python is not installed but backend 'llama_cpp' "
                "was requested in the configuration."
            ) from exc

        local_path = model_cfg.get("local_path")
        if not local_path:
            raise RuntimeError(
                "Configuration for backend 'llama_cpp' must define "
                "`model.local_path` with a path to a GGUF model file."
            )

        # Minimal Llama initialisation; additional parameters can be added
        # later according to hardware and performance needs.
        self._llama = Llama(model_path=str(local_path), n_gpu_layers=-1, verbose=True)

    def chat(self, messages: List[Message]) -> str:
        # Very simple prompt formatting for the MVP.
        prompt_parts = []
        for msg in messages:
            role = msg.role.upper()
            prompt_parts.append(f"{role}: {msg.content}")
        prompt_parts.append("ASSISTANT:")
        prompt = "\n".join(prompt_parts)

        try:
            # The exact call signature may be tuned later via ADRs.
            output = self._llama(prompt, max_tokens=256, stop=["USER:", "ASSISTANT:"])
        except Exception as exc:  # pragma: no cover - backend-specific
            raise RuntimeError("Model backend failed to generate a reply.") from exc

        try:
            text = output["choices"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Model backend returned an unexpected response format.") from exc

        return text.strip()


@lru_cache()
def load_config() -> Dict[str, Any]:
    """Load the YAML configuration for the model server.

    If no config file exists, a default stub configuration is returned.
    """
    path_str = os.environ.get(_CONFIG_ENV_VAR, str(_DEFAULT_CONFIG_PATH))
    path = Path(path_str)

    if not path.exists():
        # No config present: fall back to the stub backend.
        return {"model": {"backend": "stub"}}

    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to read the model server configuration, "
            "but it is not installed."
        )

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except OSError as exc:
        raise RuntimeError(f"Failed to read model server config at {path}.") from exc

    return data


@lru_cache()
def get_model_backend() -> ModelBackend:
    """Instantiate the configured model backend (cached)."""

    cfg = load_config()
    model_cfg = cfg.get("model") or {}
    backend_name = model_cfg.get("backend", "stub")

    if backend_name == "stub":
        return StubBackend()

    if backend_name == "llama_cpp":
        return LlamaCppBackend(model_cfg)

    raise RuntimeError(f"Unsupported model backend '{backend_name}'.")


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Chat endpoint backed by the configured model backend."""

    try:
        backend = get_model_backend()
        content = backend.chat(req.messages)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(content=content)

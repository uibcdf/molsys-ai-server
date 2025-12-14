
"""FastAPI-based model server for MolSys-AI (MVP skeleton).

The server exposes a `/v1/chat` endpoint and can be configured to use:

- a simple stub backend that echoes the last user message, or
- (in the future) a real backend powered by llama.cpp or other engines.

The configuration is read from a YAML file whose path is given by the
``MOLSYS_AI_MODEL_CONFIG`` environment variable, or defaults to
``server/model_server/config.yaml``. If no configuration file is found, the
server falls back to the stub backend.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

try:  # Optional dependency; only needed if a YAML config exists.
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


app = FastAPI(title="MolSys-AI Model Server (MVP)")

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
_CONFIG_ENV_VAR = "MOLSYS_AI_MODEL_CONFIG"
_CHAT_KEYS_ENV_VAR = "MOLSYS_AI_CHAT_API_KEYS"


class Message(BaseModel):
    role: str
    content: str


class GenerationParams(BaseModel):
    """Optional generation controls for `/v1/chat`.

    These fields are optional to keep `/v1/chat` stable while still enabling
    deterministic sub-calls (for example router/classifier prompts).
    """

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None


class ChatRequest(BaseModel):
    messages: List[Message]
    generation: GenerationParams | None = None


class ChatResponse(BaseModel):
    content: str


@app.get("/healthz")
async def healthz() -> dict:
    # Do not call `get_model_backend()` here: it may trigger heavy model loading.
    cfg = load_config()
    model_cfg = cfg.get("model") or {}
    backend = str(model_cfg.get("backend", "stub"))
    auth_enabled = bool(get_chat_api_keys())
    return {"status": "ok", "backend": backend, "chat_auth_enabled": auth_enabled}


class ModelBackend:
    """Base interface for model backends."""

    def chat(
        self, messages: List[Message], *, generation: GenerationParams | None = None
    ) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class StubBackend(ModelBackend):
    """Echo backend used when no model is configured."""

    def chat(self, messages: List[Message], *, generation: GenerationParams | None = None) -> str:
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

    def chat(self, messages: List[Message], *, generation: GenerationParams | None = None) -> str:
        # Very simple prompt formatting for the MVP.
        prompt_parts = []
        for msg in messages:
            role = msg.role.upper()
            prompt_parts.append(f"{role}: {msg.content}")
        prompt_parts.append("ASSISTANT:")
        prompt = "\n".join(prompt_parts)

        max_tokens = 256
        temperature = 0.7
        top_p = 0.95
        if generation is not None:
            if generation.max_tokens is not None:
                max_tokens = int(generation.max_tokens)
            if generation.temperature is not None:
                temperature = float(generation.temperature)
            if generation.top_p is not None:
                top_p = float(generation.top_p)

        try:
            # The exact call signature may be tuned later via ADRs.
            output = self._llama(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["USER:", "ASSISTANT:"],
            )
        except Exception as exc:  # pragma: no cover - backend-specific
            raise RuntimeError("Model backend failed to generate a reply.") from exc

        try:
            text = output["choices"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Model backend returned an unexpected response format.") from exc

        return text.strip()


class VLLMBackend(ModelBackend):
    """Backend powered by vLLM.

    This implementation expects `local_path` and `tensor_parallel_size`
    fields under the `model` section of the configuration, pointing to
    a local model directory or HuggingFace ID, and the number of GPUs to use.
    """

    def __init__(self, model_cfg: Dict[str, Any]) -> None:
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on external package
            raise RuntimeError(
                "vLLM is not installed but backend 'vllm' "
                "was requested in the configuration."
            ) from exc

        try:
            from transformers import AutoTokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on external package
            raise RuntimeError(
                "transformers is required for backend 'vllm' "
                "(to apply chat templates), but it is not installed."
            ) from exc

        model_path = model_cfg.get("local_path")
        if not model_path:
            raise RuntimeError(
                "Configuration for backend 'vllm' must define "
                "`model.local_path` with a path to a model file or HuggingFace ID."
            )

        tensor_parallel_size = model_cfg.get("tensor_parallel_size", 1)
        quantization = model_cfg.get(
            "quantization", None
        )  # Default to None to use full precision or model's default
        max_model_len = model_cfg.get("max_model_len")
        gpu_memory_utilization = model_cfg.get("gpu_memory_utilization")
        enforce_eager = model_cfg.get("enforce_eager")
        dtype = model_cfg.get("dtype")
        self._max_model_len: Optional[int] = int(max_model_len) if max_model_len is not None else None

        def _coerce_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"1", "true", "yes", "y", "on"}:
                    return True
                if lowered in {"0", "false", "no", "n", "off"}:
                    return False
            raise ValueError(f"Invalid boolean value: {value!r}")

        llm_kwargs: Dict[str, Any] = {
            "model": model_path,
            "tensor_parallel_size": int(tensor_parallel_size),
            "quantization": quantization,
            "trust_remote_code": True,  # Required for some models, e.g., Qwen
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = int(max_model_len)
        if gpu_memory_utilization is not None:
            llm_kwargs["gpu_memory_utilization"] = float(gpu_memory_utilization)
        if enforce_eager is not None:
            llm_kwargs["enforce_eager"] = _coerce_bool(enforce_eager)
        if dtype is not None:
            llm_kwargs["dtype"] = str(dtype)

        # Minimal LLM initialisation; additional parameters can be added
        # later according to hardware and performance needs.
        self._llm = LLM(**llm_kwargs)
        self._sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=256,
        )
        self._max_new_tokens = int(self._sampling_params.max_tokens or 0)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

    def _sampling_for(self, generation: GenerationParams | None) -> Any:
        if generation is None:
            return self._sampling_params

        try:
            from vllm import SamplingParams  # type: ignore
        except Exception:
            return self._sampling_params

        max_tokens = int(self._sampling_params.max_tokens or 256)
        temperature = float(self._sampling_params.temperature or 0.7)
        top_p = float(self._sampling_params.top_p or 0.95)

        if generation.max_tokens is not None:
            max_tokens = int(generation.max_tokens)
        if generation.temperature is not None:
            temperature = float(generation.temperature)
        if generation.top_p is not None:
            top_p = float(generation.top_p)

        return SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    def _format_prompt(self, messages: List[Message]) -> str:
        chat_messages = [{"role": m.role, "content": m.content} for m in messages]

        # Prefer the model's chat template when available (required for Llama-3.1 Instruct).
        try:
            has_template = bool(getattr(self._tokenizer, "chat_template", None))
        except Exception:
            has_template = False

        if has_template and hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Fallback: basic role-tag formatting.
        parts: list[str] = []
        for msg in messages:
            parts.append(f"{msg.role.upper()}: {msg.content}")
        parts.append("ASSISTANT:")
        return "\n".join(parts)

    def _truncate_history_if_needed(self, messages: List[Message]) -> List[Message]:
        if self._max_model_len is None:
            return messages

        # Keep some headroom for generation.
        budget = max(self._max_model_len - self._max_new_tokens, 1)

        def prompt_len(msgs: List[Message]) -> int:
            prompt = self._format_prompt(msgs)
            return int(len(self._tokenizer(prompt, add_special_tokens=False).input_ids))

        if prompt_len(messages) <= budget:
            return messages

        system_msgs = [m for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]

        # Drop oldest non-system messages until it fits.
        tail: List[Message] = []
        for msg in reversed(non_system):
            tail.append(msg)
            candidate = system_msgs + list(reversed(tail))
            if prompt_len(candidate) > budget:
                tail.pop()
                break

        candidate = system_msgs + list(reversed(tail))
        if candidate and prompt_len(candidate) <= budget:
            return candidate

        # Last resort: keep only the last user message.
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        return system_msgs + ([last_user] if last_user else [])

    def chat(self, messages: List[Message], *, generation: GenerationParams | None = None) -> str:
        if not messages:
            raise RuntimeError("No messages provided.")

        trimmed_messages = self._truncate_history_if_needed(messages)
        prompt = self._format_prompt(trimmed_messages)

        sampling = self._sampling_for(generation)
        try:
            outputs = self._llm.generate(
                [prompt],
                sampling,
                # For chat models, it's better to use `apply_chat_template`
                # if the model supports it, but for a general `ModelBackend`
                # this is simpler.
            )
            text = outputs[0].outputs[0].text
        except Exception as exc: # pragma: no cover - backend-specific
            raise RuntimeError("Model backend failed to generate a reply.") from exc

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

    if backend_name == "vllm":
        return VLLMBackend(model_cfg)

    raise RuntimeError(f"Unsupported model backend '{backend_name}'.")

def _parse_api_keys(raw: str) -> Set[str]:
    return {k.strip() for k in raw.split(",") if k.strip()}


@lru_cache()
def get_chat_api_keys() -> Set[str]:
    """Return accepted API keys for `/v1/chat`.

    Behavior:
    - If `MOLSYS_AI_CHAT_API_KEYS` is empty/unset: endpoint is open (dev default).
    - If set: the request must include either:
      - `Authorization: Bearer <key>`, or
      - `X-API-Key: <key>`.
    """

    raw = (os.environ.get(_CHAT_KEYS_ENV_VAR) or "").strip()
    return _parse_api_keys(raw)


def _extract_api_key(request: Request) -> str | None:
    auth = (request.headers.get("authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        return token or None
    x_api_key = (request.headers.get("x-api-key") or "").strip()
    return x_api_key or None


def require_chat_api_key(request: Request) -> None:
    allowed_keys = get_chat_api_keys()
    if not allowed_keys:
        return
    key = _extract_api_key(request)
    if not key or key not in allowed_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


@app.post(
    "/v1/chat",
    response_model=ChatResponse,
    dependencies=[Depends(require_chat_api_key)],
)
async def chat(req: ChatRequest) -> ChatResponse:
    """Chat endpoint backed by the configured model backend."""

    if not req.messages:
        raise HTTPException(status_code=400, detail="At least one message is required.")

    try:
        backend = get_model_backend()
        content = backend.chat(req.messages, generation=req.generation)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(content=content)

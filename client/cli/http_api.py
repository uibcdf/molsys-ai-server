from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


def _auth_headers(api_key: str | None) -> Dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def is_u1_key(api_key: str | None) -> bool:
    return bool(api_key) and api_key.startswith("u1_")


def pick_base_urls(
    *,
    api_key: str | None,
    public_base_url: str,
    lan_base_url: str | None,
) -> List[str]:
    public_base_url = public_base_url.rstrip("/")
    lan_base_url = (lan_base_url or "").rstrip("/") or None

    if is_u1_key(api_key) and lan_base_url:
        return [lan_base_url, public_base_url]
    return [public_base_url]


def post_engine_chat(
    *,
    base_urls: Sequence[str],
    api_key: str | None,
    messages: List[Dict[str, str]],
    timeout: tuple[float, float],
) -> str:
    last_exc: Exception | None = None
    for base in base_urls:
        url = f"{base.rstrip('/')}/v1/engine/chat"
        try:
            resp = requests.post(
                url,
                json={"messages": messages},
                headers={"Content-Type": "application/json", **_auth_headers(api_key)},
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last_exc = exc
            continue

        if resp.status_code == 401:
            raise RuntimeError("Unauthorized: invalid or missing API key for /v1/engine/chat.")
        if resp.status_code >= 400:
            raise RuntimeError(f"Server error calling {url}: HTTP {resp.status_code} {resp.text[:200]}")

        try:
            data = resp.json()
        except ValueError as exc:
            raise RuntimeError("Server returned invalid JSON.") from exc

        content = data.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Server response missing 'content' string field.")
        return content

    raise RuntimeError(f"Failed to reach any configured server. Last error: {last_exc!r}")


def post_chat(
    *,
    base_url: str,
    api_key: str | None,
    query: str,
    k: int = 5,
    timeout: tuple[float, float],
) -> str:
    data, _sources = post_chat_json(
        base_url=base_url,
        api_key=api_key,
        query=query,
        messages=None,
        k=k,
        client=None,
        rag=None,
        sources=None,
        timeout=timeout,
    )
    answer = data.get("answer")
    if not isinstance(answer, str):
        raise RuntimeError("Server response missing 'answer' string field.")
    return answer


def post_chat_json(
    *,
    base_url: str,
    api_key: str | None,
    query: str | None,
    messages: List[Dict[str, str]] | None,
    k: int,
    client: str | None,
    rag: str | None,
    sources: str | None,
    timeout: tuple[float, float],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Call `/v1/chat` and return the parsed JSON response plus sources list."""

    if not query and not messages:
        raise ValueError("Provide either query or messages.")

    url = f"{base_url.rstrip('/')}/v1/chat"
    payload: Dict[str, Any] = {"k": int(k)}
    if query:
        payload["query"] = query
    if messages is not None:
        payload["messages"] = messages
    if client:
        payload["client"] = client
    if rag:
        payload["rag"] = rag
    if sources:
        payload["sources"] = sources

    try:
        resp = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json", **_auth_headers(api_key)},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to call chat API at {url}") from exc

    if resp.status_code == 401:
        raise RuntimeError("Unauthorized: invalid or missing API key for /v1/chat.")
    if resp.status_code >= 400:
        raise RuntimeError(f"Server error calling {url}: HTTP {resp.status_code} {resp.text[:200]}")

    try:
        data = resp.json()
    except ValueError as exc:
        raise RuntimeError("Server returned invalid JSON.") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Server response is not a JSON object.")

    srcs = data.get("sources")
    if not isinstance(srcs, list):
        srcs = []
    # Ensure dict elements.
    clean_srcs: List[Dict[str, Any]] = [s for s in srcs if isinstance(s, dict)]
    return data, clean_srcs


def post_chat_json_any(
    *,
    base_urls: Sequence[str],
    api_key: str | None,
    query: str | None,
    messages: List[Dict[str, str]] | None,
    k: int,
    client: str | None,
    rag: str | None,
    sources: str | None,
    timeout: tuple[float, float],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    last_exc: Exception | None = None
    for base in base_urls:
        try:
            return post_chat_json(
                base_url=base,
                api_key=api_key,
                query=query,
                messages=messages,
                k=k,
                client=client,
                rag=rag,
                sources=sources,
                timeout=timeout,
            )
        except RuntimeError as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Failed to reach any configured server. Last error: {last_exc!r}")


def post_chat_messages(
    *,
    base_url: str,
    api_key: str | None,
    messages: List[Dict[str, str]],
    k: int = 5,
    timeout: tuple[float, float],
) -> str:
    data, _sources = post_chat_json(
        base_url=base_url,
        api_key=api_key,
        query=None,
        messages=messages,
        k=k,
        client=None,
        rag=None,
        sources=None,
        timeout=timeout,
    )
    answer = data.get("answer")
    if not isinstance(answer, str):
        raise RuntimeError("Server response missing 'answer' string field.")
    return answer


def resolve_api_key(cli_arg: str | None, config_key: str | None) -> str | None:
    return (
        (cli_arg or "").strip()
        or (os.environ.get("MOLSYS_AI_API_KEY") or "").strip()
        or (os.environ.get("MOLSYS_AI_CHAT_API_KEY") or "").strip()
        or (config_key or "").strip()
        or None
    )

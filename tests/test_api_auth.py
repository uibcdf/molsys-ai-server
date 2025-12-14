import importlib

import pytest
from fastapi.testclient import TestClient


def _reload_model_server():
    import model_server.server as server

    importlib.reload(server)
    return server


def test_model_server_chat_open_by_default(monkeypatch):
    monkeypatch.delenv("MOLSYS_AI_CHAT_API_KEYS", raising=False)
    server = _reload_model_server()

    client = TestClient(server.app)
    resp = client.post("/v1/chat", json={"messages": [{"role": "user", "content": "hello"}]})
    assert resp.status_code == 200
    assert "content" in resp.json()


def test_model_server_chat_requires_api_key_when_configured(monkeypatch):
    monkeypatch.setenv("MOLSYS_AI_CHAT_API_KEYS", "k1,k2")
    server = _reload_model_server()

    client = TestClient(server.app)
    payload = {"messages": [{"role": "user", "content": "hello"}]}

    resp = client.post("/v1/chat", json=payload)
    assert resp.status_code == 401

    resp = client.post("/v1/chat", json=payload, headers={"Authorization": "Bearer k2"})
    assert resp.status_code == 200

    resp = client.post("/v1/chat", json=payload, headers={"X-API-Key": "k1"})
    assert resp.status_code == 200


def _reload_docs_chat(monkeypatch, tmp_path):
    monkeypatch.setenv("MOLSYS_AI_EMBEDDINGS", "hashing")
    monkeypatch.setenv("MOLSYS_AI_DOCS_DIR", str(tmp_path / "docs"))
    monkeypatch.setenv("MOLSYS_AI_DOCS_INDEX", str(tmp_path / "rag_index.pkl"))

    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs" / "example.md").write_text("# Example\n\nHello.\n", encoding="utf-8")

    import docs_chat.backend as backend

    importlib.reload(backend)
    return backend


def test_docs_chat_can_be_protected_with_api_keys(monkeypatch, mocker, tmp_path):
    backend = _reload_docs_chat(monkeypatch, tmp_path)

    monkeypatch.setenv("MOLSYS_AI_DOCS_CHAT_API_KEYS", "doc-key")
    importlib.reload(backend)

    mocker.patch.object(backend.HTTPModelClient, "generate", return_value="ok")

    client = TestClient(backend.app)
    payload = {"query": "hi", "k": 1}

    resp = client.post("/v1/docs-chat", json=payload)
    assert resp.status_code == 401

    resp = client.post("/v1/docs-chat", json=payload, headers={"Authorization": "Bearer doc-key"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "ok"


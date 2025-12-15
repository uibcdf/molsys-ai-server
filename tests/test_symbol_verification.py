import importlib
import json

from fastapi.testclient import TestClient


def _reload_chat_api(monkeypatch, tmp_path):
    monkeypatch.setenv("MOLSYS_AI_EMBEDDINGS", "hashing")
    monkeypatch.setenv("MOLSYS_AI_DOCS_DIR", str(tmp_path / "docs"))
    monkeypatch.setenv("MOLSYS_AI_DOCS_INDEX", str(tmp_path / "rag_index.pkl"))
    monkeypatch.setenv("MOLSYS_AI_CHAT_VERIFY_SYMBOLS", "true")

    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs" / "example.md").write_text("# Example\n\nHello.\n", encoding="utf-8")

    import chat_api.backend as backend

    importlib.reload(backend)
    return backend


def test_chat_api_symbol_verification_rewrites_invalid_symbols(monkeypatch, mocker, tmp_path):
    backend = _reload_chat_api(monkeypatch, tmp_path)

    # Minimal symbol registry: allow one real-ish symbol, but not `molsysmt.fetch`.
    symbols_path = tmp_path / "docs" / "_symbols.json"
    symbols_path.write_text(
        json.dumps(
            {
                "generated_at_utc": "x",
                "dest": str(tmp_path / "docs"),
                "projects": {
                    "molsysmt": {
                        "ok": True,
                        "count": 1,
                        "symbols": ["molsysmt.structure.get_rmsd"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MOLSYS_AI_SYMBOLS_PATH", str(symbols_path))
    importlib.reload(backend)

    # First generation returns an invented alias-based call `msmt.fetch(...)`.
    # The symbol verifier should force a rewrite (second generation).
    def fake_generate(messages, generation=None):
        text = "\n".join(m.get("content", "") for m in messages if isinstance(m, dict))
        if "Draft answer:" in text:
            return "NOT_DOCUMENTED: I could not find `molsysmt.fetch` in the available API surface."
        return "```python\nimport molsysmt as msmt\nsys = msmt.fetch('pdb:1VII')\n```"

    mocker.patch.object(backend.HTTPModelClient, "generate", side_effect=fake_generate)

    client = TestClient(backend.app)
    resp = client.post(
        "/v1/chat",
        json={"messages": [{"role": "user", "content": "How do I fetch PDB 1VII?"}], "rag": "off", "sources": "off"},
    )
    assert resp.status_code == 200
    out = resp.json()["answer"]
    assert "msmt.fetch" not in out
    assert "NOT_DOCUMENTED" in out


def test_chat_api_reread_does_not_add_unknown_symbols(monkeypatch, mocker, tmp_path):
    backend = _reload_chat_api(monkeypatch, tmp_path)

    symbols_path = tmp_path / "docs" / "_symbols.json"
    symbols_path.write_text(
        json.dumps(
            {
                "generated_at_utc": "x",
                "dest": str(tmp_path / "docs"),
                "projects": {
                    "molsysmt": {
                        "ok": True,
                        "count": 1,
                        "symbols": ["molsysmt.structure.get_rmsd"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MOLSYS_AI_SYMBOLS_PATH", str(symbols_path))
    monkeypatch.setenv("MOLSYS_AI_CHAT_REREAD_SYMBOLS", "true")

    # Provide a minimal api_surface snippet in the docs corpus so retrieval has something.
    api_dir = tmp_path / "docs" / "molsysmt" / "api_surface" / "structure"
    api_dir.mkdir(parents=True, exist_ok=True)
    (api_dir / "get_rmsd.md").write_text("Project: molsysmt\nModule: molsysmt.structure.get_rmsd\n\n### `get_rmsd(...)`\n", encoding="utf-8")

    importlib.reload(backend)

    # The model returns an answer that contains a valid symbol; the re-read rewrite pass
    # should not introduce unknown symbols.
    def fake_generate(messages, generation=None):
        text = "\n".join(m.get("content", "") for m in messages if isinstance(m, dict))
        if "API excerpts" in text:
            return "Use `molsysmt.structure.get_rmsd(...)` to compute RMSD."
        return "Use `molsysmt.structure.get_rmsd(...)` to compute RMSD."

    mocker.patch.object(backend.HTTPModelClient, "generate", side_effect=fake_generate)

    client = TestClient(backend.app)
    resp = client.post(
        "/v1/chat",
        json={"messages": [{"role": "user", "content": "How do I compute RMSD?"}], "rag": "on", "sources": "off"},
    )
    assert resp.status_code == 200
    out = resp.json()["answer"]
    assert "NOT_DOCUMENTED" not in out
    assert "molsysmt.structure.get_rmsd" in out

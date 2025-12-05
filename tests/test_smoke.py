
def test_smoke_imports():
    """Very minimal smoke test to ensure basic modules import without errors."""
    import importlib

    for mod in [
        "agent.core",
        "agent.model_client",
        "agent.notebook",
        "agent.planner",
        "agent.executor",
        "rag.build_index",
        "rag.embeddings",
        "rag.retriever",
        "model_server.server",
        "docs_chat.backend",
        "cli.main",
    ]:
        importlib.import_module(mod)

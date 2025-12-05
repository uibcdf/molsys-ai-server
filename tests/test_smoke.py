
def test_smoke_imports():
    """Very minimal smoke test to ensure basic modules import without errors."""
    import importlib

    for mod in [
        "agent.core",
        "rag.build_index",
        "model_server.server",
        "cli.main",
    ]:
        importlib.import_module(mod)

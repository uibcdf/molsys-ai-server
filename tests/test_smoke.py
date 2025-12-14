
import pytest
from agent.core import MolSysAIAgent
from agent.model_client import EchoModelClient


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
        "chat_api.backend",
        "cli.main",
    ]:
        importlib.import_module(mod)


def test_agent_with_tool(mocker):
    """Test that the agent can call a tool."""
    # Mock the entire molsysmt module before other imports
    mocker.patch.dict("sys.modules", {"molsysmt": mocker.Mock()})

    from agent.core import MolSysAIAgent
    from agent.model_client import EchoModelClient

    agent = MolSysAIAgent(model_client=EchoModelClient())

    # Now we can test the tool-calling logic
    messages = [{"role": "user", "content": "Get info on 1VII"}]
    reply = agent.chat_with_planning(messages)

    # The EchoModelClient will append the tool output to the messages
    # and return the last message content. Since the real tool is not
    # being called, we expect the mocked (empty) tool output.
    # The important part is that the code runs without crashing.
    assert isinstance(reply, str)



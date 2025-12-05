
"""Core agent loop for MolSys-AI.

This is a very early skeleton. The goal is to keep the public API stable while
we iterate on the internal implementation.
"""

from typing import List, Dict, Any

class MolSysAIAgent:
    """Minimal placeholder agent.

    For now it only echoes messages through a model client stub.
    Later it will perform planning, tool-calling and RAG.
    """

    def __init__(self, model_client: "ModelClient") -> None:
        self.model_client = model_client

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send a list of messages to the underlying model.

        Parameters
        ----------
        messages:
            List of messages in the usual ChatML-like structure:
            [{"role": "user"|"assistant"|"system", "content": "..."}, ...]

        Returns
        -------
        str
            The assistant reply (plain text) for now.
        """
        return self.model_client.generate(messages)

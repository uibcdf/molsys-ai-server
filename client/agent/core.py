
"""Core agent loop for MolSys-AI.

This is a very early skeleton. The goal is to keep the public API stable while
we iterate on the internal implementation.
"""

from typing import Dict, List, Optional

from .executor import ToolExecutor, create_default_executor
from .model_client import ModelClient
from .planner import Plan, SimplePlanner


class MolSysAIAgent:
    """Minimal placeholder agent.

    For now it only forwards messages to a model client implementation.
    Later it will perform planning, tool-calling and RAG.
    """

    def __init__(
        self,
        model_client: ModelClient,
        planner: Optional[SimplePlanner] = None,
        executor: Optional[ToolExecutor] = None,
    ) -> None:
        self.model_client = model_client
        self.planner = planner or SimplePlanner()
        self.executor = executor or create_default_executor()

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

    def chat_with_planning(self, messages: List[Dict[str, str]], force_rag: bool = False) -> str:
        """High-level chat method that uses the planner (and later tools).

        For the MVP this method:
        - asks the :class:`SimplePlanner` for a :class:`Plan`,
        - if `plan.use_rag` is True, rebuilds the messages with RAG context,
        - calls the model client with the resulting messages.
        """

        plan: Plan = self.planner.decide(messages, force_rag=force_rag)

        if plan.use_tools:
            tool_name = plan.tool_name or ""
            tool_args = plan.tool_args or {}
            try:
                tool_output = self.executor.execute(tool_name, **tool_args)
            except Exception as e:
                tool_output = f"Error executing tool {tool_name}: {e}"

            tool_message = {"role": "tool", "content": tool_output}
            messages.append(tool_message)
            return self.model_client.generate(messages)

        planned_messages = self.planner.build_model_messages(messages, plan)
        return self.model_client.generate(planned_messages)


# MolSys-AI Agent

This directory contains the core logic of the MolSys-AI agent.

## Modules

- `core.py`:
  - Defines the main `MolSysAIAgent` class.
  - Responsible for orchestrating:
    - user messages,
    - calls to the language model (via a `ModelClient`),
    - basic planning and (later) tool execution.

- `planner.py`:
  - Implements a very simple planner that:
    - analyses the last user message,
    - decides whether to use RAG (documentation context),
    - builds the messages that should be sent to the model.
  - This is an MVP interface that can grow into more advanced planning.

- `executor.py`:
  - Implements a minimal tool executor:
    - maps tool names to Python callables,
    - provides a basic registry and `execute()` method.
  - A default executor is created with a dummy MolSysMT tool used for wiring tests.

- `tools/`:
  - Houses tool wrappers around:
    - MolSysMT,
    - MolSysViewer,
    - TopoMT,
    - and other ecosystem tools (e.g. OpenMM).
  - Each file in this package will implement a small, focused set
    of tool functions with clear signatures.

## High-level flow (target design)

1. The CLI or the docs chatbot sends messages to the agent.
2. The agent:
   - sees the current conversation,
   - optionally calls the planner to decide on a next action.
3. The planner may:
   - ask the model for thoughts,
   - decide to call a tool,
   - or decide to answer directly.
4. If a tool is needed:
   - the executor is invoked,
   - the tool is run in a controlled environment,
   - the result is added back into the conversation context.
5. A final answer is produced and returned to the caller.

This flow is still in its MVP stage and will evolve,
but the goal is to keep `core.py` as the stable entrypoint.

## Future Evolution: Scaling Tool Capabilities (Hybrid Approach)

The current agent architecture is built on a minimalist, in-house implementation of a ReAct-style loop (`MolSysAIAgent`, `SimplePlanner`, `ToolExecutor`). This approach was chosen deliberately to maintain full control, ensure simplicity for the MVP, and avoid large external dependencies like agent frameworks (e.g., LangChain, LlamaIndex).

For future scalability, especially when adding general-purpose tools (e.g., a shell/terminal, a Python REPL, git integration), the recommended strategy is a **hybrid approach**:

1.  **Keep the Core Agent Loop**: Continue using the custom `MolSysAIAgent` and `ToolExecutor` to maintain control over the agent's core logic.
2.  **Integrate Pre-built Tools**: Instead of re-implementing common tools from scratch, import battle-tested tool implementations from community libraries (e.g., `langchain-community`). These pre-built tools can be registered into our existing `ToolExecutor` just like any custom-built tool.

This strategy offers the best of both worlds: it preserves the simplicity and control of the custom agent core while leveraging the power and robustness of the open-source ecosystem for common, general-purpose functionalities. This allows the development effort to remain focused on creating unique, high-value tools for the `molsysmt` ecosystem.

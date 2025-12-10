
"""Tool execution layer for MolSys-AI.

The executor is responsible for:
- mapping tool names to Python callables,
- executing them safely,
- marshalling inputs/outputs.

For the MVP we provide a small, in-memory registry and a very simple executor
that can call tools such as the dummy MolSysMT helper in `agent.tools`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, MutableMapping


ToolFunc = Callable[..., Any]


@dataclass
class ToolSpec:
    """Metadata for a registered tool."""

    name: str
    func: ToolFunc
    description: str = ""


class ToolExecutor:
    """Minimal tool executor for the MolSys-AI agent."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    @property
    def tools(self) -> Mapping[str, ToolSpec]:
        """Read-only view of the registered tools."""

        return dict(self._tools)

    def register(self, name: str, func: ToolFunc, description: str = "") -> None:
        """Register a tool under the given name."""

        self._tools[name] = ToolSpec(name=name, func=func, description=description)

    def execute(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a registered tool.

        This MVP implementation does not perform argument validation or
        sandboxing; those concerns will be addressed in future iterations.
        """

        spec = self._tools.get(name)
        if spec is None:
            raise KeyError(f"Tool '{name}' is not registered.")
        return spec.func(*args, **kwargs)


def create_default_executor() -> ToolExecutor:
    """Create a tool executor with the default tool set registered."""

    from .tools.molsysmt_tools import get_info

    executor = ToolExecutor()
    executor.register(
        name="molsysmt.get_info",
        func=get_info,
        description="Load a molecular system from a PDB ID and get basic information.",
    )
    return executor

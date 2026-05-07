"""Tools layer — Phase 1 stub.

The harness treats retrieval as a Tools-layer concern. In Phase 1 the
client is a stub: it can advertise tool names (so orchestration patterns
can mention them in prompts) but cannot invoke them. Phase 2 wires real
MCP servers behind the same protocol.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class ToolClient(Protocol):
    """A minimal tool-access interface."""

    def list_tools(self) -> list[str]: ...
    def invoke(self, *, name: str, args: dict[str, Any]) -> Any: ...


@dataclass
class StubToolClient:
    """Phase 1 no-op tool client.

    Advertised names are returned by list_tools() — useful for testing that
    orchestration patterns include tool specs in prompts. Calling invoke()
    raises NotImplementedError to make missing real-MCP wiring visible.
    """

    advertised: list[str] = field(default_factory=list)

    def list_tools(self) -> list[str]:
        return list(self.advertised)

    def invoke(self, *, name: str, args: dict[str, Any]) -> Any:
        raise NotImplementedError(
            f"StubToolClient.invoke({name!r}) — real tool calls land in Phase 2"
        )

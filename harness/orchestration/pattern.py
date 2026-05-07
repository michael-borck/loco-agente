"""OrchestrationPattern — the protocol every pattern implements."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from harness.conversation import Conversation
from harness.inference.client import InferenceClient

if TYPE_CHECKING:
    from harness.context.bundle import ContextBundle


class OrchestrationPattern(Protocol):
    """A composable pattern that produces a Conversation trace.

    Patterns differ in how they sequence calls to the E primitive
    (SinglePass, multi-round Debate, source-grounded Synthesis,
    iterative refinement). Each pattern uses the same primitive contract.
    """

    def run(
        self,
        *,
        brief: str,
        client: InferenceClient,
        profile_name: str,
        context: "ContextBundle | None" = None,
    ) -> Conversation: ...

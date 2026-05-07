"""OrchestrationPattern — the protocol every pattern implements."""
from __future__ import annotations

from typing import Protocol

from harness.conversation import Conversation
from harness.inference.client import InferenceClient


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
    ) -> Conversation: ...

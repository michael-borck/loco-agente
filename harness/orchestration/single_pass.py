"""SinglePass — the smallest orchestration pattern.

One call to the E primitive, return all variants in one Round. Useful as
a building block and as a baseline for studies comparing more complex
patterns.
"""
from __future__ import annotations

from dataclasses import dataclass

from harness.conversation import Conversation, Round
from harness.core import generate_variants
from harness.frames import FrameStrategy
from harness.inference.client import InferenceClient


@dataclass
class SinglePass:
    """One batch of N variants from one E-primitive call."""

    frame_strategy: FrameStrategy
    n: int
    rule_names: list[str]

    def run(
        self,
        *,
        brief: str,
        client: InferenceClient,
        profile_name: str,
    ) -> Conversation:
        # Phase 1: rules are not loaded from a ContextBundle yet. The CLI
        # composes them when it calls the pattern; here rule_names is just
        # a label list. See Task 16 for full ContextBundle integration.
        variants = generate_variants(
            prompt=brief,
            n=self.n,
            frame_strategy=self.frame_strategy,
            client=client,
            rules=[],  # rules wired by CLI in Task 16
        )
        run_id = variants[0].metadata["run_id"]
        return Conversation(
            run_id=run_id,
            brief=brief,
            pattern="SinglePass",
            profile=profile_name,
            rounds=[Round(round_index=0, variants=variants)],
        )

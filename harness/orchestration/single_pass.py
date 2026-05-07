"""SinglePass — the smallest orchestration pattern.

One call to the E primitive, return all variants in one Round. Useful as
a building block and as a baseline for studies comparing more complex
patterns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from harness.conversation import Conversation, Round
from harness.core import generate_variants
from harness.frames import FrameStrategy
from harness.inference.client import InferenceClient

if TYPE_CHECKING:
    from harness.context.bundle import ContextBundle


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
        context: "ContextBundle | None" = None,
    ) -> Conversation:
        rules = (
            context.select_rules(self.rule_names)
            if context is not None and self.rule_names
            else []
        )
        variants = generate_variants(
            prompt=brief,
            n=self.n,
            frame_strategy=self.frame_strategy,
            client=client,
            rules=rules,
        )
        run_id = variants[0].metadata["run_id"]
        return Conversation(
            run_id=run_id,
            brief=brief,
            pattern="SinglePass",
            profile=profile_name,
            rounds=[Round(round_index=0, variants=variants)],
        )

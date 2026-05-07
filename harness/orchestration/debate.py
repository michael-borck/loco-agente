"""DebatePattern — N framed agents debate across R rounds.

Each round, every frame produces one variant. From round 2 onward, every
frame's prompt is augmented with the prior rounds' variants as context,
so subsequent frames can respond, refine, or push back.
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


def _format_prior_rounds(rounds: list[Round]) -> str:
    """Render prior rounds as a 'PRIOR ROUNDS:' block for downstream prompts."""
    lines: list[str] = ["PRIOR ROUNDS:"]
    for r in rounds:
        lines.append(f"\nRound {r.round_index}:")
        for v in r.variants:
            frame = v.metadata.get("frame_name", "?")
            lines.append(f"  [{frame}]: {v.text}")
            lines.append(f"    rationale: {v.rationale}")
    return "\n".join(lines)


@dataclass
class DebatePattern:
    """Multi-round multi-frame debate using the E primitive each round."""

    frame_strategy: FrameStrategy
    n: int
    rounds: int
    rule_names: list[str]

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError(f"rounds must be >= 1; got {self.rounds}")

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
        accumulated_rounds: list[Round] = []
        run_id: str | None = None

        for round_index in range(self.rounds):
            if accumulated_rounds:
                prior_block = _format_prior_rounds(accumulated_rounds)
                augmented_brief = f"{brief}\n\n{prior_block}"
            else:
                augmented_brief = brief

            variants = generate_variants(
                prompt=augmented_brief,
                n=self.n,
                frame_strategy=self.frame_strategy,
                client=client,
                rules=rules,
                run_id=run_id,
            )
            if run_id is None:
                run_id = variants[0].metadata["run_id"]
            accumulated_rounds.append(
                Round(round_index=round_index, variants=variants)
            )

        assert run_id is not None
        return Conversation(
            run_id=run_id,
            brief=brief,
            pattern="DebatePattern",
            profile=profile_name,
            rounds=accumulated_rounds,
        )

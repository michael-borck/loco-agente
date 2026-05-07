"""IterativeRefinement — variants → critique → re-variants → repeat."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from harness.conversation import Conversation, Round
from harness.core import Variant, generate_variants
from harness.frames import FrameStrategy
from harness.inference.client import InferenceClient


CritiqueFn = Callable[[list[Variant]], str]


@dataclass
class IterativeRefinement:
    """Repeatedly produce variants, gather critique, refine.

    critique_fn takes the latest round's variants and returns a critique
    string. Phase 1 supports any callable (a human prompt UI, an
    automatic rule check, or a fixed string). The critique is appended
    to the brief in subsequent rounds.
    """

    frame_strategy: FrameStrategy
    n: int
    max_iterations: int
    critique_fn: CritiqueFn
    rule_names: list[str]

    def run(
        self,
        *,
        brief: str,
        client: InferenceClient,
        profile_name: str,
    ) -> Conversation:
        accumulated: list[Round] = []
        run_id: str | None = None
        critique_history: list[str] = []

        for iteration in range(self.max_iterations):
            if critique_history:
                critique_block = "\n".join(
                    f"Critique after round {i}: {c}"
                    for i, c in enumerate(critique_history)
                )
                augmented = f"{brief}\n\n{critique_block}"
            else:
                augmented = brief

            variants = generate_variants(
                prompt=augmented,
                n=self.n,
                frame_strategy=self.frame_strategy,
                client=client,
                rules=[],
                run_id=run_id,
            )
            if run_id is None:
                run_id = variants[0].metadata["run_id"]

            accumulated.append(
                Round(round_index=iteration, variants=variants)
            )

            if iteration < self.max_iterations - 1:
                critique = self.critique_fn(variants)
                critique_history.append(critique)

        assert run_id is not None
        return Conversation(
            run_id=run_id,
            brief=brief,
            pattern="IterativeRefinement",
            profile=profile_name,
            rounds=accumulated,
        )

"""Conversation trace — the structured artifact of a harness run.

Every interaction with the harness produces a Conversation: which brief,
which pattern, which profile, every variant in every round, every user
decision. Serialised to JSON for persistence and downstream studies.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from harness.core import Uncertainty, Variant


VALID_DECISION_EVENTS = ("pick", "reject_all", "edit", "ask_more")


@dataclass
class UserDecision:
    """One user-facing decision recorded against the trace."""

    event: str
    variant_index: int
    round_index: int
    note: str = ""

    def __post_init__(self) -> None:
        if self.event not in VALID_DECISION_EVENTS:
            raise ValueError(
                f"event must be one of {VALID_DECISION_EVENTS}; got {self.event!r}"
            )


@dataclass
class Round:
    """One round of variant generation within a Conversation."""

    round_index: int
    variants: list[Variant]


@dataclass
class Conversation:
    """The full trace of one harness run."""

    run_id: str
    brief: str
    pattern: str
    profile: str
    rounds: list[Round] = field(default_factory=list)
    user_decisions: list[UserDecision] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "Conversation":
        data = json.loads(raw)
        rounds = [
            Round(
                round_index=r["round_index"],
                variants=[
                    Variant(
                        text=v["text"],
                        rationale=v["rationale"],
                        uncertainty=Uncertainty(**v["uncertainty"]),
                        metadata=v.get("metadata", {}),
                    )
                    for v in r["variants"]
                ],
            )
            for r in data.get("rounds", [])
        ]
        decisions = [
            UserDecision(**d) for d in data.get("user_decisions", [])
        ]
        return cls(
            run_id=data["run_id"],
            brief=data["brief"],
            pattern=data["pattern"],
            profile=data["profile"],
            rounds=rounds,
            user_decisions=decisions,
            metadata=data.get("metadata", {}),
        )

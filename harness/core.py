"""Core types and the E primitive for the LocoAgente harness.

This module defines the foundational contract every layer above must respect:
- Variant carries text + rationale + surfaced uncertainty.
- Uncertainty has load-bearing fields (flags, verification_hooks) and an
  auxiliary self-reported confidence treated as a research artifact.

See spec §2 for the full contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Uncertainty:
    """Surfaced uncertainty signals on a Variant.

    Load-bearing fields (UI surfaces these prominently):
      flags              — known failure modes the model acknowledges
      verification_hooks — what the human should check

    Auxiliary signal (research artifact, not load-bearing):
      confidence         — self-reported 0.0-1.0; famously miscalibrated on
                           small models. Logged but not used to gate decisions.
      relative_rank      — optional rank within a batch (1 = highest); more
                           informative than absolute confidence when available.
    """

    confidence: float
    flags: list[str]
    verification_hooks: list[str]
    relative_rank: int | None = None


@dataclass
class Variant:
    """One generated alternative produced by the E primitive.

    Always plural at the primitive level (n >= 2 enforced in generate_variants).
    Always carries a rationale (no bare outputs allowed).
    Always carries surfaced uncertainty.
    """

    text: str
    rationale: str
    uncertainty: Uncertainty
    metadata: dict[str, Any] = field(default_factory=dict)

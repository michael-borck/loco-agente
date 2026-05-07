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


import re


class VariantParseError(ValueError):
    """Raised when a model response cannot be parsed into a Variant."""


_TAG_PATTERNS: dict[str, re.Pattern[str]] = {
    "text": re.compile(r"<text>\s*(.*?)\s*</text>", re.DOTALL),
    "rationale": re.compile(r"<rationale>\s*(.*?)\s*</rationale>", re.DOTALL),
    "confidence": re.compile(r"<confidence>\s*(.*?)\s*</confidence>", re.DOTALL),
    "flags": re.compile(r"<flags>\s*(.*?)\s*</flags>", re.DOTALL),
    "verification": re.compile(
        r"<verification>\s*(.*?)\s*</verification>", re.DOTALL
    ),
}


def _extract_tag(raw: str, tag: str) -> str | None:
    m = _TAG_PATTERNS[tag].search(raw)
    return m.group(1) if m else None


def _split_semicolons(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(";") if item.strip()]


def parse_variant(raw: str, *, frame_name: str) -> Variant:
    """Parse a model XML-tagged response into a Variant.

    Required tags: <text>, <rationale>.
    Optional tags: <confidence> (default 0.5), <flags> (default empty),
    <verification> (default empty).

    Raises VariantParseError when required tags are missing or confidence
    cannot be parsed as a float.
    """
    text = _extract_tag(raw, "text")
    if text is None:
        raise VariantParseError(f"missing required <text> tag in: {raw[:200]!r}")
    rationale = _extract_tag(raw, "rationale")
    if rationale is None:
        raise VariantParseError(
            f"missing required <rationale> tag in: {raw[:200]!r}"
        )

    confidence_raw = _extract_tag(raw, "confidence")
    if confidence_raw is None or confidence_raw == "":
        confidence = 0.5
    else:
        try:
            confidence = float(confidence_raw)
        except ValueError as e:
            raise VariantParseError(
                f"could not parse <confidence>{confidence_raw}</confidence> as float"
            ) from e
    confidence = max(0.0, min(1.0, confidence))

    flags_raw = _extract_tag(raw, "flags") or ""
    verification_raw = _extract_tag(raw, "verification") or ""

    return Variant(
        text=text,
        rationale=rationale,
        uncertainty=Uncertainty(
            confidence=confidence,
            flags=_split_semicolons(flags_raw),
            verification_hooks=_split_semicolons(verification_raw),
        ),
        metadata={"frame_name": frame_name},
    )

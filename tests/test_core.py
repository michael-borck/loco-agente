"""Tests for harness.core — Variant, Uncertainty, parser, generate_variants."""
from __future__ import annotations

import pytest

from harness.core import Uncertainty, Variant


def test_variant_dataclass_minimal_fields() -> None:
    v = Variant(
        text="hello",
        rationale="testing",
        uncertainty=Uncertainty(confidence=0.5, flags=[], verification_hooks=[]),
        metadata={"frame_name": "test"},
    )
    assert v.text == "hello"
    assert v.uncertainty.confidence == 0.5
    assert v.metadata["frame_name"] == "test"


def test_uncertainty_default_relative_rank_is_none() -> None:
    u = Uncertainty(confidence=0.7, flags=["a"], verification_hooks=["b"])
    assert u.relative_rank is None


def test_uncertainty_accepts_relative_rank() -> None:
    u = Uncertainty(
        confidence=0.7, relative_rank=2, flags=[], verification_hooks=[]
    )
    assert u.relative_rank == 2


from harness.core import VariantParseError, parse_variant


def test_parse_variant_well_formed(sample_variant_xml: str) -> None:
    v = parse_variant(sample_variant_xml, frame_name="ceo")

    assert "scale, speed, simplicity" in v.text
    assert "executive clarity" in v.rationale or "strategic" in v.rationale
    assert v.uncertainty.confidence == 0.7
    assert "strategic priorities are subjective" in v.uncertainty.flags
    assert any("speed claim" in h for h in v.uncertainty.verification_hooks)
    assert v.metadata["frame_name"] == "ceo"


def test_parse_variant_missing_text_raises() -> None:
    raw = (
        "<variant>\n"
        "<rationale>missing the text tag</rationale>\n"
        "<confidence>0.5</confidence>\n"
        "</variant>"
    )
    with pytest.raises(VariantParseError, match="text"):
        parse_variant(raw, frame_name="x")


def test_parse_variant_missing_rationale_raises() -> None:
    raw = (
        "<variant>\n"
        "<text>some answer</text>\n"
        "<confidence>0.5</confidence>\n"
        "</variant>"
    )
    with pytest.raises(VariantParseError, match="rationale"):
        parse_variant(raw, frame_name="x")


def test_parse_variant_missing_optional_fields_uses_defaults() -> None:
    raw = (
        "<variant>\n"
        "<text>answer</text>\n"
        "<rationale>because</rationale>\n"
        "</variant>"
    )
    v = parse_variant(raw, frame_name="x")
    assert v.uncertainty.confidence == 0.5
    assert v.uncertainty.flags == []
    assert v.uncertainty.verification_hooks == []


def test_parse_variant_invalid_confidence_raises() -> None:
    raw = (
        "<variant>\n"
        "<text>answer</text>\n"
        "<rationale>because</rationale>\n"
        "<confidence>not a number</confidence>\n"
        "</variant>"
    )
    with pytest.raises(VariantParseError, match="confidence"):
        parse_variant(raw, frame_name="x")


def test_parse_variant_clamps_confidence_to_unit_interval() -> None:
    raw = (
        "<variant>\n"
        "<text>answer</text>\n"
        "<rationale>because</rationale>\n"
        "<confidence>1.5</confidence>\n"
        "</variant>"
    )
    v = parse_variant(raw, frame_name="x")
    assert v.uncertainty.confidence == 1.0


def test_parse_variant_splits_semicolon_lists() -> None:
    raw = (
        "<variant>\n"
        "<text>answer</text>\n"
        "<rationale>because</rationale>\n"
        "<flags>flag one; flag two; flag three</flags>\n"
        "<verification>check A; check B</verification>\n"
        "</variant>"
    )
    v = parse_variant(raw, frame_name="x")
    assert v.uncertainty.flags == ["flag one", "flag two", "flag three"]
    assert v.uncertainty.verification_hooks == ["check A", "check B"]

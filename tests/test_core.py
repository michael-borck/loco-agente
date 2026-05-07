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


from pathlib import Path

from harness.core import generate_variants
from harness.frames import IdentityFrames
from harness.inference.client import FakeInferenceClient


def test_generate_variants_rejects_n_below_two(tmp_profile: Path) -> None:
    fake = FakeInferenceClient(responses=["a"])
    strategy = IdentityFrames(frames=["ceo"], profile_root=tmp_profile)

    with pytest.raises(ValueError, match="n must be >= 2"):
        generate_variants(
            prompt="x",
            n=1,
            frame_strategy=strategy,
            client=fake,
            rules=[],
        )


def test_generate_variants_returns_n_parsed_variants(
    tmp_profile: Path, fake_inference_responses: list[str]
) -> None:
    fake = FakeInferenceClient(responses=fake_inference_responses[:2])
    strategy = IdentityFrames(
        frames=["ceo", "legal"], profile_root=tmp_profile
    )

    variants = generate_variants(
        prompt="Should we ship now?",
        n=2,
        frame_strategy=strategy,
        client=fake,
        rules=[],
    )

    assert len(variants) == 2
    assert {v.metadata["frame_name"] for v in variants} == {"ceo", "legal"}
    for v in variants:
        assert v.text
        assert v.rationale
        assert 0.0 <= v.uncertainty.confidence <= 1.0


def test_generate_variants_passes_sampling_params_to_client(
    tmp_profile: Path,
) -> None:
    canned = (
        "<variant>\n<text>x</text>\n<rationale>y</rationale>\n</variant>"
    )
    fake = FakeInferenceClient(responses=[canned, canned, canned, canned])

    from harness.frames import TemperatureLadder

    strategy = TemperatureLadder(temperatures=[0.3, 0.7, 1.0, 1.3])
    generate_variants(
        prompt="x", n=4, frame_strategy=strategy, client=fake, rules=[]
    )

    temps_passed = [c["sampling_params"].get("temperature") for c in fake.calls]
    assert temps_passed == [0.3, 0.7, 1.0, 1.3]


def test_generate_variants_records_run_id_in_metadata(
    tmp_profile: Path, fake_inference_responses: list[str]
) -> None:
    fake2 = FakeInferenceClient(responses=fake_inference_responses[:2])
    strategy2 = IdentityFrames(
        frames=["ceo", "legal"], profile_root=tmp_profile
    )
    variants = generate_variants(
        prompt="x",
        n=2,
        frame_strategy=strategy2,
        client=fake2,
        rules=[],
        run_id="my-run-id",
    )
    assert all(v.metadata["run_id"] == "my-run-id" for v in variants)

"""Tests for harness.frames — FrameStrategy implementations."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.frames import (
    FrameStrategy,
    IdentityFrames,
    PromptSpec,
)


def test_promptspec_minimal_fields() -> None:
    spec = PromptSpec(text="hello", frame_name="ceo", sampling_params={})
    assert spec.text == "hello"
    assert spec.frame_name == "ceo"
    assert spec.sampling_params == {}


def test_identity_frames_emits_one_prompt_per_frame(tmp_profile: Path) -> None:
    strategy = IdentityFrames(
        frames=["ceo", "legal"],
        profile_root=tmp_profile,
    )
    specs = strategy.generate_prompt_specs(
        base_prompt="Should we ship the feature now?",
        n=2,
        rules=[(tmp_profile / "rules" / "never_fabricate.md").read_text()],
    )

    assert len(specs) == 2
    assert {s.frame_name for s in specs} == {"ceo", "legal"}
    ceo_spec = next(s for s in specs if s.frame_name == "ceo")
    assert "CEO" in ceo_spec.text or "ceo" in ceo_spec.text.lower()
    assert "Should we ship the feature now?" in ceo_spec.text
    assert "Never fabricate" in ceo_spec.text or "never_fabricate" in ceo_spec.text.lower()


def test_identity_frames_n_must_match_frame_count(tmp_profile: Path) -> None:
    strategy = IdentityFrames(
        frames=["ceo", "legal"],
        profile_root=tmp_profile,
    )
    with pytest.raises(ValueError, match="n must equal the number of frames"):
        strategy.generate_prompt_specs(base_prompt="x", n=3, rules=[])


def test_identity_frames_missing_personality_file_raises(tmp_profile: Path) -> None:
    strategy = IdentityFrames(
        frames=["nonexistent"],
        profile_root=tmp_profile,
    )
    with pytest.raises(FileNotFoundError, match="nonexistent"):
        strategy.generate_prompt_specs(base_prompt="x", n=1, rules=[])


from harness.frames import DisciplineFrames, TemperatureLadder


def test_temperature_ladder_emits_n_prompts_with_distinct_temps() -> None:
    strategy = TemperatureLadder(temperatures=[0.3, 0.7, 1.0, 1.3])
    specs = strategy.generate_prompt_specs(
        base_prompt="Tell me a story about a dragon", n=4, rules=[]
    )

    assert len(specs) == 4
    temps = [s.sampling_params["temperature"] for s in specs]
    assert temps == [0.3, 0.7, 1.0, 1.3]
    # All four prompts share the same text — only sampling differs
    assert len({s.text for s in specs}) == 1
    # Frame name encodes the temperature for traceability
    assert {s.frame_name for s in specs} == {"temp_0.3", "temp_0.7", "temp_1.0", "temp_1.3"}


def test_temperature_ladder_n_must_match_temp_count() -> None:
    strategy = TemperatureLadder(temperatures=[0.3, 0.7])
    with pytest.raises(ValueError, match="n must equal"):
        strategy.generate_prompt_specs(base_prompt="x", n=3, rules=[])


def test_discipline_frames_inserts_discipline_label() -> None:
    strategy = DisciplineFrames(
        disciplines=["systems biology", "operations research"]
    )
    specs = strategy.generate_prompt_specs(
        base_prompt="How would you approach reducing supply chain failures?",
        n=2,
        rules=[],
    )

    assert len(specs) == 2
    assert {s.frame_name for s in specs} == {
        "systems biology",
        "operations research",
    }
    biology_spec = next(s for s in specs if s.frame_name == "systems biology")
    assert "systems biology" in biology_spec.text.lower()
    assert "supply chain failures" in biology_spec.text


def test_discipline_frames_n_must_match_discipline_count() -> None:
    strategy = DisciplineFrames(disciplines=["x", "y"])
    with pytest.raises(ValueError, match="n must equal"):
        strategy.generate_prompt_specs(base_prompt="x", n=3, rules=[])


from harness.frames import ConstraintInversion


def test_constraint_inversion_emits_one_prompt_per_constraint() -> None:
    strategy = ConstraintInversion(
        constraints_to_flip=[
            "the timeline is fixed",
            "the budget cannot increase",
            "the team size is fixed",
        ]
    )
    specs = strategy.generate_prompt_specs(
        base_prompt="Plan the project launch",
        n=3,
        rules=[],
    )

    assert len(specs) == 3
    assert {s.frame_name for s in specs} == {
        "flipped: the timeline is fixed",
        "flipped: the budget cannot increase",
        "flipped: the team size is fixed",
    }
    for spec in specs:
        constraint_clause = spec.frame_name.removeprefix("flipped: ")
        assert constraint_clause in spec.text
        assert "Plan the project launch" in spec.text


def test_constraint_inversion_n_must_match_constraint_count() -> None:
    strategy = ConstraintInversion(constraints_to_flip=["a", "b"])
    with pytest.raises(ValueError, match="n must equal"):
        strategy.generate_prompt_specs(base_prompt="x", n=3, rules=[])

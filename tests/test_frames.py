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

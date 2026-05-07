"""Tests for harness.orchestration — composable patterns over the E primitive."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.conversation import Conversation
from harness.frames import IdentityFrames
from harness.inference.client import FakeInferenceClient
from harness.orchestration.single_pass import SinglePass


def test_single_pass_produces_one_round_with_n_variants(
    tmp_profile: Path, fake_inference_responses: list[str]
) -> None:
    fake = FakeInferenceClient(responses=fake_inference_responses[:2])
    pattern = SinglePass(
        frame_strategy=IdentityFrames(
            frames=["ceo", "legal"], profile_root=tmp_profile
        ),
        n=2,
        rule_names=[],
    )

    conv = pattern.run(brief="ship now?", client=fake, profile_name="test")

    assert isinstance(conv, Conversation)
    assert conv.pattern == "SinglePass"
    assert conv.profile == "test"
    assert conv.brief == "ship now?"
    assert len(conv.rounds) == 1
    assert len(conv.rounds[0].variants) == 2


def test_single_pass_records_run_id(
    tmp_profile: Path, fake_inference_responses: list[str]
) -> None:
    fake = FakeInferenceClient(responses=fake_inference_responses[:2])
    pattern = SinglePass(
        frame_strategy=IdentityFrames(
            frames=["ceo", "legal"], profile_root=tmp_profile
        ),
        n=2,
        rule_names=[],
    )
    conv = pattern.run(brief="x", client=fake, profile_name="test")

    assert conv.run_id
    assert conv.rounds[0].variants[0].metadata["run_id"] == conv.run_id


from harness.orchestration.debate import DebatePattern


def _build_round_responses(n_frames: int, n_rounds: int) -> list[str]:
    """Build canned variant responses for n_frames frames over n_rounds rounds."""
    out: list[str] = []
    for r in range(n_rounds):
        for f in range(n_frames):
            out.append(
                f"<variant>\n"
                f"<text>round {r} frame {f} answer</text>\n"
                f"<rationale>round {r} frame {f} rationale</rationale>\n"
                f"<confidence>0.5</confidence>\n"
                f"<flags></flags>\n"
                f"<verification></verification>\n"
                f"</variant>"
            )
    return out


def test_debate_pattern_produces_rounds_x_frames_variants(
    tmp_profile: Path,
) -> None:
    responses = _build_round_responses(n_frames=2, n_rounds=3)
    fake = FakeInferenceClient(responses=responses)
    pattern = DebatePattern(
        frame_strategy=IdentityFrames(
            frames=["ceo", "legal"], profile_root=tmp_profile
        ),
        n=2,
        rounds=3,
        rule_names=[],
    )

    conv = pattern.run(brief="ship?", client=fake, profile_name="test")

    assert conv.pattern == "DebatePattern"
    assert len(conv.rounds) == 3
    for r in conv.rounds:
        assert len(r.variants) == 2


def test_debate_pattern_later_rounds_see_earlier_variants_in_prompt(
    tmp_profile: Path,
) -> None:
    """Round 2 prompts must reference round 1 variants in their context."""
    responses = _build_round_responses(n_frames=2, n_rounds=2)
    fake = FakeInferenceClient(responses=responses)
    pattern = DebatePattern(
        frame_strategy=IdentityFrames(
            frames=["ceo", "legal"], profile_root=tmp_profile
        ),
        n=2,
        rounds=2,
        rule_names=[],
    )

    pattern.run(brief="ship?", client=fake, profile_name="test")

    round1_calls = fake.calls[:2]
    round2_calls = fake.calls[2:4]
    for c in round1_calls:
        assert "PRIOR ROUNDS" not in c["prompt"]
    for c in round2_calls:
        assert "PRIOR ROUNDS" in c["prompt"]
        assert "round 0 frame 0" in c["prompt"]
        assert "round 0 frame 1" in c["prompt"]


def test_debate_pattern_rejects_zero_or_negative_rounds(tmp_profile: Path) -> None:
    with pytest.raises(ValueError, match="rounds must be >= 1"):
        DebatePattern(
            frame_strategy=IdentityFrames(
                frames=["ceo"], profile_root=tmp_profile
            ),
            n=1,
            rounds=0,
            rule_names=[],
        )

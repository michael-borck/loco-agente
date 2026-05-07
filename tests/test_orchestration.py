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

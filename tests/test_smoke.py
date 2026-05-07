"""Smoke test: exercise the full pipeline against a fake inference backend.

Builds a temporary profile bundle, runs the Perspective Debate flow end-to-end,
and verifies that the resulting trace is well-formed.
"""
from __future__ import annotations

from pathlib import Path

from harness.context.bundle import load_bundle
from harness.conversation import Conversation
from harness.frames import IdentityFrames
from harness.inference.client import FakeInferenceClient
from harness.orchestration.debate import DebatePattern


def _canned_response(text: str) -> str:
    return (
        "<variant>\n"
        f"<text>{text}</text>\n"
        "<rationale>frame-specific rationale</rationale>\n"
        "<confidence>0.6</confidence>\n"
        "<flags>self-reported confidence is miscalibrated</flags>\n"
        "<verification>verify the headline claim</verification>\n"
        "</variant>"
    )


def test_full_debate_pipeline_against_fake_inference(tmp_path: Path) -> None:
    profile = tmp_path / "profiles" / "test"
    (profile / "personality").mkdir(parents=True)
    (profile / "rules").mkdir(parents=True)
    (profile / "personality" / "alpha.md").write_text("Alpha frame.")
    (profile / "personality" / "beta.md").write_text("Beta frame.")
    (profile / "rules" / "be_explicit.md").write_text(
        "Be explicit about disagreements."
    )

    fake = FakeInferenceClient(
        responses=[
            _canned_response("alpha r0"),
            _canned_response("beta r0"),
            _canned_response("alpha r1"),
            _canned_response("beta r1"),
        ]
    )
    bundle = load_bundle(profile)
    pattern = DebatePattern(
        frame_strategy=IdentityFrames(
            frames=["alpha", "beta"], profile_root=profile
        ),
        n=2,
        rounds=2,
        rule_names=["be_explicit"],
    )

    conv = pattern.run(
        brief="The brief",
        client=fake,
        profile_name="test",
        context=bundle,
    )

    assert isinstance(conv, Conversation)
    assert conv.pattern == "DebatePattern"
    assert len(conv.rounds) == 2
    assert all(len(r.variants) == 2 for r in conv.rounds)

    # Round 1 prompts should reference round 0 variants
    round1_prompts = [c["prompt"] for c in fake.calls[2:4]]
    for p in round1_prompts:
        assert "alpha r0" in p
        assert "beta r0" in p

    # Trace serialises and round-trips
    raw = conv.to_json()
    restored = Conversation.from_json(raw)
    assert restored.run_id == conv.run_id
    assert len(restored.rounds) == 2

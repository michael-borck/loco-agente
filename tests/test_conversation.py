"""Tests for harness.conversation — Conversation trace serialisation."""
from __future__ import annotations

import json

import pytest

from harness.conversation import Conversation, Round, UserDecision
from harness.core import Uncertainty, Variant


def _make_variant(text: str, frame: str) -> Variant:
    return Variant(
        text=text,
        rationale="rationale",
        uncertainty=Uncertainty(confidence=0.5, flags=[], verification_hooks=[]),
        metadata={"frame_name": frame, "run_id": "run-1"},
    )


def test_conversation_minimal_fields() -> None:
    conv = Conversation(
        run_id="run-1",
        brief="should we ship?",
        pattern="DebatePattern",
        profile="business",
        rounds=[],
        user_decisions=[],
    )
    assert conv.run_id == "run-1"
    assert conv.rounds == []


def test_conversation_round_holds_variants() -> None:
    v1 = _make_variant("CEO answer", "ceo")
    v2 = _make_variant("Legal answer", "legal")
    round_ = Round(round_index=0, variants=[v1, v2])
    assert round_.round_index == 0
    assert len(round_.variants) == 2


def test_conversation_to_json_roundtrip() -> None:
    v1 = _make_variant("text one", "ceo")
    v2 = _make_variant("text two", "legal")
    conv = Conversation(
        run_id="run-1",
        brief="should we ship?",
        pattern="DebatePattern",
        profile="business",
        rounds=[Round(round_index=0, variants=[v1, v2])],
        user_decisions=[
            UserDecision(event="pick", variant_index=0, round_index=0)
        ],
    )

    raw = conv.to_json()
    parsed = json.loads(raw)
    assert parsed["run_id"] == "run-1"
    assert parsed["pattern"] == "DebatePattern"
    assert len(parsed["rounds"]) == 1
    assert len(parsed["rounds"][0]["variants"]) == 2

    restored = Conversation.from_json(raw)
    assert restored.run_id == conv.run_id
    assert restored.brief == conv.brief
    assert len(restored.rounds) == 1
    assert restored.rounds[0].variants[0].text == "text one"
    assert restored.user_decisions[0].event == "pick"


def test_user_decision_event_validated() -> None:
    valid = UserDecision(event="pick", variant_index=0, round_index=0)
    assert valid.event == "pick"
    with pytest.raises(ValueError, match="event must be one of"):
        UserDecision(event="invalid", variant_index=0, round_index=0)

"""Tests for harness.context — bundle loader and calibration log."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.context.bundle import ContextBundle, load_bundle


def test_load_bundle_reads_personality_and_rules(tmp_profile: Path) -> None:
    bundle = load_bundle(tmp_profile)

    assert isinstance(bundle, ContextBundle)
    assert bundle.profile_root == tmp_profile
    assert "ceo" in bundle.personalities
    assert "legal" in bundle.personalities
    assert "CEO" in bundle.personalities["ceo"]
    assert "never_fabricate" in bundle.rules
    assert "Never fabricate" in bundle.rules["never_fabricate"]


def test_load_bundle_rules_returned_as_dict(tmp_profile: Path) -> None:
    bundle = load_bundle(tmp_profile)
    assert isinstance(bundle.rules, dict)
    assert all(isinstance(v, str) for v in bundle.rules.values())


def test_load_bundle_missing_profile_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="profile"):
        load_bundle(tmp_path / "does_not_exist")


def test_load_bundle_missing_personality_dir_is_ok(tmp_path: Path) -> None:
    """A profile may have rules but no personalities (e.g. for synthesis)."""
    root = tmp_path / "rules_only"
    (root / "rules").mkdir(parents=True)
    (root / "rules" / "rule_a.md").write_text("rule body")

    bundle = load_bundle(root)

    assert bundle.personalities == {}
    assert bundle.rules == {"rule_a": "rule body"}


def test_context_bundle_select_rules() -> None:
    bundle = ContextBundle(
        profile_root=Path("/x"),
        personalities={},
        rules={"a": "AAA", "b": "BBB", "c": "CCC"},
    )
    selected = bundle.select_rules(["a", "c"])
    assert selected == ["AAA", "CCC"]


def test_context_bundle_select_rules_unknown_raises() -> None:
    bundle = ContextBundle(
        profile_root=Path("/x"), personalities={}, rules={"a": "AAA"}
    )
    with pytest.raises(KeyError, match="unknown"):
        bundle.select_rules(["a", "unknown"])


import json

from harness.context.calibration import CalibrationLog


def test_calibration_log_records_pick(tmp_path: Path) -> None:
    log_path = tmp_path / "calibration.jsonl"
    log = CalibrationLog(path=log_path)

    log.record_pick(
        run_id="run-123",
        variant_index=2,
        confidence=0.7,
        frame_name="ceo",
    )

    lines = log_path.read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["event"] == "pick"
    assert entry["run_id"] == "run-123"
    assert entry["variant_index"] == 2
    assert entry["confidence"] == 0.7
    assert entry["frame_name"] == "ceo"
    assert "timestamp" in entry


def test_calibration_log_records_reject_all(tmp_path: Path) -> None:
    log_path = tmp_path / "calibration.jsonl"
    log = CalibrationLog(path=log_path)
    log.record_reject_all(run_id="run-456", reason="none felt right")

    entry = json.loads(log_path.read_text().splitlines()[0])
    assert entry["event"] == "reject_all"
    assert entry["run_id"] == "run-456"
    assert entry["reason"] == "none felt right"


def test_calibration_log_records_edit(tmp_path: Path) -> None:
    log_path = tmp_path / "calibration.jsonl"
    log = CalibrationLog(path=log_path)
    log.record_edit(
        run_id="run-789",
        variant_index=0,
        original_text="original",
        edited_text="edited and changed substantially",
    )

    entry = json.loads(log_path.read_text().splitlines()[0])
    assert entry["event"] == "edit"
    assert entry["variant_index"] == 0
    assert entry["original_length"] == len("original")
    assert entry["edited_length"] == len("edited and changed substantially")
    assert entry["edit_distance"] > 0


def test_calibration_log_appends_across_calls(tmp_path: Path) -> None:
    log_path = tmp_path / "calibration.jsonl"
    log = CalibrationLog(path=log_path)
    log.record_pick(run_id="r1", variant_index=0, confidence=0.5, frame_name="x")
    log.record_pick(run_id="r2", variant_index=1, confidence=0.8, frame_name="y")

    lines = log_path.read_text().splitlines()
    assert len(lines) == 2


def test_calibration_log_creates_parent_dir(tmp_path: Path) -> None:
    log_path = tmp_path / "deep" / "nested" / "calibration.jsonl"
    log = CalibrationLog(path=log_path)
    log.record_pick(run_id="r", variant_index=0, confidence=0.5, frame_name="x")
    assert log_path.exists()

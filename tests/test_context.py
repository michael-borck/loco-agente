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

"""Smoke tests for CLI commands."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from cli.locoagente import main


def test_main_help() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "debate" in result.output
    assert "synthesise" in result.output
    assert "log" in result.output


def test_debate_command_writes_conversation_trace(
    tmp_path: Path, fake_inference_responses: list[str]
) -> None:
    profile_root = tmp_path / "profiles" / "business"
    (profile_root / "personality").mkdir(parents=True)
    (profile_root / "rules").mkdir(parents=True)
    (profile_root / "personality" / "ceo.md").write_text("CEO voice.")
    (profile_root / "personality" / "legal.md").write_text("Legal voice.")
    (profile_root / "rules" / "never_fabricate.md").write_text("Don't make stuff up.")

    runner = CliRunner()
    output_path = tmp_path / "trace.json"

    from harness.inference.client import FakeInferenceClient
    with patch("cli.commands.debate.build_inference_client") as mock_build:
        mock_build.return_value = FakeInferenceClient(
            responses=fake_inference_responses[:2]
        )
        result = runner.invoke(
            main,
            [
                "debate",
                "--brief", "should we ship?",
                "--frames", "ceo,legal",
                "--rounds", "1",
                "--profile-root", str(profile_root),
                "--output", str(output_path),
            ],
        )

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    trace = json.loads(output_path.read_text())
    assert trace["brief"] == "should we ship?"
    assert trace["pattern"] == "DebatePattern"
    assert len(trace["rounds"]) == 1
    assert len(trace["rounds"][0]["variants"]) == 2


def test_synthesise_command_writes_trace_with_corpus(
    tmp_path: Path, fake_inference_responses: list[str]
) -> None:
    # Build minimal corpus
    corpus = tmp_path / "papers"
    bio = corpus / "biology"
    ops = corpus / "operations"
    bio.mkdir(parents=True)
    ops.mkdir(parents=True)
    (bio / "p1.md").write_text("Title: cellular automata\nAbstract: foo")
    (ops / "p1.md").write_text("Title: queueing models\nAbstract: bar")

    profile_root = tmp_path / "profiles" / "academic"
    (profile_root / "rules").mkdir(parents=True)
    (profile_root / "rules" / "cite_or_flag.md").write_text("Cite or flag.")

    runner = CliRunner()
    output_path = tmp_path / "trace.json"

    from harness.inference.client import FakeInferenceClient
    canned = (
        "<variant>\n<text>insight</text>\n<rationale>r</rationale>\n</variant>"
    )
    with patch("cli.commands.synthesise.build_inference_client") as mock_build:
        mock_build.return_value = FakeInferenceClient(
            responses=[canned, canned, canned, canned]
        )
        result = runner.invoke(
            main,
            [
                "synthesise",
                "--brief", "supply chain reliability",
                "--disciplines", "biology,operations",
                "--corpus-root", str(corpus),
                "--profile-root", str(profile_root),
                "--output", str(output_path),
            ],
        )

    assert result.exit_code == 0, result.output
    trace = json.loads(output_path.read_text())
    assert trace["pattern"] == "SynthesisPattern"
    assert len(trace["rounds"][0]["variants"]) == 2

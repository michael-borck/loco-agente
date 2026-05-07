"""Shared pytest fixtures for the LocoAgente test suite."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def tmp_profile(tmp_path: Path) -> Path:
    """Create a minimal profile bundle on disk for tests."""
    root = tmp_path / "profiles" / "test_bundle"
    (root / "personality").mkdir(parents=True)
    (root / "rules").mkdir(parents=True)
    (root / "personality" / "ceo.md").write_text(
        "# CEO frame\n\nYou are a CEO. Speak with executive clarity. "
        "Focus on strategic implications.\n"
    )
    (root / "personality" / "legal.md").write_text(
        "# Legal frame\n\nYou are legal counsel. Identify risk and compliance "
        "concerns. Be precise about what is verified vs what is uncertain.\n"
    )
    (root / "rules" / "never_fabricate.md").write_text(
        "# Never fabricate\n\nDo not invent citations, statistics, or quotes. "
        "If you do not know, say so via an uncertainty flag.\n"
    )
    return root


@pytest.fixture
def sample_variant_xml() -> str:
    """A well-formed variant in the XML output format the parser expects."""
    return (
        "<variant>\n"
        "<text>The answer is to focus on three things: scale, speed, simplicity.</text>\n"
        "<rationale>Speaking from a CEO frame, strategic clarity beats nuance.</rationale>\n"
        "<confidence>0.7</confidence>\n"
        "<flags>strategic priorities are subjective; may not generalise</flags>\n"
        "<verification>verify the speed claim with engineering; confirm market timing</verification>\n"
        "</variant>"
    )


@pytest.fixture
def fake_inference_responses() -> list[str]:
    """Canned responses for the FakeInferenceClient — three distinct frames."""
    return [
        (
            "<variant>\n"
            "<text>From the CEO seat: ship the simpler version, learn from the market, iterate.</text>\n"
            "<rationale>CEOs trade certainty for speed; speed compounds.</rationale>\n"
            "<confidence>0.8</confidence>\n"
            "<flags>strategic claim, not empirical</flags>\n"
            "<verification>verify market readiness</verification>\n"
            "</variant>"
        ),
        (
            "<variant>\n"
            "<text>From legal: shipping early without GDPR review exposes liability.</text>\n"
            "<rationale>Legal frame anchors on downside risk and compliance.</rationale>\n"
            "<confidence>0.9</confidence>\n"
            "<flags>jurisdiction-specific; varies by region</flags>\n"
            "<verification>confirm GDPR scope; check internal compliance review status</verification>\n"
            "</variant>"
        ),
        (
            "<variant>\n"
            "<text>From marketing: a soft launch lets us collect testimonials before full release.</text>\n"
            "<rationale>Marketing favours optionality and narrative-building.</rationale>\n"
            "<confidence>0.6</confidence>\n"
            "<flags>tactical assumption; depends on product fit</flags>\n"
            "<verification>verify ICP availability for soft-launch testimonials</verification>\n"
            "</variant>"
        ),
    ]

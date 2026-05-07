# Conversational Harness Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the LocoAgente conversational harness library, two demo applications (Perspective Debate end-to-end + Research Synthesis skeleton), three Context profile bundles (business, academic, writing), a CLI, and Phase 1 documentation — all running on a small local model via Ollama.

**Architecture:** A flat-layout Python package at the repo root: `harness/` (core library), `cli/` (CLI commands), `applications/` (demo wiring), `profiles/` (Context bundles as data), `tests/` (pytest). The harness library implements the four named subsystems from the spec — Orchestration, Context, Tools, Inference — built on the foundational E primitive (`generate_variants` returning rationale-bearing variants with surfaced uncertainty). The Astro docs site under `src/` is left untouched.

**Tech Stack:** Python 3.10+, `uv` for venv management, `openai` SDK for OpenAI-compatible HTTP (talks to Ollama / llama.cpp / vLLM / SGLang), `click` for CLI, `PyYAML` for config, `pytest` for tests, Ollama as the default local inference backend.

**Spec reference:** `docs/superpowers/specs/2026-05-07-conversational-harness-design.md`

---

## File structure

### New files (all under `loco-agente/`)

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Python package metadata + dependency groups |
| `harness/__init__.py` | top-level package marker |
| `harness/core.py` | `Variant`, `Uncertainty` dataclasses + `generate_variants` E primitive + variant-output parser |
| `harness/frames.py` | `FrameStrategy` protocol + `IdentityFrames`, `DisciplineFrames`, `TemperatureLadder`, `ConstraintInversion` implementations |
| `harness/conversation.py` | `Conversation` trace dataclass + JSON serialisation |
| `harness/inference/__init__.py` | inference subpackage marker |
| `harness/inference/client.py` | `InferenceClient` protocol + `OpenAICompatibleClient` + `FakeInferenceClient` for tests |
| `harness/context/__init__.py` | context subpackage marker |
| `harness/context/bundle.py` | `ContextBundle` dataclass + `load_bundle()` (loads files from `profiles/<name>/`) |
| `harness/context/calibration.py` | `CalibrationLog` for capturing user picks/rejects/edits |
| `harness/tools/__init__.py` | tools subpackage marker |
| `harness/tools/client.py` | `ToolClient` protocol + `StubToolClient` (Phase 1 no-op) |
| `harness/orchestration/__init__.py` | orchestration subpackage marker |
| `harness/orchestration/pattern.py` | `OrchestrationPattern` protocol |
| `harness/orchestration/single_pass.py` | `SinglePass` pattern |
| `harness/orchestration/debate.py` | `DebatePattern` (multi-round multi-frame) |
| `harness/orchestration/synthesis.py` | `SynthesisPattern` (corpus-grounded) |
| `harness/orchestration/iterative.py` | `IterativeRefinement` |
| `cli/__init__.py` | CLI subpackage marker |
| `cli/locoagente.py` | CLI entry point (`click` group) |
| `cli/commands/__init__.py` | commands subpackage marker |
| `cli/commands/debate.py` | `locoagente debate` |
| `cli/commands/synthesise.py` | `locoagente synthesise` |
| `cli/commands/log.py` | `locoagente log` (inspect Conversation traces) |
| `profiles/business/personality/{ceo,legal,marketing}.md` | business frame personalities |
| `profiles/business/rules/{never_fabricate,disagree_explicitly}.md` | business rules |
| `profiles/academic/personality/{methodologist,theorist,practitioner,cross_disciplinary_scout}.md` | academic frames |
| `profiles/academic/rules/{cite_or_flag,explicit_uncertainty_on_citations}.md` | academic rules |
| `profiles/writing/personality/{protagonist,antagonist,narrator}.md` | writing frames |
| `profiles/writing/rules/{stay_in_voice,no_self_referential}.md` | writing rules |
| `profiles/_common/output_format.md` | the variant output format spec (XML tags) — prepended to every prompt |
| `applications/__init__.py` | applications package marker |
| `applications/conversation/__init__.py` | conversation subpackage |
| `applications/conversation/perspective_debate/__init__.py` | demo package |
| `applications/conversation/perspective_debate/runner.py` | demo runner (composes `DebatePattern` + business profile) |
| `applications/conversation/research_synthesis/__init__.py` | demo package |
| `applications/conversation/research_synthesis/runner.py` | demo runner (composes `SynthesisPattern` + academic profile) |
| `applications/conversation/research_synthesis/sample_papers/{biology,operations,history,linguistics}/*.md` | curated paper corpus |
| `tests/__init__.py` | test package marker |
| `tests/conftest.py` | shared pytest fixtures |
| `tests/test_core.py` | tests for `Variant`, `Uncertainty`, parser, `generate_variants` |
| `tests/test_frames.py` | tests for the four `FrameStrategy` impls |
| `tests/test_conversation.py` | tests for `Conversation` serialisation |
| `tests/test_inference.py` | tests for `OpenAICompatibleClient` (mocked) and `FakeInferenceClient` |
| `tests/test_context.py` | tests for `ContextBundle` loading and `CalibrationLog` |
| `tests/test_tools.py` | tests for `StubToolClient` |
| `tests/test_orchestration.py` | tests for the four orchestration patterns |
| `tests/test_cli.py` | smoke tests for CLI commands |
| `docs/philosophy/conversation-not-delegation.md` | philosophy doc 1 |
| `docs/philosophy/creativity-needs-variance-not-precision.md` | philosophy doc 2 |
| `docs/philosophy/confidence-is-not-competence.md` | philosophy doc 3 |
| `docs/architecture/four-subsystems.md` | named-not-numbered architecture overview |
| `docs/tutorials/perspective-debate.md` | tutorial 1 |
| `docs/tutorials/research-synthesis.md` | tutorial 2 |
| `docs/cli-reference.md` | CLI reference |

### Modified files

| Path | Change |
|---|---|
| `README.md` | replace track A/B/C/D narrative with conversational-harness mission (refs the spec); link tutorials |
| `.gitignore` | add `.venv/`, `__pycache__/`, `*.egg-info/`, `htmlcov/`, `.pytest_cache/` |

### Existing files NOT touched

`src/` (Astro docs site), `astro.config.mjs`, `package.json`, `tsconfig.json`, `public/`, `node_modules/`, `experiments/`, `LICENSE`. The Python harness coexists with the Astro docs site; neither touches the other.

---

## Phase 0: Bootstrap

### Task 0: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `harness/__init__.py`, `harness/inference/__init__.py`, `harness/context/__init__.py`, `harness/tools/__init__.py`, `harness/orchestration/__init__.py`
- Create: `cli/__init__.py`, `cli/commands/__init__.py`
- Create: `applications/__init__.py`, `applications/conversation/__init__.py`, `applications/conversation/perspective_debate/__init__.py`, `applications/conversation/research_synthesis/__init__.py`
- Create: `tests/__init__.py`, `tests/conftest.py`
- Modify: `.gitignore`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "locoagente"
version = "0.1.0"
description = "Conversational harness for small local models"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [
    { name = "Michael Borck", email = "michael@borck.id.au" },
]
keywords = ["agentic", "harness", "small-language-models", "local-ai", "consumer-hardware"]
dependencies = [
    "openai>=1.0",
    "PyYAML>=6.0",
    "click>=8.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.0",
]

[project.scripts]
locoagente = "cli.locoagente:main"

[tool.setuptools.packages.find]
include = ["harness*", "cli*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-ra"

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"
```

- [ ] **Step 2: Create the venv with uv and install deps**

```bash
cd /Users/michael/Projects/loco-lab/loco-agente
uv venv .venv
uv pip install -e '.[dev]' --python .venv/bin/python
.venv/bin/python -m pytest --version
```

Expected: pytest version printed, no errors.

- [ ] **Step 3: Create empty `__init__.py` markers and a top-level docstring for the harness package**

Create `harness/__init__.py` with:

```python
"""LocoAgente conversational harness.

Four-subsystem architecture for small local models: Orchestration, Context,
Tools, Inference. Built on the E primitive (generate_variants).
See docs/superpowers/specs/2026-05-07-conversational-harness-design.md.
"""

__version__ = "0.1.0"
```

Create empty `__init__.py` (zero bytes is fine) for these packages:
- `harness/inference/__init__.py`
- `harness/context/__init__.py`
- `harness/tools/__init__.py`
- `harness/orchestration/__init__.py`
- `cli/__init__.py`
- `cli/commands/__init__.py`
- `applications/__init__.py`
- `applications/conversation/__init__.py`
- `applications/conversation/perspective_debate/__init__.py`
- `applications/conversation/research_synthesis/__init__.py`
- `tests/__init__.py`

- [ ] **Step 4: Create `tests/conftest.py` with shared fixtures**

```python
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
```

- [ ] **Step 5: Update `.gitignore`**

Read the existing `.gitignore` (if present) and append (or create) with:

```
# Python
.venv/
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
htmlcov/
.coverage

# Build artifacts
build/
dist/
```

- [ ] **Step 6: Verify pytest collection works on empty test suite**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: `no tests ran` or `0 collected` — packages import cleanly.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml harness/ cli/ applications/ tests/ .gitignore
git commit -m "Bootstrap LocoAgente Python package layout and dependencies"
```

---

## Phase 1: Core data types

### Task 1: `Variant` and `Uncertainty` dataclasses

**Files:**
- Create: `harness/core.py`
- Test: `tests/test_core.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_core.py`:

```python
"""Tests for harness.core — Variant, Uncertainty, parser, generate_variants."""
from __future__ import annotations

import pytest

from harness.core import Uncertainty, Variant


def test_variant_dataclass_minimal_fields() -> None:
    v = Variant(
        text="hello",
        rationale="testing",
        uncertainty=Uncertainty(confidence=0.5, flags=[], verification_hooks=[]),
        metadata={"frame_name": "test"},
    )
    assert v.text == "hello"
    assert v.uncertainty.confidence == 0.5
    assert v.metadata["frame_name"] == "test"


def test_uncertainty_default_relative_rank_is_none() -> None:
    u = Uncertainty(confidence=0.7, flags=["a"], verification_hooks=["b"])
    assert u.relative_rank is None


def test_uncertainty_accepts_relative_rank() -> None:
    u = Uncertainty(
        confidence=0.7, relative_rank=2, flags=[], verification_hooks=[]
    )
    assert u.relative_rank == 2
```

- [ ] **Step 2: Verify failing**

```bash
.venv/bin/python -m pytest tests/test_core.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'harness.core'`.

- [ ] **Step 3: Implement `harness/core.py` (data types only — parser and `generate_variants` come in later tasks)**

```python
"""Core types and the E primitive for the LocoAgente harness.

This module defines the foundational contract every layer above must respect:
- Variant carries text + rationale + surfaced uncertainty.
- Uncertainty has load-bearing fields (flags, verification_hooks) and an
  auxiliary self-reported confidence treated as a research artifact.

See spec §2 for the full contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Uncertainty:
    """Surfaced uncertainty signals on a Variant.

    Load-bearing fields (UI surfaces these prominently):
      flags              — known failure modes the model acknowledges
      verification_hooks — what the human should check

    Auxiliary signal (research artifact, not load-bearing):
      confidence         — self-reported 0.0-1.0; famously miscalibrated on
                           small models. Logged but not used to gate decisions.
      relative_rank      — optional rank within a batch (1 = highest); more
                           informative than absolute confidence when available.
    """

    confidence: float
    flags: list[str]
    verification_hooks: list[str]
    relative_rank: int | None = None


@dataclass
class Variant:
    """One generated alternative produced by the E primitive.

    Always plural at the primitive level (n >= 2 enforced in generate_variants).
    Always carries a rationale (no bare outputs allowed).
    Always carries surfaced uncertainty.
    """

    text: str
    rationale: str
    uncertainty: Uncertainty
    metadata: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/test_core.py -v
```

Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add harness/core.py tests/test_core.py
git commit -m "Add Variant and Uncertainty dataclasses for the E primitive"
```

---

### Task 2: Variant XML parser

**Files:**
- Modify: `harness/core.py` (append parser)
- Modify: `tests/test_core.py` (append parser tests)

- [ ] **Step 1: Append failing tests to `tests/test_core.py`**

Append to `tests/test_core.py`:

```python
from harness.core import VariantParseError, parse_variant


def test_parse_variant_well_formed(sample_variant_xml: str) -> None:
    v = parse_variant(sample_variant_xml, frame_name="ceo")

    assert "scale, speed, simplicity" in v.text
    assert "executive clarity" in v.rationale or "strategic" in v.rationale
    assert v.uncertainty.confidence == 0.7
    assert "strategic priorities are subjective" in v.uncertainty.flags
    assert any("speed claim" in h for h in v.uncertainty.verification_hooks)
    assert v.metadata["frame_name"] == "ceo"


def test_parse_variant_missing_text_raises() -> None:
    raw = (
        "<variant>\n"
        "<rationale>missing the text tag</rationale>\n"
        "<confidence>0.5</confidence>\n"
        "</variant>"
    )
    with pytest.raises(VariantParseError, match="text"):
        parse_variant(raw, frame_name="x")


def test_parse_variant_missing_rationale_raises() -> None:
    raw = (
        "<variant>\n"
        "<text>some answer</text>\n"
        "<confidence>0.5</confidence>\n"
        "</variant>"
    )
    with pytest.raises(VariantParseError, match="rationale"):
        parse_variant(raw, frame_name="x")


def test_parse_variant_missing_optional_fields_uses_defaults() -> None:
    raw = (
        "<variant>\n"
        "<text>answer</text>\n"
        "<rationale>because</rationale>\n"
        "</variant>"
    )
    v = parse_variant(raw, frame_name="x")
    assert v.uncertainty.confidence == 0.5  # default
    assert v.uncertainty.flags == []
    assert v.uncertainty.verification_hooks == []


def test_parse_variant_invalid_confidence_raises() -> None:
    raw = (
        "<variant>\n"
        "<text>answer</text>\n"
        "<rationale>because</rationale>\n"
        "<confidence>not a number</confidence>\n"
        "</variant>"
    )
    with pytest.raises(VariantParseError, match="confidence"):
        parse_variant(raw, frame_name="x")


def test_parse_variant_clamps_confidence_to_unit_interval() -> None:
    raw = (
        "<variant>\n"
        "<text>answer</text>\n"
        "<rationale>because</rationale>\n"
        "<confidence>1.5</confidence>\n"
        "</variant>"
    )
    v = parse_variant(raw, frame_name="x")
    assert v.uncertainty.confidence == 1.0


def test_parse_variant_splits_semicolon_lists() -> None:
    raw = (
        "<variant>\n"
        "<text>answer</text>\n"
        "<rationale>because</rationale>\n"
        "<flags>flag one; flag two; flag three</flags>\n"
        "<verification>check A; check B</verification>\n"
        "</variant>"
    )
    v = parse_variant(raw, frame_name="x")
    assert v.uncertainty.flags == ["flag one", "flag two", "flag three"]
    assert v.uncertainty.verification_hooks == ["check A", "check B"]
```

- [ ] **Step 2: Verify failing**

```bash
.venv/bin/python -m pytest tests/test_core.py -v
```

Expected: FAIL with `ImportError: cannot import name 'parse_variant'` (and `VariantParseError`).

- [ ] **Step 3: Append parser to `harness/core.py`**

Append to `harness/core.py`:

```python
import re


class VariantParseError(ValueError):
    """Raised when a model response cannot be parsed into a Variant."""


_TAG_PATTERNS: dict[str, re.Pattern[str]] = {
    "text": re.compile(r"<text>\s*(.*?)\s*</text>", re.DOTALL),
    "rationale": re.compile(r"<rationale>\s*(.*?)\s*</rationale>", re.DOTALL),
    "confidence": re.compile(r"<confidence>\s*(.*?)\s*</confidence>", re.DOTALL),
    "flags": re.compile(r"<flags>\s*(.*?)\s*</flags>", re.DOTALL),
    "verification": re.compile(
        r"<verification>\s*(.*?)\s*</verification>", re.DOTALL
    ),
}


def _extract_tag(raw: str, tag: str) -> str | None:
    m = _TAG_PATTERNS[tag].search(raw)
    return m.group(1) if m else None


def _split_semicolons(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(";") if item.strip()]


def parse_variant(raw: str, *, frame_name: str) -> Variant:
    """Parse a model XML-tagged response into a Variant.

    Required tags: <text>, <rationale>.
    Optional tags: <confidence> (default 0.5), <flags> (default empty),
    <verification> (default empty).

    Raises VariantParseError when required tags are missing or confidence
    cannot be parsed as a float.
    """
    text = _extract_tag(raw, "text")
    if text is None:
        raise VariantParseError(f"missing required <text> tag in: {raw[:200]!r}")
    rationale = _extract_tag(raw, "rationale")
    if rationale is None:
        raise VariantParseError(
            f"missing required <rationale> tag in: {raw[:200]!r}"
        )

    confidence_raw = _extract_tag(raw, "confidence")
    if confidence_raw is None or confidence_raw == "":
        confidence = 0.5
    else:
        try:
            confidence = float(confidence_raw)
        except ValueError as e:
            raise VariantParseError(
                f"could not parse <confidence>{confidence_raw}</confidence> as float"
            ) from e
    confidence = max(0.0, min(1.0, confidence))

    flags_raw = _extract_tag(raw, "flags") or ""
    verification_raw = _extract_tag(raw, "verification") or ""

    return Variant(
        text=text,
        rationale=rationale,
        uncertainty=Uncertainty(
            confidence=confidence,
            flags=_split_semicolons(flags_raw),
            verification_hooks=_split_semicolons(verification_raw),
        ),
        metadata={"frame_name": frame_name},
    )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/test_core.py -v
```

Expected: PASS (10 tests total: 3 from Task 1 + 7 new).

- [ ] **Step 5: Commit**

```bash
git add harness/core.py tests/test_core.py
git commit -m "Add Variant XML parser with required/optional tag handling"
```

---

## Phase 2: Frame strategies

### Task 3: `FrameStrategy` protocol + `IdentityFrames`

**Files:**
- Create: `harness/frames.py`
- Test: `tests/test_frames.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_frames.py`:

```python
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
    # Each prompt includes its personality content + base prompt + rules
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
```

- [ ] **Step 2: Verify failing**

```bash
.venv/bin/python -m pytest tests/test_frames.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `harness/frames.py`**

```python
"""Frame strategies — the variance-generation knob of the E primitive.

Variance is engineered, not hoped for. Each FrameStrategy turns one base
prompt into N deliberately differentiated PromptSpecs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class PromptSpec:
    """One framed prompt ready to send to an inference backend."""

    text: str
    frame_name: str
    sampling_params: dict[str, Any] = field(default_factory=dict)


class FrameStrategy(Protocol):
    """Produces N deliberately differentiated prompts from one base prompt."""

    def generate_prompt_specs(
        self,
        *,
        base_prompt: str,
        n: int,
        rules: list[str],
    ) -> list[PromptSpec]: ...


_OUTPUT_FORMAT_INSTRUCTIONS = """
Produce your response wrapped in these XML tags:

<variant>
<text>
[your actual response — the perspective, answer, or idea]
</text>

<rationale>
[why this response — the framing or reasoning that produced it]
</rationale>

<confidence>
[0.0-1.0, your self-rated confidence in this response]
</confidence>

<flags>
[semicolon-separated failure modes you acknowledge, e.g.: "citation may be hallucinated"; "outside training cutoff"]
</flags>

<verification>
[semicolon-separated things the human should check, e.g.: "verify the regulatory claim"; "confirm the date"]
</verification>
</variant>

Always include all five tags. If a tag has no content, leave it empty but include the tag.
""".strip()


def _compose_prompt(*, personality: str, rules: list[str], base_prompt: str) -> str:
    """Assemble a final prompt from personality + rules + base + output format."""
    sections: list[str] = [personality.strip()]
    for rule in rules:
        sections.append(rule.strip())
    sections.append(_OUTPUT_FORMAT_INSTRUCTIONS)
    sections.append(f"BRIEF: {base_prompt.strip()}")
    return "\n\n---\n\n".join(sections)


@dataclass
class IdentityFrames:
    """Identity-based frames: one PromptSpec per named frame.

    Loads personality content from <profile_root>/personality/<frame>.md.
    """

    frames: list[str]
    profile_root: Path

    def generate_prompt_specs(
        self,
        *,
        base_prompt: str,
        n: int,
        rules: list[str],
    ) -> list[PromptSpec]:
        if n != len(self.frames):
            raise ValueError(
                f"n must equal the number of frames; got n={n}, "
                f"frames={len(self.frames)}"
            )
        specs: list[PromptSpec] = []
        for frame in self.frames:
            path = self.profile_root / "personality" / f"{frame}.md"
            if not path.exists():
                raise FileNotFoundError(
                    f"personality file for frame {frame!r} not found at {path}"
                )
            personality = path.read_text()
            text = _compose_prompt(
                personality=personality, rules=rules, base_prompt=base_prompt
            )
            specs.append(
                PromptSpec(text=text, frame_name=frame, sampling_params={})
            )
        return specs
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_frames.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add harness/frames.py tests/test_frames.py
git commit -m "Add FrameStrategy protocol and IdentityFrames implementation"
```

---

### Task 4: `TemperatureLadder` and `DisciplineFrames`

**Files:**
- Modify: `harness/frames.py` (append two strategies)
- Modify: `tests/test_frames.py` (append tests)

- [ ] **Step 1: Append failing tests to `tests/test_frames.py`**

Append to `tests/test_frames.py`:

```python
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
```

- [ ] **Step 2: Verify failing**

```bash
.venv/bin/python -m pytest tests/test_frames.py -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Append `TemperatureLadder` and `DisciplineFrames` to `harness/frames.py`**

Append to `harness/frames.py`:

```python
@dataclass
class TemperatureLadder:
    """Vanilla diversity sampling: same prompt, varying sampling temperature.

    Less principled than identity/discipline frames but useful as a baseline.
    """

    temperatures: list[float]

    def generate_prompt_specs(
        self,
        *,
        base_prompt: str,
        n: int,
        rules: list[str],
    ) -> list[PromptSpec]:
        if n != len(self.temperatures):
            raise ValueError(
                f"n must equal the number of temperatures; got n={n}, "
                f"temperatures={len(self.temperatures)}"
            )
        text = _compose_prompt(
            personality="(no specific frame; vary by sampling)",
            rules=rules,
            base_prompt=base_prompt,
        )
        return [
            PromptSpec(
                text=text,
                frame_name=f"temp_{t}",
                sampling_params={"temperature": t},
            )
            for t in self.temperatures
        ]


@dataclass
class DisciplineFrames:
    """Cross-disciplinary scanning: one PromptSpec per discipline.

    The discipline label is inserted into a generic 'scout' prompt template;
    no personality file is required (unlike IdentityFrames). Use IdentityFrames
    when you want fully customised personalities; use DisciplineFrames for
    quick cross-disciplinary scans.
    """

    disciplines: list[str]

    def generate_prompt_specs(
        self,
        *,
        base_prompt: str,
        n: int,
        rules: list[str],
    ) -> list[PromptSpec]:
        if n != len(self.disciplines):
            raise ValueError(
                f"n must equal the number of disciplines; got n={n}, "
                f"disciplines={len(self.disciplines)}"
            )
        specs: list[PromptSpec] = []
        for discipline in self.disciplines:
            personality = (
                f"You are a researcher with deep training in {discipline}. "
                f"Read the brief through your discipline's analytical lens. "
                f"Surface connections, methods, or vocabulary your discipline "
                f"would bring to bear that an outsider might miss."
            )
            text = _compose_prompt(
                personality=personality, rules=rules, base_prompt=base_prompt
            )
            specs.append(
                PromptSpec(text=text, frame_name=discipline, sampling_params={})
            )
        return specs
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_frames.py -v
```

Expected: PASS (8 tests total).

- [ ] **Step 5: Commit**

```bash
git add harness/frames.py tests/test_frames.py
git commit -m "Add TemperatureLadder and DisciplineFrames strategies"
```

---

### Task 5: `ConstraintInversion`

**Files:**
- Modify: `harness/frames.py`
- Modify: `tests/test_frames.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_frames.py`:

```python
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
    # Each prompt explicitly mentions inverting its assigned constraint
    for spec in specs:
        constraint_clause = spec.frame_name.removeprefix("flipped: ")
        assert constraint_clause in spec.text
        assert "Plan the project launch" in spec.text


def test_constraint_inversion_n_must_match_constraint_count() -> None:
    strategy = ConstraintInversion(constraints_to_flip=["a", "b"])
    with pytest.raises(ValueError, match="n must equal"):
        strategy.generate_prompt_specs(base_prompt="x", n=3, rules=[])
```

- [ ] **Step 2: Verify failing**

```bash
.venv/bin/python -m pytest tests/test_frames.py -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Append `ConstraintInversion` to `harness/frames.py`**

```python
@dataclass
class ConstraintInversion:
    """Generate variants by deliberately inverting one constraint each.

    Surfaces which constraints are load-bearing by showing what happens when
    each is flipped. Useful for design reviews and assumption-testing.
    """

    constraints_to_flip: list[str]

    def generate_prompt_specs(
        self,
        *,
        base_prompt: str,
        n: int,
        rules: list[str],
    ) -> list[PromptSpec]:
        if n != len(self.constraints_to_flip):
            raise ValueError(
                f"n must equal the number of constraints; got n={n}, "
                f"constraints={len(self.constraints_to_flip)}"
            )
        specs: list[PromptSpec] = []
        for constraint in self.constraints_to_flip:
            personality = (
                f"For this response, deliberately invert the following "
                f"assumption from the brief: '{constraint}'. "
                f"Reason as if the opposite were true. Surface what changes "
                f"in the recommendation when this constraint is flipped."
            )
            text = _compose_prompt(
                personality=personality, rules=rules, base_prompt=base_prompt
            )
            specs.append(
                PromptSpec(
                    text=text,
                    frame_name=f"flipped: {constraint}",
                    sampling_params={},
                )
            )
        return specs
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_frames.py -v
```

Expected: PASS (10 tests total).

- [ ] **Step 5: Commit**

```bash
git add harness/frames.py tests/test_frames.py
git commit -m "Add ConstraintInversion frame strategy"
```

---

## Phase 3: Inference layer

### Task 6: `InferenceClient` protocol + `OpenAICompatibleClient` + `FakeInferenceClient`

**Files:**
- Create: `harness/inference/client.py`
- Test: `tests/test_inference.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_inference.py`:

```python
"""Tests for harness.inference.client."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from harness.inference.client import (
    FakeInferenceClient,
    InferenceClient,
    OpenAICompatibleClient,
)


def test_fake_client_returns_canned_responses_in_order() -> None:
    client = FakeInferenceClient(responses=["one", "two", "three"])
    assert client.complete("p1") == "one"
    assert client.complete("p2") == "two"
    assert client.complete("p3") == "three"


def test_fake_client_raises_when_exhausted() -> None:
    client = FakeInferenceClient(responses=["only one"])
    client.complete("p1")
    with pytest.raises(RuntimeError, match="exhausted"):
        client.complete("p2")


def test_fake_client_records_calls() -> None:
    client = FakeInferenceClient(responses=["a", "b"])
    client.complete("first prompt", temperature=0.7)
    client.complete("second prompt", temperature=0.3)
    assert len(client.calls) == 2
    assert client.calls[0]["prompt"] == "first prompt"
    assert client.calls[0]["sampling_params"] == {"temperature": 0.7}
    assert client.calls[1]["prompt"] == "second prompt"


@patch("harness.inference.client.OpenAI")
def test_openai_compatible_client_calls_completions_api(mock_openai_cls) -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="hello world"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_cls.return_value = mock_client

    client = OpenAICompatibleClient(
        base_url="http://localhost:11434/v1",
        api_key="not-needed",
        model="qwen3:4b",
    )
    result = client.complete("test prompt", temperature=0.5)

    assert result == "hello world"
    mock_openai_cls.assert_called_once_with(
        base_url="http://localhost:11434/v1", api_key="not-needed"
    )
    args, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["model"] == "qwen3:4b"
    assert kwargs["temperature"] == 0.5
    assert kwargs["messages"] == [{"role": "user", "content": "test prompt"}]


@patch("harness.inference.client.OpenAI")
def test_openai_compatible_client_omits_temperature_when_not_supplied(
    mock_openai_cls,
) -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="x"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_cls.return_value = mock_client

    client = OpenAICompatibleClient(
        base_url="http://x", api_key="x", model="m"
    )
    client.complete("p")

    _, kwargs = mock_client.chat.completions.create.call_args
    assert "temperature" not in kwargs
```

- [ ] **Step 2: Verify failing**

```bash
.venv/bin/python -m pytest tests/test_inference.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `harness/inference/client.py`**

```python
"""Inference layer — talks to OpenAI-compatible HTTP endpoints.

The harness is inference-engine-agnostic. This module wraps the OpenAI SDK
to talk to Ollama, llama.cpp server, vLLM, SGLang, or any other OpenAI-
compatible backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import OpenAI


class InferenceClient(Protocol):
    """A minimal completion interface — one prompt in, one string out."""

    def complete(self, prompt: str, **sampling_params: Any) -> str: ...


@dataclass
class OpenAICompatibleClient:
    """Talks to any OpenAI-compatible chat-completions endpoint."""

    base_url: str
    api_key: str
    model: str

    def complete(self, prompt: str, **sampling_params: Any) -> str:
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        for key, value in sampling_params.items():
            kwargs[key] = value
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


@dataclass
class FakeInferenceClient:
    """Test double that returns canned responses in order.

    Records every call for assertion against. Raises RuntimeError when the
    canned-response list is exhausted (so tests fail loudly on miswiring).
    """

    responses: list[str]
    calls: list[dict[str, Any]] = field(default_factory=list)
    _cursor: int = 0

    def complete(self, prompt: str, **sampling_params: Any) -> str:
        self.calls.append({"prompt": prompt, "sampling_params": sampling_params})
        if self._cursor >= len(self.responses):
            raise RuntimeError(
                f"FakeInferenceClient responses exhausted "
                f"(received {self._cursor + 1} calls, "
                f"only {len(self.responses)} responses configured)"
            )
        response = self.responses[self._cursor]
        self._cursor += 1
        return response
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_inference.py -v
```

Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add harness/inference/client.py tests/test_inference.py
git commit -m "Add InferenceClient protocol with OpenAI-compatible and Fake impls"
```

---

## Phase 4: Context layer

### Task 7: `ContextBundle` loader

**Files:**
- Create: `harness/context/bundle.py`
- Test: `tests/test_context.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_context.py`:

```python
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
```

- [ ] **Step 2: Verify failing**

```bash
.venv/bin/python -m pytest tests/test_context.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `harness/context/bundle.py`**

```python
"""Context layer — hierarchical instruction file loader.

A profile bundle is a directory of markdown files organised as:
    <profile_root>/
        personality/   — one file per frame (ceo.md, legal.md, …)
        rules/         — hard constraints (never_fabricate.md, …)
        memory/        — persistent context (optional)
        tools/         — MCP tool specs (optional)

This loader reads personality and rules into in-memory dicts keyed by
filename stem. Memory and tools are loaded by other modules as needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ContextBundle:
    """A loaded profile bundle ready for use by an OrchestrationPattern."""

    profile_root: Path
    personalities: dict[str, str] = field(default_factory=dict)
    rules: dict[str, str] = field(default_factory=dict)

    def select_rules(self, rule_names: list[str]) -> list[str]:
        """Return rule bodies in the requested order. Unknown names raise."""
        out: list[str] = []
        for name in rule_names:
            if name not in self.rules:
                raise KeyError(
                    f"unknown rule {name!r}; "
                    f"available: {sorted(self.rules.keys())}"
                )
            out.append(self.rules[name])
        return out


def load_bundle(profile_root: Path) -> ContextBundle:
    """Load all personality and rules files from a profile directory."""
    if not profile_root.exists():
        raise FileNotFoundError(f"profile root not found: {profile_root}")

    personality_dir = profile_root / "personality"
    rules_dir = profile_root / "rules"

    personalities: dict[str, str] = {}
    if personality_dir.is_dir():
        for path in sorted(personality_dir.glob("*.md")):
            personalities[path.stem] = path.read_text()

    rules: dict[str, str] = {}
    if rules_dir.is_dir():
        for path in sorted(rules_dir.glob("*.md")):
            rules[path.stem] = path.read_text()

    return ContextBundle(
        profile_root=profile_root,
        personalities=personalities,
        rules=rules,
    )
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_context.py -v
```

Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add harness/context/bundle.py tests/test_context.py
git commit -m "Add ContextBundle loader for profile bundles"
```

---

### Task 8: `CalibrationLog` for capturing user decisions

**Files:**
- Create: `harness/context/calibration.py`
- Modify: `tests/test_context.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_context.py`:

```python
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
```

- [ ] **Step 2: Verify failing**

```bash
.venv/bin/python -m pytest tests/test_context.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `harness/context/calibration.py`**

```python
"""Calibration capture — records user picks/rejects/edits as JSONL.

Every interaction is logged in machine-readable form so downstream studies
(see studies/context/) can analyse model self-confidence vs human-validated
outcomes. The data capture starts on day one; the analysis is downstream
research.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance — small implementation, no dependency."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if not b:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr_row = [i + 1]
        for j, cb in enumerate(b):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (ca != cb)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


@dataclass
class CalibrationLog:
    """Append-only JSONL log of user decisions over harness output."""

    path: Path

    def _append(self, entry: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry["timestamp"] = time.time()
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def record_pick(
        self,
        *,
        run_id: str,
        variant_index: int,
        confidence: float,
        frame_name: str,
    ) -> None:
        self._append(
            {
                "event": "pick",
                "run_id": run_id,
                "variant_index": variant_index,
                "confidence": confidence,
                "frame_name": frame_name,
            }
        )

    def record_reject_all(self, *, run_id: str, reason: str = "") -> None:
        self._append(
            {"event": "reject_all", "run_id": run_id, "reason": reason}
        )

    def record_edit(
        self,
        *,
        run_id: str,
        variant_index: int,
        original_text: str,
        edited_text: str,
    ) -> None:
        self._append(
            {
                "event": "edit",
                "run_id": run_id,
                "variant_index": variant_index,
                "original_length": len(original_text),
                "edited_length": len(edited_text),
                "edit_distance": _edit_distance(original_text, edited_text),
            }
        )
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_context.py -v
```

Expected: PASS (11 tests total).

- [ ] **Step 5: Commit**

```bash
git add harness/context/calibration.py tests/test_context.py
git commit -m "Add CalibrationLog for capturing user picks/rejects/edits"
```

---

## Phase 5: Tools layer (stub)

### Task 9: `ToolClient` protocol + `StubToolClient`

**Files:**
- Create: `harness/tools/client.py`
- Test: `tests/test_tools.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_tools.py`:

```python
"""Tests for harness.tools — Phase 1 stub."""
from __future__ import annotations

import pytest

from harness.tools.client import StubToolClient


def test_stub_tool_client_lists_no_tools() -> None:
    client = StubToolClient()
    assert client.list_tools() == []


def test_stub_tool_client_invoke_raises() -> None:
    client = StubToolClient()
    with pytest.raises(NotImplementedError, match="Phase 2"):
        client.invoke(name="search", args={"q": "x"})


def test_stub_tool_client_with_advertised_tools() -> None:
    """A stub may advertise tool names without implementing them — useful
    for testing that orchestration patterns surface tool specs to prompts."""
    client = StubToolClient(advertised=["search_corpus", "fetch_paper"])
    assert client.list_tools() == ["search_corpus", "fetch_paper"]
    with pytest.raises(NotImplementedError):
        client.invoke(name="search_corpus", args={"q": "x"})
```

- [ ] **Step 2: Verify failing**

```bash
.venv/bin/python -m pytest tests/test_tools.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `harness/tools/client.py`**

```python
"""Tools layer — Phase 1 stub.

The harness treats retrieval as a Tools-layer concern. In Phase 1 the
client is a stub: it can advertise tool names (so orchestration patterns
can mention them in prompts) but cannot invoke them. Phase 2 wires real
MCP servers behind the same protocol.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class ToolClient(Protocol):
    """A minimal tool-access interface."""

    def list_tools(self) -> list[str]: ...
    def invoke(self, *, name: str, args: dict[str, Any]) -> Any: ...


@dataclass
class StubToolClient:
    """Phase 1 no-op tool client.

    Advertised names are returned by list_tools() — useful for testing that
    orchestration patterns include tool specs in prompts. Calling invoke()
    raises NotImplementedError to make missing real-MCP wiring visible.
    """

    advertised: list[str] = field(default_factory=list)

    def list_tools(self) -> list[str]:
        return list(self.advertised)

    def invoke(self, *, name: str, args: dict[str, Any]) -> Any:
        raise NotImplementedError(
            f"StubToolClient.invoke({name!r}) — real tool calls land in Phase 2"
        )
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_tools.py -v
```

Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add harness/tools/client.py tests/test_tools.py
git commit -m "Add ToolClient protocol with Phase 1 StubToolClient"
```

---

## Phase 6: The E primitive

### Task 10: `generate_variants`

**Files:**
- Modify: `harness/core.py` (append the primitive)
- Modify: `tests/test_core.py` (append primitive tests)

- [ ] **Step 1: Append failing tests**

Append to `tests/test_core.py`:

```python
from pathlib import Path

from harness.core import generate_variants
from harness.frames import IdentityFrames
from harness.inference.client import FakeInferenceClient


def test_generate_variants_rejects_n_below_two(tmp_profile: Path) -> None:
    fake = FakeInferenceClient(responses=["a"])
    strategy = IdentityFrames(frames=["ceo"], profile_root=tmp_profile)

    with pytest.raises(ValueError, match="n must be >= 2"):
        generate_variants(
            prompt="x",
            n=1,
            frame_strategy=strategy,
            client=fake,
            rules=[],
        )


def test_generate_variants_returns_n_parsed_variants(
    tmp_profile: Path, fake_inference_responses: list[str]
) -> None:
    fake = FakeInferenceClient(responses=fake_inference_responses[:2])
    strategy = IdentityFrames(
        frames=["ceo", "legal"], profile_root=tmp_profile
    )

    variants = generate_variants(
        prompt="Should we ship now?",
        n=2,
        frame_strategy=strategy,
        client=fake,
        rules=[],
    )

    assert len(variants) == 2
    assert {v.metadata["frame_name"] for v in variants} == {"ceo", "legal"}
    # Each variant has parsed text/rationale/uncertainty
    for v in variants:
        assert v.text
        assert v.rationale
        assert 0.0 <= v.uncertainty.confidence <= 1.0


def test_generate_variants_passes_sampling_params_to_client(
    tmp_profile: Path,
) -> None:
    canned = (
        "<variant>\n<text>x</text>\n<rationale>y</rationale>\n</variant>"
    )
    fake = FakeInferenceClient(responses=[canned, canned, canned, canned])

    from harness.frames import TemperatureLadder

    strategy = TemperatureLadder(temperatures=[0.3, 0.7, 1.0, 1.3])
    generate_variants(
        prompt="x", n=4, frame_strategy=strategy, client=fake, rules=[]
    )

    # Each call's sampling_params reflect the ladder
    temps_passed = [c["sampling_params"].get("temperature") for c in fake.calls]
    assert temps_passed == [0.3, 0.7, 1.0, 1.3]


def test_generate_variants_records_run_id_in_metadata(
    tmp_profile: Path, fake_inference_responses: list[str]
) -> None:
    fake = FakeInferenceClient(responses=fake_inference_responses[:1])
    strategy = IdentityFrames(frames=["ceo"], profile_root=tmp_profile)

    # n=2 would mismatch a 1-frame strategy; use a 2-frame one
    fake2 = FakeInferenceClient(responses=fake_inference_responses[:2])
    strategy2 = IdentityFrames(
        frames=["ceo", "legal"], profile_root=tmp_profile
    )
    variants = generate_variants(
        prompt="x",
        n=2,
        frame_strategy=strategy2,
        client=fake2,
        rules=[],
        run_id="my-run-id",
    )
    assert all(v.metadata["run_id"] == "my-run-id" for v in variants)
```

- [ ] **Step 2: Verify failing**

```bash
.venv/bin/python -m pytest tests/test_core.py -v
```

Expected: FAIL with `ImportError: cannot import name 'generate_variants'`.

- [ ] **Step 3: Append `generate_variants` to `harness/core.py`**

Append to `harness/core.py`:

```python
import uuid

from harness.frames import FrameStrategy
from harness.inference.client import InferenceClient


def generate_variants(
    *,
    prompt: str,
    n: int,
    frame_strategy: FrameStrategy,
    client: InferenceClient,
    rules: list[str],
    run_id: str | None = None,
) -> list[Variant]:
    """The E primitive — produce N rationale-bearing variants with surfaced uncertainty.

    Contract:
      - n >= 2 (singular outputs forbidden at this layer)
      - Each variant carries text, rationale, and Uncertainty
      - Variant.metadata records frame_name and run_id

    See spec §2.
    """
    if n < 2:
        raise ValueError(
            "n must be >= 2 — singular outputs are forbidden at the primitive level"
        )

    if run_id is None:
        run_id = uuid.uuid4().hex[:12]

    specs = frame_strategy.generate_prompt_specs(
        base_prompt=prompt, n=n, rules=rules
    )
    if len(specs) != n:
        raise ValueError(
            f"frame_strategy returned {len(specs)} prompts; expected {n}"
        )

    variants: list[Variant] = []
    for spec in specs:
        response = client.complete(spec.text, **spec.sampling_params)
        variant = parse_variant(response, frame_name=spec.frame_name)
        variant.metadata["run_id"] = run_id
        variants.append(variant)
    return variants
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_core.py -v
```

Expected: PASS (14 tests total: 10 prior + 4 new).

- [ ] **Step 5: Commit**

```bash
git add harness/core.py tests/test_core.py
git commit -m "Add generate_variants — the E primitive"
```

---

This plan continues in `2026-05-07-conversational-harness-phase1-part2.md` (Phases 7-12: Conversation trace, orchestration patterns, profile bundles, CLI, demos, docs).

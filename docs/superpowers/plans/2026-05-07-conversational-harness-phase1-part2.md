# Conversational Harness Phase 1 Implementation Plan — Part 2

> **For agentic workers:** continuation of `2026-05-07-conversational-harness-phase1.md`. Part 1 covers Tasks 0-10 (Bootstrap → E primitive). This part covers Tasks 11-25 (Conversation trace → final integration).

---

## Phase 7: Conversation trace

### Task 11: `Conversation` dataclass + JSON serialisation

**Files:**
- Create: `harness/conversation.py`
- Test: `tests/test_conversation.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_conversation.py`:

```python
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
```

- [ ] **Step 2: Verify failing**

```bash
.venv/bin/python -m pytest tests/test_conversation.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `harness/conversation.py`**

```python
"""Conversation trace — the structured artifact of a harness run.

Every interaction with the harness produces a Conversation: which brief,
which pattern, which profile, every variant in every round, every user
decision. Serialised to JSON for persistence and downstream studies.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from harness.core import Uncertainty, Variant


VALID_DECISION_EVENTS = ("pick", "reject_all", "edit", "ask_more")


@dataclass
class UserDecision:
    """One user-facing decision recorded against the trace."""

    event: str
    variant_index: int
    round_index: int
    note: str = ""

    def __post_init__(self) -> None:
        if self.event not in VALID_DECISION_EVENTS:
            raise ValueError(
                f"event must be one of {VALID_DECISION_EVENTS}; got {self.event!r}"
            )


@dataclass
class Round:
    """One round of variant generation within a Conversation."""

    round_index: int
    variants: list[Variant]


@dataclass
class Conversation:
    """The full trace of one harness run."""

    run_id: str
    brief: str
    pattern: str
    profile: str
    rounds: list[Round] = field(default_factory=list)
    user_decisions: list[UserDecision] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "Conversation":
        data = json.loads(raw)
        rounds = [
            Round(
                round_index=r["round_index"],
                variants=[
                    Variant(
                        text=v["text"],
                        rationale=v["rationale"],
                        uncertainty=Uncertainty(**v["uncertainty"]),
                        metadata=v.get("metadata", {}),
                    )
                    for v in r["variants"]
                ],
            )
            for r in data.get("rounds", [])
        ]
        decisions = [
            UserDecision(**d) for d in data.get("user_decisions", [])
        ]
        return cls(
            run_id=data["run_id"],
            brief=data["brief"],
            pattern=data["pattern"],
            profile=data["profile"],
            rounds=rounds,
            user_decisions=decisions,
            metadata=data.get("metadata", {}),
        )
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_conversation.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add harness/conversation.py tests/test_conversation.py
git commit -m "Add Conversation trace dataclass with JSON roundtrip"
```

---

## Phase 8: Orchestration patterns

### Task 12: `OrchestrationPattern` protocol + `SinglePass`

**Files:**
- Create: `harness/orchestration/pattern.py`
- Create: `harness/orchestration/single_pass.py`
- Test: `tests/test_orchestration.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_orchestration.py`:

```python
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
```

- [ ] **Step 2: Verify failing**

```bash
.venv/bin/python -m pytest tests/test_orchestration.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `harness/orchestration/pattern.py`**

```python
"""OrchestrationPattern — the protocol every pattern implements."""
from __future__ import annotations

from typing import Protocol

from harness.conversation import Conversation
from harness.inference.client import InferenceClient


class OrchestrationPattern(Protocol):
    """A composable pattern that produces a Conversation trace.

    Patterns differ in how they sequence calls to the E primitive
    (SinglePass, multi-round Debate, source-grounded Synthesis,
    iterative refinement). Each pattern uses the same primitive contract.
    """

    def run(
        self,
        *,
        brief: str,
        client: InferenceClient,
        profile_name: str,
    ) -> Conversation: ...
```

- [ ] **Step 4: Implement `harness/orchestration/single_pass.py`**

```python
"""SinglePass — the smallest orchestration pattern.

One call to the E primitive, return all variants in one Round. Useful as
a building block and as a baseline for studies comparing more complex
patterns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.conversation import Conversation, Round
from harness.core import generate_variants
from harness.frames import FrameStrategy
from harness.inference.client import InferenceClient


@dataclass
class SinglePass:
    """One batch of N variants from one E-primitive call."""

    frame_strategy: FrameStrategy
    n: int
    rule_names: list[str]

    def run(
        self,
        *,
        brief: str,
        client: InferenceClient,
        profile_name: str,
    ) -> Conversation:
        # Phase 1: rules are not loaded from a ContextBundle yet. The CLI
        # composes them when it calls the pattern; here rule_names is just
        # a label list. See Task 16 for full ContextBundle integration.
        variants = generate_variants(
            prompt=brief,
            n=self.n,
            frame_strategy=self.frame_strategy,
            client=client,
            rules=[],  # rules wired by CLI in Task 16
        )
        run_id = variants[0].metadata["run_id"]
        return Conversation(
            run_id=run_id,
            brief=brief,
            pattern="SinglePass",
            profile=profile_name,
            rounds=[Round(round_index=0, variants=variants)],
        )
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/python -m pytest tests/test_orchestration.py -v
```

Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add harness/orchestration/pattern.py harness/orchestration/single_pass.py tests/test_orchestration.py
git commit -m "Add OrchestrationPattern protocol and SinglePass"
```

---

### Task 13: `DebatePattern`

**Files:**
- Create: `harness/orchestration/debate.py`
- Modify: `tests/test_orchestration.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_orchestration.py`:

```python
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
```

- [ ] **Step 2: Verify failing**

```bash
.venv/bin/python -m pytest tests/test_orchestration.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `harness/orchestration/debate.py`**

```python
"""DebatePattern — N framed agents debate across R rounds.

Each round, every frame produces one variant. From round 2 onward, every
frame's prompt is augmented with the prior rounds' variants as context,
so subsequent frames can respond, refine, or push back.
"""
from __future__ import annotations

from dataclasses import dataclass

from harness.conversation import Conversation, Round
from harness.core import generate_variants
from harness.frames import FrameStrategy
from harness.inference.client import InferenceClient


def _format_prior_rounds(rounds: list[Round]) -> str:
    """Render prior rounds as a 'PRIOR ROUNDS:' block for downstream prompts."""
    lines: list[str] = ["PRIOR ROUNDS:"]
    for r in rounds:
        lines.append(f"\nRound {r.round_index}:")
        for v in r.variants:
            frame = v.metadata.get("frame_name", "?")
            lines.append(f"  [{frame}]: {v.text}")
            lines.append(f"    rationale: {v.rationale}")
    return "\n".join(lines)


@dataclass
class DebatePattern:
    """Multi-round multi-frame debate using the E primitive each round."""

    frame_strategy: FrameStrategy
    n: int
    rounds: int
    rule_names: list[str]

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError(f"rounds must be >= 1; got {self.rounds}")

    def run(
        self,
        *,
        brief: str,
        client: InferenceClient,
        profile_name: str,
    ) -> Conversation:
        accumulated_rounds: list[Round] = []
        run_id: str | None = None

        for round_index in range(self.rounds):
            if accumulated_rounds:
                prior_block = _format_prior_rounds(accumulated_rounds)
                augmented_brief = f"{brief}\n\n{prior_block}"
            else:
                augmented_brief = brief

            variants = generate_variants(
                prompt=augmented_brief,
                n=self.n,
                frame_strategy=self.frame_strategy,
                client=client,
                rules=[],  # rules wired by CLI in Task 16
                run_id=run_id,
            )
            if run_id is None:
                run_id = variants[0].metadata["run_id"]
            accumulated_rounds.append(
                Round(round_index=round_index, variants=variants)
            )

        assert run_id is not None
        return Conversation(
            run_id=run_id,
            brief=brief,
            pattern="DebatePattern",
            profile=profile_name,
            rounds=accumulated_rounds,
        )
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_orchestration.py -v
```

Expected: PASS (5 tests total).

- [ ] **Step 5: Commit**

```bash
git add harness/orchestration/debate.py tests/test_orchestration.py
git commit -m "Add DebatePattern with prior-rounds context propagation"
```

---

### Task 14: `SynthesisPattern`

**Files:**
- Create: `harness/orchestration/synthesis.py`
- Modify: `tests/test_orchestration.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_orchestration.py`:

```python
from harness.frames import DisciplineFrames
from harness.orchestration.synthesis import SynthesisPattern


def test_synthesis_pattern_produces_one_round_per_discipline(
    tmp_profile: Path,
) -> None:
    # SynthesisPattern calls generate_variants once per discipline with n=2
    # (two angles per discipline). 3 disciplines × 2 angles = 6 inference calls.
    responses = _build_round_responses(n_frames=3, n_rounds=2)
    fake = FakeInferenceClient(responses=responses)
    pattern = SynthesisPattern(
        frame_strategy=DisciplineFrames(
            disciplines=["biology", "operations", "history"]
        ),
        n=3,
        rule_names=[],
    )

    conv = pattern.run(brief="reduce supply failures", client=fake, profile_name="academic")

    assert conv.pattern == "SynthesisPattern"
    assert len(conv.rounds) == 1
    assert len(conv.rounds[0].variants) == 3
    frames_seen = {v.metadata["frame_name"] for v in conv.rounds[0].variants}
    assert frames_seen == {"biology", "operations", "history"}


def test_synthesis_pattern_corpus_paths_appended_to_brief(
    tmp_profile: Path, tmp_path: Path
) -> None:
    """When a corpus is provided, file contents are appended to the brief."""
    corpus = tmp_path / "papers"
    bio = corpus / "biology"
    bio.mkdir(parents=True)
    (bio / "paper1.md").write_text("Title: Cellular automata in metabolism\nAbstract: …")

    responses = _build_round_responses(n_frames=1, n_rounds=1)
    fake = FakeInferenceClient(responses=responses)
    pattern = SynthesisPattern(
        frame_strategy=DisciplineFrames(disciplines=["biology"]),
        n=1,
        rule_names=[],
        corpus_root=corpus,
    )

    pattern.run(brief="study idea", client=fake, profile_name="academic")

    biology_call = fake.calls[0]
    assert "Cellular automata in metabolism" in biology_call["prompt"]
    assert "study idea" in biology_call["prompt"]
```

Note: the second test's frame strategy uses `n=1` for the synthesis pattern; this is the deliberate exception to the n>=2 primitive rule below — see implementation comment.

Wait — actually `generate_variants` enforces `n >= 2`. SynthesisPattern with one discipline would violate this. We need to either:
(a) Allow n=1 in `generate_variants` only when called from SynthesisPattern (a flag), or
(b) Require SynthesisPattern to use n>=2 disciplines (the spec's typical case anyway)

Option (b) is cleaner and matches the spec's intent (cross-disciplinary = multiple disciplines). Update the test to use 2 disciplines:

Replace the second test above with:

```python
def test_synthesis_pattern_corpus_paths_appended_to_brief(
    tmp_profile: Path, tmp_path: Path
) -> None:
    """When a corpus is provided, file contents per discipline are appended to that discipline's prompt."""
    corpus = tmp_path / "papers"
    bio = corpus / "biology"
    ops = corpus / "operations"
    bio.mkdir(parents=True)
    ops.mkdir(parents=True)
    (bio / "paper1.md").write_text(
        "Title: Cellular automata in metabolism\nAbstract: foo"
    )
    (ops / "paper1.md").write_text(
        "Title: Queueing under uncertainty\nAbstract: bar"
    )

    responses = _build_round_responses(n_frames=2, n_rounds=1)
    fake = FakeInferenceClient(responses=responses)
    pattern = SynthesisPattern(
        frame_strategy=DisciplineFrames(disciplines=["biology", "operations"]),
        n=2,
        rule_names=[],
        corpus_root=corpus,
    )

    pattern.run(brief="study idea", client=fake, profile_name="academic")

    biology_call = fake.calls[0]
    operations_call = fake.calls[1]
    assert "Cellular automata in metabolism" in biology_call["prompt"]
    assert "Queueing under uncertainty" in operations_call["prompt"]
    assert "study idea" in biology_call["prompt"]
    assert "Cellular automata in metabolism" not in operations_call["prompt"]
```

- [ ] **Step 2: Verify failing**

```bash
.venv/bin/python -m pytest tests/test_orchestration.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `harness/orchestration/synthesis.py`**

```python
"""SynthesisPattern — corpus-grounded multi-discipline synthesis.

For each discipline, the brief is augmented with that discipline's corpus
files (if a corpus_root is provided). The pattern produces one Round with
one variant per discipline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from harness.conversation import Conversation, Round
from harness.core import generate_variants
from harness.frames import DisciplineFrames, FrameStrategy
from harness.inference.client import InferenceClient


def _load_corpus_for_discipline(corpus_root: Path, discipline: str) -> str:
    """Read all *.md files in <corpus_root>/<discipline>/ and concatenate."""
    discipline_dir = corpus_root / discipline
    if not discipline_dir.exists():
        return ""
    parts: list[str] = []
    for path in sorted(discipline_dir.glob("*.md")):
        parts.append(f"--- {path.name} ---\n{path.read_text()}")
    return "\n\n".join(parts)


@dataclass
class SynthesisPattern:
    """Cross-disciplinary synthesis with optional per-discipline corpus.

    Phase 1: corpus is a folder-of-papers (deterministic, reproducible).
    Phase 2: corpus_root is replaced by an MCP retrieval client.
    """

    frame_strategy: FrameStrategy
    n: int
    rule_names: list[str]
    corpus_root: Path | None = None

    def run(
        self,
        *,
        brief: str,
        client: InferenceClient,
        profile_name: str,
    ) -> Conversation:
        if not isinstance(self.frame_strategy, DisciplineFrames):
            raise TypeError(
                "SynthesisPattern requires a DisciplineFrames strategy "
                f"to know which corpus subdirs to read; got {type(self.frame_strategy).__name__}"
            )

        all_variants = []
        run_id: str | None = None

        for discipline in self.frame_strategy.disciplines:
            corpus_text = ""
            if self.corpus_root is not None:
                corpus_text = _load_corpus_for_discipline(
                    self.corpus_root, discipline
                )

            if corpus_text:
                augmented = (
                    f"{brief}\n\nCORPUS for {discipline}:\n{corpus_text}"
                )
            else:
                augmented = brief

            single_strategy = DisciplineFrames(disciplines=[discipline])
            variants = generate_variants(
                prompt=augmented,
                n=2,  # generate two angles per discipline; one of them gets used
                frame_strategy=DisciplineFrames(
                    disciplines=[f"{discipline} (angle A)", f"{discipline} (angle B)"]
                ),
                client=client,
                rules=[],
                run_id=run_id,
            )
            if run_id is None:
                run_id = variants[0].metadata["run_id"]
            # For Phase 1 simplicity, take the first angle and re-tag with the discipline name
            v = variants[0]
            v.metadata["frame_name"] = discipline
            all_variants.append(v)

        # Wait — the test expects exactly len(disciplines) variants and exactly
        # len(disciplines) calls. The above generates 2 per discipline = 2N calls.
        # Simplify: one call per discipline, single-discipline frame strategy.
        # Replace the loop body:
        raise NotImplementedError("see corrected implementation below")
```

Replace the entire `run()` method with this corrected implementation:

```python
    def run(
        self,
        *,
        brief: str,
        client: InferenceClient,
        profile_name: str,
    ) -> Conversation:
        if not isinstance(self.frame_strategy, DisciplineFrames):
            raise TypeError(
                "SynthesisPattern requires a DisciplineFrames strategy; got "
                f"{type(self.frame_strategy).__name__}"
            )

        # The test expects: len(disciplines) variants total, one per discipline.
        # We loop discipline-by-discipline so each discipline's prompt can carry
        # its own corpus content. Each iteration calls generate_variants with
        # n=2 minimum; we discard the second variant. (The n>=2 contract is
        # respected at the primitive level; SynthesisPattern collapses to one
        # variant per discipline at the orchestration level.)
        all_variants = []
        run_id: str | None = None

        for discipline in self.frame_strategy.disciplines:
            corpus_text = ""
            if self.corpus_root is not None:
                corpus_text = _load_corpus_for_discipline(
                    self.corpus_root, discipline
                )
            augmented = (
                f"{brief}\n\nCORPUS for {discipline}:\n{corpus_text}"
                if corpus_text
                else brief
            )
            two_disciplines = [
                f"{discipline} (primary angle)",
                f"{discipline} (alternate angle)",
            ]
            variants = generate_variants(
                prompt=augmented,
                n=2,
                frame_strategy=DisciplineFrames(disciplines=two_disciplines),
                client=client,
                rules=[],
                run_id=run_id,
            )
            if run_id is None:
                run_id = variants[0].metadata["run_id"]
            primary = variants[0]
            primary.metadata["frame_name"] = discipline
            all_variants.append(primary)

        assert run_id is not None
        return Conversation(
            run_id=run_id,
            brief=brief,
            pattern="SynthesisPattern",
            profile=profile_name,
            rounds=[Round(round_index=0, variants=all_variants)],
        )
```

But this calls `generate_variants` twice per discipline, so the test expectation of `fake.calls[0]` having biology and `fake.calls[1]` having operations is wrong. Update the test:

Replace the `assert biology_call = fake.calls[0]` block with:

```python
    # Each discipline triggers 2 inference calls (primary + alternate angles).
    # First two calls are biology; next two are operations.
    biology_call = fake.calls[0]
    operations_call = fake.calls[2]
    assert "Cellular automata in metabolism" in biology_call["prompt"]
    assert "Queueing under uncertainty" in operations_call["prompt"]
    assert "study idea" in biology_call["prompt"]
    assert "Cellular automata in metabolism" not in operations_call["prompt"]
```

And update `_build_round_responses` usage in the synthesis test — it needs 4 responses (2 disciplines × 2 angles), not 2:

```python
    responses = _build_round_responses(n_frames=2, n_rounds=2)  # = 4 responses
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_orchestration.py -v
```

Expected: PASS (7 tests total).

- [ ] **Step 5: Commit**

```bash
git add harness/orchestration/synthesis.py tests/test_orchestration.py
git commit -m "Add SynthesisPattern with per-discipline corpus loading"
```

---

### Task 15: `IterativeRefinement`

**Files:**
- Create: `harness/orchestration/iterative.py`
- Modify: `tests/test_orchestration.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_orchestration.py`:

```python
from harness.orchestration.iterative import IterativeRefinement


def test_iterative_refinement_runs_critique_then_revise(
    tmp_profile: Path,
) -> None:
    # Cycle: initial 2 variants → critique appended → 2 revised variants
    responses = _build_round_responses(n_frames=2, n_rounds=2)
    fake = FakeInferenceClient(responses=responses)
    pattern = IterativeRefinement(
        frame_strategy=IdentityFrames(
            frames=["ceo", "legal"], profile_root=tmp_profile
        ),
        n=2,
        max_iterations=2,
        critique_fn=lambda variants: "User says: be more specific",
        rule_names=[],
    )

    conv = pattern.run(brief="x", client=fake, profile_name="test")

    assert conv.pattern == "IterativeRefinement"
    assert len(conv.rounds) == 2
    # Round 2's prompts must reference the critique
    round2_call = fake.calls[2]
    assert "be more specific" in round2_call["prompt"]


def test_iterative_refinement_stops_at_max_iterations(
    tmp_profile: Path,
) -> None:
    responses = _build_round_responses(n_frames=2, n_rounds=3)
    fake = FakeInferenceClient(responses=responses)
    pattern = IterativeRefinement(
        frame_strategy=IdentityFrames(
            frames=["ceo", "legal"], profile_root=tmp_profile
        ),
        n=2,
        max_iterations=3,
        critique_fn=lambda variants: "more",
        rule_names=[],
    )

    conv = pattern.run(brief="x", client=fake, profile_name="test")
    assert len(conv.rounds) == 3
```

- [ ] **Step 2: Verify failing**

```bash
.venv/bin/python -m pytest tests/test_orchestration.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `harness/orchestration/iterative.py`**

```python
"""IterativeRefinement — variants → critique → re-variants → repeat."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from harness.conversation import Conversation, Round
from harness.core import Variant, generate_variants
from harness.frames import FrameStrategy
from harness.inference.client import InferenceClient


CritiqueFn = Callable[[list[Variant]], str]


@dataclass
class IterativeRefinement:
    """Repeatedly produce variants, gather critique, refine.

    critique_fn takes the latest round's variants and returns a critique
    string. Phase 1 supports any callable (a human prompt UI, an
    automatic rule check, or a fixed string). The critique is appended
    to the brief in subsequent rounds.
    """

    frame_strategy: FrameStrategy
    n: int
    max_iterations: int
    critique_fn: CritiqueFn
    rule_names: list[str]

    def run(
        self,
        *,
        brief: str,
        client: InferenceClient,
        profile_name: str,
    ) -> Conversation:
        accumulated: list[Round] = []
        run_id: str | None = None
        critique_history: list[str] = []

        for iteration in range(self.max_iterations):
            if critique_history:
                critique_block = "\n".join(
                    f"Critique after round {i}: {c}"
                    for i, c in enumerate(critique_history)
                )
                augmented = f"{brief}\n\n{critique_block}"
            else:
                augmented = brief

            variants = generate_variants(
                prompt=augmented,
                n=self.n,
                frame_strategy=self.frame_strategy,
                client=client,
                rules=[],
                run_id=run_id,
            )
            if run_id is None:
                run_id = variants[0].metadata["run_id"]

            accumulated.append(
                Round(round_index=iteration, variants=variants)
            )

            if iteration < self.max_iterations - 1:
                critique = self.critique_fn(variants)
                critique_history.append(critique)

        assert run_id is not None
        return Conversation(
            run_id=run_id,
            brief=brief,
            pattern="IterativeRefinement",
            profile=profile_name,
            rounds=accumulated,
        )
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_orchestration.py -v
```

Expected: PASS (9 tests total).

- [ ] **Step 5: Commit**

```bash
git add harness/orchestration/iterative.py tests/test_orchestration.py
git commit -m "Add IterativeRefinement pattern with critique/revise loop"
```

---

### Task 16: Wire `ContextBundle` rules through orchestration patterns

**Files:**
- Modify: `harness/orchestration/single_pass.py`, `debate.py`, `synthesis.py`, `iterative.py`
- Modify: `tests/test_orchestration.py`

- [ ] **Step 1: Append a failing test that requires real rule wiring**

Append to `tests/test_orchestration.py`:

```python
from harness.context.bundle import load_bundle


def test_single_pass_loads_rules_from_bundle(
    tmp_profile: Path, fake_inference_responses: list[str]
) -> None:
    bundle = load_bundle(tmp_profile)
    fake = FakeInferenceClient(responses=fake_inference_responses[:2])
    pattern = SinglePass(
        frame_strategy=IdentityFrames(
            frames=["ceo", "legal"], profile_root=tmp_profile
        ),
        n=2,
        rule_names=["never_fabricate"],
    )

    pattern.run(
        brief="x", client=fake, profile_name="test", context=bundle
    )

    # Both prompts should include the never_fabricate rule body
    for call in fake.calls:
        assert "Never fabricate" in call["prompt"]


def test_single_pass_works_without_context_for_backward_compat(
    tmp_profile: Path, fake_inference_responses: list[str]
) -> None:
    fake = FakeInferenceClient(responses=fake_inference_responses[:2])
    pattern = SinglePass(
        frame_strategy=IdentityFrames(
            frames=["ceo", "legal"], profile_root=tmp_profile
        ),
        n=2,
        rule_names=[],  # no rules requested
    )
    conv = pattern.run(brief="x", client=fake, profile_name="test")
    assert len(conv.rounds[0].variants) == 2
```

- [ ] **Step 2: Verify failing**

```bash
.venv/bin/python -m pytest tests/test_orchestration.py -v
```

Expected: FAIL — the `context=bundle` kwarg isn't accepted.

- [ ] **Step 3: Update each pattern's `run()` to accept an optional `context` and resolve rules**

In `harness/orchestration/single_pass.py`, update `run()`:

```python
    def run(
        self,
        *,
        brief: str,
        client: InferenceClient,
        profile_name: str,
        context: "ContextBundle | None" = None,
    ) -> Conversation:
        rules = (
            context.select_rules(self.rule_names)
            if context is not None and self.rule_names
            else []
        )
        variants = generate_variants(
            prompt=brief,
            n=self.n,
            frame_strategy=self.frame_strategy,
            client=client,
            rules=rules,
        )
        run_id = variants[0].metadata["run_id"]
        return Conversation(
            run_id=run_id,
            brief=brief,
            pattern="SinglePass",
            profile=profile_name,
            rounds=[Round(round_index=0, variants=variants)],
        )
```

Add the import at the top of `single_pass.py`:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from harness.context.bundle import ContextBundle
```

Apply the same pattern (accept `context: ContextBundle | None = None`, resolve `rules` from `context.select_rules(self.rule_names)` when both are present, pass the result into `generate_variants`) to:
- `harness/orchestration/debate.py` (in the per-round loop)
- `harness/orchestration/synthesis.py` (in the per-discipline loop)
- `harness/orchestration/iterative.py` (in the per-iteration loop)

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_orchestration.py -v
```

Expected: PASS (11 tests total).

- [ ] **Step 5: Commit**

```bash
git add harness/orchestration/ tests/test_orchestration.py
git commit -m "Wire ContextBundle rule selection through all orchestration patterns"
```

---

## Phase 9: Profile bundles (data, no TDD)

### Task 17: Business profile bundle

**Files:**
- Create: `profiles/_common/output_format.md`
- Create: `profiles/business/personality/{ceo,legal,marketing}.md`
- Create: `profiles/business/rules/{never_fabricate,disagree_explicitly}.md`

- [ ] **Step 1: Create `profiles/_common/output_format.md`**

```markdown
# Output Format

You produce structured output wrapped in these XML tags:

```
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
[semicolon-separated failure modes you acknowledge, e.g.: "citation may be hallucinated"; "outside training cutoff"; "claim is opinion, not data"]
</flags>

<verification>
[semicolon-separated things the human should check, e.g.: "verify the regulatory claim"; "confirm the date"; "cross-check with engineering"]
</verification>
</variant>
```

Always include all five tags. If a tag has no content, leave it empty but include the tag.

Be honest about uncertainty. Confident-sounding claims with no verification hooks are a quality problem, not a positive trait.
```

- [ ] **Step 2: Create `profiles/business/personality/ceo.md`**

```markdown
# CEO Frame

You are a CEO speaking in your own voice — direct, strategic, accountable to outcomes.

Your distinctive lens:
- Strategic implications over tactical detail
- Speed and decisiveness over comprehensive analysis
- Owners' framing: "what would I do with my own money?"
- Comfort with calculated risk; impatience with paralysis
- Concern for incentives, narrative, and momentum

What you do NOT do:
- Defer to consensus when you disagree
- Bury the lead in qualifications
- Pretend certainty you don't have

When you produce a response, speak as a CEO would speak in a board meeting: short, pointed, with a clear recommendation and the reasoning behind it.
```

- [ ] **Step 3: Create `profiles/business/personality/legal.md`**

```markdown
# Legal Counsel Frame

You are senior legal counsel — precise, risk-aware, careful with claims.

Your distinctive lens:
- Identify regulatory, contractual, and liability exposure
- Distinguish what is established vs what is uncertain
- Highlight jurisdictional variation when relevant
- Quantify risk where possible; flag where it cannot be quantified
- Push back on assumptions that shift legal exposure

What you do NOT do:
- Block decisions you disagree with — surface concerns, recommend mitigations
- Speak to non-legal questions outside your expertise
- Confuse "low risk" with "no risk"

When you produce a response, speak as legal counsel would speak in a planning meeting: precise about the risk landscape, explicit about uncertainty, constructive on mitigation paths.
```

- [ ] **Step 4: Create `profiles/business/personality/marketing.md`**

```markdown
# Marketing Director Frame

You are a marketing director — narrative-aware, customer-centric, optionality-loving.

Your distinctive lens:
- How will the market hear this?
- What's the story we can tell, and who tells it?
- Optionality: keep doors open until you have data
- Tactical sequencing — soft launches, beta cohorts, testimonial mining
- Customer voice: what does the ICP actually want?

What you do NOT do:
- Fall in love with your own narrative at the expense of evidence
- Ignore product or operational realities the story would have to bend
- Claim certainty about market response without data

When you produce a response, speak as a marketing director would speak in a product strategy meeting: narrative-first, customer-grounded, willing to advocate for an angle while flagging what would need to be true for it to work.
```

- [ ] **Step 5: Create `profiles/business/rules/never_fabricate.md`**

```markdown
# Never Fabricate

Do not invent:
- Citations to papers, books, or articles
- Statistics, percentages, or quantitative claims
- Quotes attributed to real people
- Names of products, companies, or studies you are not certain exist
- Regulatory references (specific laws, sections, or rulings)

If you would normally cite a source but cannot verify it from training, do one of the following:
1. Make the claim general ("research has examined…") rather than specific
2. Add a flag in your `<flags>` block ("citation not verifiable from training")
3. Add a verification hook ("verify [specific claim] against current literature")

Fabrication is a contract violation, not a stylistic choice.
```

- [ ] **Step 6: Create `profiles/business/rules/disagree_explicitly.md`**

```markdown
# Disagree Explicitly

When the brief or prior rounds contain a position you disagree with, surface the disagreement in plain language. Do not paper over it.

Specifically:
- Name the prior position you are pushing back on
- State your alternative
- Explain what would have to be true for the prior position to be right
- Explain what would have to be true for your position to be right

The harness's value comes from variance and friction, not from agreement. A frame that nods along to other frames produces no useful signal.
```

- [ ] **Step 7: Commit**

```bash
git add profiles/_common/ profiles/business/
git commit -m "Add business profile bundle (CEO/legal/marketing personalities + rules)"
```

---

### Task 18: Academic profile bundle

**Files:**
- Create: `profiles/academic/personality/{methodologist,theorist,practitioner,cross_disciplinary_scout}.md`
- Create: `profiles/academic/rules/{cite_or_flag,explicit_uncertainty_on_citations}.md`

- [ ] **Step 1: Create `profiles/academic/personality/methodologist.md`**

```markdown
# Methodologist Frame

You are a research methodologist — concerned with how knowledge gets produced, not just what it claims.

Your distinctive lens:
- What evidence would be needed to support this claim?
- What study design would test it cleanly?
- Where are the confounds, selection effects, or measurement issues?
- How replicable is the underlying finding?
- What does the methods section say (or fail to say)?

When you produce a response, speak as a methodologist would speak in a peer review: precise about claim-evidence gaps, constructive about what would strengthen the work, willing to flag when a confident result rests on shaky methodology.
```

- [ ] **Step 2: Create `profiles/academic/personality/theorist.md`**

```markdown
# Theorist Frame

You are a theorist — concerned with mechanisms, frameworks, and conceptual coherence.

Your distinctive lens:
- What is the underlying mechanism this claim relies on?
- Does this finding fit existing theory, or extend/challenge it?
- What are the boundary conditions where the claim holds vs breaks?
- How does this connect to deeper or older bodies of theory?
- What would a clean theoretical formulation look like?

When you produce a response, speak as a theorist would speak in a working group: framework-first, concerned with internal coherence, willing to step back and ask whether the question itself is well-posed.
```

- [ ] **Step 3: Create `profiles/academic/personality/practitioner.md`**

```markdown
# Practitioner Frame

You are a practitioner working in the field this research touches — the person who would have to use this in the real world.

Your distinctive lens:
- Does this work in a real setting, with real constraints?
- What does the implementation cost look like?
- Where will this fail when it leaves the lab?
- What ground-truth signals would tell us this is working?
- Whose lived experience is missing from the framing?

When you produce a response, speak as a practitioner would speak in a debrief: ground-truthed, skeptical of clean models, attentive to who pays the cost when the abstraction leaks.
```

- [ ] **Step 4: Create `profiles/academic/personality/cross_disciplinary_scout.md`**

```markdown
# Cross-Disciplinary Scout Frame

You are a researcher who reads broadly across fields. Your job is not to be the expert in any one field but to surface unexpected connections between fields that field-bound experts miss.

Your distinctive lens:
- What is this problem analogous to in other fields?
- Which other field has solved a structurally similar problem?
- What vocabulary from another field would clarify this one?
- Which methodological technique from another field has not been imported here?
- Where does academic siloing make this problem look harder than it is?

When you produce a response, lead with the unexpected connection. Name the source field. Be explicit about how confident you are that the analogy holds. Flag where the analogy breaks.
```

- [ ] **Step 5: Create `profiles/academic/rules/cite_or_flag.md`**

```markdown
# Cite or Flag

When you reference specific work (a paper, book, study, or finding):
1. **If you can cite it confidently from training, do so** — title, year, lead author or known landmark
2. **If you can't cite it confidently, do not invent a citation.** Add a verification hook: "verify the [specific claim] against current literature"
3. **If you are summarising a body of work without naming a specific source**, that is fine — make the generalisation level explicit ("research has examined…", "a body of work argues…")

Inventing citations is the cardinal sin in academic discourse and the fastest way to discredit the harness. When in doubt, generalise rather than fabricate.
```

- [ ] **Step 6: Create `profiles/academic/rules/explicit_uncertainty_on_citations.md`**

```markdown
# Explicit Uncertainty on Citations

Even when a citation feels familiar, treat it as something the human should verify. Add a verification hook for any specific:
- Year of publication
- Author name on a specific paper
- Study sample size or finding
- Quotation from a named work

Small models are notorious for citation drift — getting the field right but the source wrong, or fusing two real papers into one fake one. The verification hook is not a sign of weakness; it is the harness doing its job.
```

- [ ] **Step 7: Commit**

```bash
git add profiles/academic/
git commit -m "Add academic profile bundle (4 frames + 2 citation-discipline rules)"
```

---

### Task 19: Writing profile bundle

**Files:**
- Create: `profiles/writing/personality/{protagonist,antagonist,narrator}.md`
- Create: `profiles/writing/rules/{stay_in_voice,no_self_referential}.md`

- [ ] **Step 1: Create `profiles/writing/personality/protagonist.md`**

```markdown
# Protagonist Frame

You write from the protagonist's interior — what they want, what they fear, what they refuse to admit.

Your distinctive lens:
- Desire: what does the protagonist want, in this scene, right now?
- Obstacle: what stands between them and that want?
- Self-deception: what are they wrong about, that the reader can see?
- Texture: what specific sensory detail anchors the moment?

When you produce a response, speak in the protagonist's voice — interior monologue, concrete sensory grounding, unspoken stakes.
```

- [ ] **Step 2: Create `profiles/writing/personality/antagonist.md`**

```markdown
# Antagonist Frame

You write from the antagonist's interior. Your job is to ensure the antagonist is not a cardboard obstacle — they have desire, logic, and (in their own mind) justification.

Your distinctive lens:
- The antagonist's desire is just as concrete as the protagonist's
- The antagonist is the hero of their own story
- Where the antagonist's logic is internally consistent
- The cost the antagonist is paying that the protagonist doesn't see

When you produce a response, give the antagonist their own ground. Make them harder to dismiss.
```

- [ ] **Step 3: Create `profiles/writing/personality/narrator.md`**

```markdown
# Narrator Frame

You are the narrator standing slightly outside the scene — the voice that knows things neither the protagonist nor antagonist can know, and chooses what to reveal.

Your distinctive lens:
- Pace: when to slow down, when to compress
- Information control: what the reader should learn now vs later
- Tone: ironic, sympathetic, distant, complicit?
- Foreshadowing without giving the game away

When you produce a response, write at the narrator's altitude — aware of what each character does not yet know, controlling the reader's view.
```

- [ ] **Step 4: Create `profiles/writing/rules/stay_in_voice.md`**

```markdown
# Stay in Voice

When you have been assigned a frame (protagonist, antagonist, narrator), write *in* that voice — not *about* it. Do not break frame to comment on what the frame is doing.

Bad: "From the protagonist's perspective, they would want…"
Good: "I want this. I can almost taste it."

Bad: "The narrator would observe that…"
Good: "She didn't know it yet, but the door had been locked for an hour."

Voice break is a contract violation in the writing profile.
```

- [ ] **Step 5: Create `profiles/writing/rules/no_self_referential.md`**

```markdown
# No Self-Referential Output

Do not refer to yourself as an AI, an LLM, a model, or as the harness. Do not refer to the prompt, the brief, or the user.

Bad: "As an AI, I would suggest the protagonist…"
Good: [the protagonist's voice, with no AI surface]

Bad: "Based on your brief, here is the antagonist's perspective:"
Good: [the antagonist's voice, with no preamble]

Stay in fictional reality. Break only via the structured `<flags>` and `<verification>` tags, never via the body of `<text>`.
```

- [ ] **Step 6: Commit**

```bash
git add profiles/writing/
git commit -m "Add writing profile bundle (protagonist/antagonist/narrator + 2 voice rules)"
```

---

## Phase 10: CLI

### Task 20: CLI entry point + `debate` command

**Files:**
- Create: `cli/locoagente.py`
- Create: `cli/commands/debate.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_cli.py`:

```python
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
    # Set up a minimal profile bundle for the test
    profile_root = tmp_path / "profiles" / "business"
    (profile_root / "personality").mkdir(parents=True)
    (profile_root / "rules").mkdir(parents=True)
    (profile_root / "personality" / "ceo.md").write_text("CEO voice.")
    (profile_root / "personality" / "legal.md").write_text("Legal voice.")
    (profile_root / "rules" / "never_fabricate.md").write_text("Don't make stuff up.")

    runner = CliRunner()
    output_path = tmp_path / "trace.json"

    # Patch inference client construction to inject FakeInferenceClient
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
```

- [ ] **Step 2: Verify failing**

```bash
.venv/bin/python -m pytest tests/test_cli.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `cli/locoagente.py`**

```python
"""LocoAgente CLI entry point."""
from __future__ import annotations

import click

from cli.commands.debate import debate
from cli.commands.synthesise import synthesise
from cli.commands.log import log


@click.group()
@click.version_option(version="0.1.0", prog_name="locoagente")
def main() -> None:
    """LocoAgente — conversational harness for small local models."""


main.add_command(debate)
main.add_command(synthesise)
main.add_command(log)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement `cli/commands/debate.py`**

```python
"""locoagente debate — runs a perspective-debate over N frames and R rounds."""
from __future__ import annotations

import os
from pathlib import Path

import click

from harness.context.bundle import load_bundle
from harness.frames import IdentityFrames
from harness.inference.client import InferenceClient, OpenAICompatibleClient
from harness.orchestration.debate import DebatePattern


def build_inference_client() -> InferenceClient:
    """Construct the default inference client from environment.

    Defaults to Ollama at http://localhost:11434/v1 with model qwen3:4b.
    Override via env vars LOCOAGENTE_BASE_URL, LOCOAGENTE_API_KEY,
    LOCOAGENTE_MODEL.
    """
    return OpenAICompatibleClient(
        base_url=os.environ.get(
            "LOCOAGENTE_BASE_URL", "http://localhost:11434/v1"
        ),
        api_key=os.environ.get("LOCOAGENTE_API_KEY", "ollama-local"),
        model=os.environ.get("LOCOAGENTE_MODEL", "qwen3:4b"),
    )


@click.command()
@click.option("--brief", required=True, help="The question or topic to debate.")
@click.option(
    "--frames",
    required=True,
    help="Comma-separated frame names (e.g., 'ceo,legal,marketing').",
)
@click.option(
    "--rounds",
    type=int,
    default=3,
    show_default=True,
    help="Number of debate rounds.",
)
@click.option(
    "--profile-root",
    type=click.Path(file_okay=False, exists=True, path_type=Path),
    required=True,
    help="Path to the profile bundle directory.",
)
@click.option(
    "--rules",
    default="",
    help="Comma-separated rule names from the profile (e.g., 'never_fabricate').",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Where to write the Conversation trace JSON.",
)
def debate(
    brief: str,
    frames: str,
    rounds: int,
    profile_root: Path,
    rules: str,
    output: Path,
) -> None:
    """Run a perspective debate."""
    frame_list = [f.strip() for f in frames.split(",") if f.strip()]
    rule_list = [r.strip() for r in rules.split(",") if r.strip()]

    bundle = load_bundle(profile_root)
    pattern = DebatePattern(
        frame_strategy=IdentityFrames(
            frames=frame_list, profile_root=profile_root
        ),
        n=len(frame_list),
        rounds=rounds,
        rule_names=rule_list,
    )
    client = build_inference_client()
    profile_name = profile_root.name

    conv = pattern.run(
        brief=brief,
        client=client,
        profile_name=profile_name,
        context=bundle,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(conv.to_json())
    click.echo(f"Wrote trace to {output}")
```

- [ ] **Step 5: Create stub `cli/commands/synthesise.py` and `cli/commands/log.py`** so the import in `cli/locoagente.py` works

`cli/commands/synthesise.py`:

```python
"""locoagente synthesise — placeholder; full impl in Task 21."""
from __future__ import annotations

import click


@click.command()
def synthesise() -> None:
    """Run cross-disciplinary synthesis (skeleton in Phase 1)."""
    click.echo("synthesise: not yet implemented (see Task 21)")
```

`cli/commands/log.py`:

```python
"""locoagente log — placeholder; full impl in Task 22."""
from __future__ import annotations

import click


@click.command()
def log() -> None:
    """Inspect Conversation trace JSON files (full impl in Task 22)."""
    click.echo("log: not yet implemented (see Task 22)")
```

- [ ] **Step 6: Run tests**

```bash
.venv/bin/python -m pytest tests/test_cli.py -v
```

Expected: PASS (2 tests).

- [ ] **Step 7: Commit**

```bash
git add cli/ tests/test_cli.py
git commit -m "Add CLI entry point and debate command"
```

---

### Task 21: `synthesise` command

**Files:**
- Modify: `cli/commands/synthesise.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Append failing test**

Append to `tests/test_cli.py`:

```python
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
    # SynthesisPattern calls generate_variants twice per discipline (n=2),
    # so 2 disciplines × 2 angles = 4 responses needed.
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
```

- [ ] **Step 2: Verify failing**

```bash
.venv/bin/python -m pytest tests/test_cli.py -v
```

Expected: FAIL — `synthesise` doesn't accept these flags yet.

- [ ] **Step 3: Replace `cli/commands/synthesise.py` with full implementation**

```python
"""locoagente synthesise — cross-disciplinary synthesis demo."""
from __future__ import annotations

import os
from pathlib import Path

import click

from harness.context.bundle import load_bundle
from harness.frames import DisciplineFrames
from harness.inference.client import InferenceClient, OpenAICompatibleClient
from harness.orchestration.synthesis import SynthesisPattern


def build_inference_client() -> InferenceClient:
    return OpenAICompatibleClient(
        base_url=os.environ.get(
            "LOCOAGENTE_BASE_URL", "http://localhost:11434/v1"
        ),
        api_key=os.environ.get("LOCOAGENTE_API_KEY", "ollama-local"),
        model=os.environ.get("LOCOAGENTE_MODEL", "qwen3:4b"),
    )


@click.command()
@click.option("--brief", required=True, help="The research question.")
@click.option(
    "--disciplines",
    required=True,
    help="Comma-separated disciplines (e.g., 'biology,operations').",
)
@click.option(
    "--corpus-root",
    type=click.Path(file_okay=False, exists=True, path_type=Path),
    default=None,
    help="Optional corpus directory with one subdir per discipline.",
)
@click.option(
    "--profile-root",
    type=click.Path(file_okay=False, exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--rules",
    default="cite_or_flag,explicit_uncertainty_on_citations",
    show_default=True,
    help="Comma-separated rule names from the profile.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
)
def synthesise(
    brief: str,
    disciplines: str,
    corpus_root: Path | None,
    profile_root: Path,
    rules: str,
    output: Path,
) -> None:
    """Run cross-disciplinary synthesis."""
    discipline_list = [d.strip() for d in disciplines.split(",") if d.strip()]
    rule_list = [r.strip() for r in rules.split(",") if r.strip()]

    bundle = load_bundle(profile_root)
    # Filter rule_list to ones actually in the bundle (so default rule list
    # is forgiving of profile bundles that don't ship every rule)
    rule_list = [r for r in rule_list if r in bundle.rules]

    pattern = SynthesisPattern(
        frame_strategy=DisciplineFrames(disciplines=discipline_list),
        n=len(discipline_list),
        rule_names=rule_list,
        corpus_root=corpus_root,
    )
    client = build_inference_client()
    profile_name = profile_root.name

    conv = pattern.run(
        brief=brief,
        client=client,
        profile_name=profile_name,
        context=bundle,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(conv.to_json())
    click.echo(f"Wrote trace to {output}")
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_cli.py -v
```

Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add cli/commands/synthesise.py tests/test_cli.py
git commit -m "Implement synthesise CLI command"
```

---

### Task 22: `log` command (inspect traces)

**Files:**
- Modify: `cli/commands/log.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Append failing test**

Append to `tests/test_cli.py`:

```python
from harness.conversation import Conversation, Round, UserDecision
from harness.core import Uncertainty, Variant


def test_log_command_summarises_trace(tmp_path: Path) -> None:
    v = Variant(
        text="answer",
        rationale="why",
        uncertainty=Uncertainty(
            confidence=0.7,
            flags=["a flag"],
            verification_hooks=["check X"],
        ),
        metadata={"frame_name": "ceo", "run_id": "r1"},
    )
    conv = Conversation(
        run_id="r1",
        brief="test brief",
        pattern="DebatePattern",
        profile="business",
        rounds=[Round(round_index=0, variants=[v])],
        user_decisions=[
            UserDecision(event="pick", variant_index=0, round_index=0)
        ],
    )
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(conv.to_json())

    runner = CliRunner()
    result = runner.invoke(main, ["log", "--trace", str(trace_path)])

    assert result.exit_code == 0, result.output
    assert "DebatePattern" in result.output
    assert "test brief" in result.output
    assert "ceo" in result.output
    assert "check X" in result.output  # verification hook surfaced
```

- [ ] **Step 2: Verify failing**

```bash
.venv/bin/python -m pytest tests/test_cli.py -v
```

Expected: FAIL — `log` doesn't accept `--trace`.

- [ ] **Step 3: Replace `cli/commands/log.py`**

```python
"""locoagente log — inspect Conversation trace JSON files."""
from __future__ import annotations

from pathlib import Path

import click

from harness.conversation import Conversation


@click.command()
@click.option(
    "--trace",
    type=click.Path(dir_okay=False, exists=True, path_type=Path),
    required=True,
    help="Path to a Conversation trace JSON file.",
)
def log(trace: Path) -> None:
    """Print a human-readable summary of a Conversation trace."""
    conv = Conversation.from_json(trace.read_text())
    click.echo(f"Run ID: {conv.run_id}")
    click.echo(f"Pattern: {conv.pattern}")
    click.echo(f"Profile: {conv.profile}")
    click.echo(f"Brief: {conv.brief}")
    click.echo(f"Rounds: {len(conv.rounds)}")
    click.echo("")
    for round_ in conv.rounds:
        click.echo(f"--- Round {round_.round_index} ---")
        for i, v in enumerate(round_.variants):
            frame = v.metadata.get("frame_name", "?")
            click.echo(f"  [{i}] frame: {frame}")
            click.echo(f"      text: {v.text[:200]}")
            click.echo(f"      rationale: {v.rationale[:200]}")
            click.echo(
                f"      confidence: {v.uncertainty.confidence:.2f}"
            )
            if v.uncertainty.flags:
                click.echo(f"      flags: {'; '.join(v.uncertainty.flags)}")
            if v.uncertainty.verification_hooks:
                click.echo(
                    f"      verify: {'; '.join(v.uncertainty.verification_hooks)}"
                )
        click.echo("")
    if conv.user_decisions:
        click.echo("--- User decisions ---")
        for d in conv.user_decisions:
            click.echo(
                f"  round {d.round_index} variant {d.variant_index}: {d.event}"
                + (f" — {d.note}" if d.note else "")
            )
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_cli.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add cli/commands/log.py tests/test_cli.py
git commit -m "Implement log CLI command for trace inspection"
```

---

## Phase 11: Demo applications

### Task 23: Perspective Debate demo runner + Research Synthesis sample corpus

**Files:**
- Create: `applications/conversation/perspective_debate/runner.py`
- Create: `applications/conversation/perspective_debate/README.md`
- Create: `applications/conversation/research_synthesis/runner.py`
- Create: `applications/conversation/research_synthesis/README.md`
- Create: `applications/conversation/research_synthesis/sample_papers/{biology,operations,history,linguistics}/*.md` (4 short paper stubs per discipline)

- [ ] **Step 1: Create `applications/conversation/perspective_debate/runner.py`**

```python
"""Perspective Debate demo runner.

Convenience wrapper around the CLI's `debate` command for the headline
business demo. Equivalent to:

    locoagente debate \\
        --brief "<your brief>" \\
        --frames "ceo,legal,marketing" \\
        --rounds 3 \\
        --profile-root ./profiles/business \\
        --rules "never_fabricate,disagree_explicitly" \\
        --output ./out/<timestamp>.json
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from cli.commands.debate import build_inference_client
from harness.context.bundle import load_bundle
from harness.frames import IdentityFrames
from harness.orchestration.debate import DebatePattern


def run_demo(
    brief: str,
    *,
    frames: list[str] | None = None,
    rounds: int = 3,
    profile_root: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Run the demo and return the path to the written trace."""
    if frames is None:
        frames = ["ceo", "legal", "marketing"]
    if profile_root is None:
        profile_root = Path(__file__).resolve().parents[3] / "profiles" / "business"
    if output_dir is None:
        output_dir = Path("./out")

    bundle = load_bundle(profile_root)
    pattern = DebatePattern(
        frame_strategy=IdentityFrames(
            frames=frames, profile_root=profile_root
        ),
        n=len(frames),
        rounds=rounds,
        rule_names=["never_fabricate", "disagree_explicitly"],
    )
    client = build_inference_client()
    conv = pattern.run(
        brief=brief,
        client=client,
        profile_name="business",
        context=bundle,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"perspective_debate_{int(time.time())}.json"
    output_path.write_text(conv.to_json())
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m applications.conversation.perspective_debate.runner '<brief>'")
        sys.exit(1)
    path = run_demo(sys.argv[1])
    print(f"Wrote {path}")
```

- [ ] **Step 2: Create `applications/conversation/perspective_debate/README.md`**

```markdown
# Perspective Debate Demo

The Phase 1 headline demo. Three business frames (CEO, legal counsel, marketing director) debate a brief over multiple rounds. The harness produces a structured `Conversation` trace surfacing each frame's perspective, where they disagree, what each is uncertain about, and what the human should verify.

## Run it

Prerequisite: an OpenAI-compatible inference backend running locally. Default is Ollama at `http://localhost:11434/v1` with model `qwen3:4b`.

### Via CLI:

```bash
.venv/bin/locoagente debate \
    --brief "Should we ship the feature without GDPR review?" \
    --frames "ceo,legal,marketing" \
    --rounds 3 \
    --profile-root ./profiles/business \
    --rules "never_fabricate,disagree_explicitly" \
    --output ./out/debate.json

.venv/bin/locoagente log --trace ./out/debate.json
```

### Via demo runner:

```bash
.venv/bin/python -m applications.conversation.perspective_debate.runner \
    "Should we ship the feature without GDPR review?"
```

## What success looks like

- Three personalities are distinguishable (blind test: a reader can guess which frame produced which variant)
- Disagreements are surfaced explicitly, not glossed
- Uncertainty hooks fire when the model claims something verifiable

## Customisation

- Different frames: pass `--frames "founder,investor,customer"` (after creating those personality files in `profiles/business/personality/`)
- Different audience: switch profile bundle with `--profile-root ./profiles/academic` and use academic frame names (e.g., `methodologist,theorist,practitioner`)
- Different model: set `LOCOAGENTE_MODEL=phi-4-mini` etc.
```

- [ ] **Step 3: Create `applications/conversation/research_synthesis/runner.py`**

```python
"""Research Synthesis demo runner — Phase 1 skeleton.

Phase 1 ships with a stubbed corpus (folder of papers grouped by discipline)
for reproducibility. Phase 2 will replace the stub with real MCP retrieval
(Semantic Scholar / arXiv / Zotero).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from cli.commands.synthesise import build_inference_client
from harness.context.bundle import load_bundle
from harness.frames import DisciplineFrames
from harness.orchestration.synthesis import SynthesisPattern


def run_demo(
    brief: str,
    *,
    disciplines: list[str] | None = None,
    profile_root: Path | None = None,
    corpus_root: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    if disciplines is None:
        disciplines = ["biology", "operations", "history", "linguistics"]
    if profile_root is None:
        profile_root = Path(__file__).resolve().parents[3] / "profiles" / "academic"
    if corpus_root is None:
        corpus_root = Path(__file__).resolve().parent / "sample_papers"
    if output_dir is None:
        output_dir = Path("./out")

    bundle = load_bundle(profile_root)
    rule_list = [
        r for r in ["cite_or_flag", "explicit_uncertainty_on_citations"]
        if r in bundle.rules
    ]
    pattern = SynthesisPattern(
        frame_strategy=DisciplineFrames(disciplines=disciplines),
        n=len(disciplines),
        rule_names=rule_list,
        corpus_root=corpus_root,
    )
    client = build_inference_client()
    conv = pattern.run(
        brief=brief,
        client=client,
        profile_name="academic",
        context=bundle,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"research_synthesis_{int(time.time())}.json"
    output_path.write_text(conv.to_json())
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m applications.conversation.research_synthesis.runner '<brief>'")
        sys.exit(1)
    path = run_demo(sys.argv[1])
    print(f"Wrote {path}")
```

- [ ] **Step 4: Create `applications/conversation/research_synthesis/README.md`**

```markdown
# Research Synthesis Demo

Phase 1: skeleton with a stubbed corpus (folder of papers, deterministic).
Phase 2: real MCP retrieval (Semantic Scholar / arXiv / Zotero) wired in behind the same orchestration code.

## Run it

```bash
.venv/bin/python -m applications.conversation.research_synthesis.runner \
    "How do other fields handle long-tail rare-event prediction?"
```

The default corpus (`sample_papers/`) groups short paper stubs by discipline (biology, operations, history, linguistics). The harness pulls all `*.md` files in each discipline subdirectory into that discipline's prompt context.

## What success looks like

- Given the corpus, produces 1–2 connections per discipline with rationale + uncertainty
- Each connection has citation hooks for verification
- Runs without confabulating citations to papers not in the corpus (the `cite_or_flag` rule's real test)

## Adding your own corpus

```bash
mkdir -p ./my_corpus/{economics,physics,sociology}
cp ~/Zotero/exports/economics/*.md ./my_corpus/economics/
# (repeat per discipline)

.venv/bin/python -m applications.conversation.research_synthesis.runner \
    "your research question" \
    --corpus-root ./my_corpus
```
```

- [ ] **Step 5: Create the sample paper corpus — 8 short stubs across 4 disciplines**

For each of the four disciplines, create `applications/conversation/research_synthesis/sample_papers/<discipline>/paper_{1,2}.md` with short, plausibly-real abstracts. Each file should be ~150–300 words and not invent citations. Example for biology:

`applications/conversation/research_synthesis/sample_papers/biology/paper_1.md`:

```markdown
# Robustness through redundancy in gene regulatory networks

## Abstract

Gene regulatory networks routinely produce stable phenotypes despite stochastic
fluctuations in protein concentrations. A growing body of work argues that this
robustness is not the product of finely-tuned parameters but of structural
redundancy: multiple parallel pathways achieving the same downstream effect.
We review evidence from three model organisms (E. coli, S. cerevisiae, C. elegans)
showing that the empirical robustness of canonical regulatory motifs significantly
exceeds what would be predicted by single-pathway models. We argue that this
redundancy has implications for fields outside biology: any system that must
maintain function under fluctuating inputs may benefit from the structural
patterns biology has discovered through evolution. We close with three open
questions about how redundancy emerges and how its costs are paid.

## Methods

A meta-analysis of published time-series perturbation studies in three model
organisms; redundancy was operationalised as the number of parallel pathways
that, when individually disrupted, fail to abolish the phenotype.
```

`applications/conversation/research_synthesis/sample_papers/biology/paper_2.md`:

```markdown
# Symbiotic stability in changing environments

## Abstract

Persistent multi-species relationships face a thermodynamic puzzle: how do they
remain stable when the environments hosting them change? We examine three
classes of microbial symbiosis (lichen, gut commensals, deep-sea hydrothermal
communities) and find a common pattern of distributed sensing — the partner
species jointly maintain a buffered chemical environment that absorbs
environmental shocks before any single partner must respond. The implication
is that stability under change is achieved by collective signal smoothing, not
by individual robustness. We discuss the analogy to engineered systems and to
human institutions where buffering occurs across organisations rather than
within them.

## Limitations

The case studies are observational; controlled experiments on lichen and
hydrothermal communities are difficult. We do not claim that all stable
symbioses use this mechanism, only that the three examined ones share it.
```

`applications/conversation/research_synthesis/sample_papers/operations/paper_1.md`:

```markdown
# Inventory buffering as redundancy

## Abstract

Operational supply chains carry inventory in excess of what immediate demand
would suggest. Standard treatment frames this as inefficiency to be eliminated
via just-in-time scheduling. We argue the framing has it backwards: in the
presence of demand and supply uncertainty, inventory buffer is the system's
mechanism for redundancy, and removing it without first reducing the
underlying uncertainty produces brittleness. We re-analyse three industries
(consumer electronics, automotive, pharmaceuticals) and find that supply chain
disruptions in 2020–2022 disproportionately affected firms that had pursued
the most aggressive inventory reduction. The implication is that inventory
levels and uncertainty levels must be considered jointly; one is the buffer
for the other.

## Method

Quasi-experimental difference-in-differences on industry-level disruption-impact
data from 2018–2023.
```

`applications/conversation/research_synthesis/sample_papers/operations/paper_2.md`:

```markdown
# Queueing under heavy-tailed service times

## Abstract

Classical queueing theory assumes service times have light tails. Empirical
data from healthcare, customer support, and IT incident response often shows
heavy-tailed service times: most incidents resolve quickly but a long tail of
incidents takes orders of magnitude longer. We extend M/G/1 results to
heavy-tailed regimes and show that average wait times are dominated by the
tail rather than the body of the distribution. The practical implication for
operations: optimising for median service time can dramatically degrade
worst-case wait. Systems serving heavy-tailed loads need explicit slack capacity
or routing rules that protect the body from the tail.

## Caveats

The mathematical results assume stationarity, which empirical incident data
often violates. We discuss extensions to non-stationary regimes informally.
```

`applications/conversation/research_synthesis/sample_papers/history/paper_1.md`:

```markdown
# Resilience in pre-industrial trading networks

## Abstract

Long-distance trading networks operating before standardised insurance and
contract law nonetheless persisted across centuries. Historical records from
the Hanseatic League, Mediterranean galley networks, and the Indian Ocean
dhow trade reveal a common pattern: relational mechanisms (reputation,
multi-generational kin trust, codified shared customs) substituted for the
formal institutions absent at the time. Disruption was managed through
network-level redundancy — when one route or partner failed, others could
absorb the load. We argue these networks discovered design patterns
(redundant pathways, distributed reputation, buffered exchange) that modern
distributed systems engineering has rediscovered. The convergence is not
coincidence; both face the same structural problem of trust and continuity
under uncertainty.

## Limitations

Historical sources are uneven; the Hanseatic record is far richer than the
dhow record. We are explicit about which claims rest on which corpus.
```

`applications/conversation/research_synthesis/sample_papers/history/paper_2.md`:

```markdown
# Information cascades in famine response

## Abstract

Mass famines in the 19th century produced characteristic information cascades:
local officials, fearing punishment for raising alarms prematurely, suppressed
warning signals; by the time the signal reached central authorities,
intervention windows had closed. We compare archival sources from the Irish
famine, the Bengal famine, and the Great Chinese Famine to identify a common
structural failure mode: the asymmetry between the cost of false-positive
warnings (career-ending) and the cost of false-negative warnings (paid by
people far from the warner). The implication for modern systems facing tail
risks is sobering: the same asymmetry persists in many institutional contexts.

## Method

Archival sources, secondary historiography, and a small set of interviews
with descendants of mid-level officials.
```

`applications/conversation/research_synthesis/sample_papers/linguistics/paper_1.md`:

```markdown
# Linguistic robustness through redundancy

## Abstract

Natural languages routinely transmit messages reliably across noisy channels:
talkative restaurants, cell-phone audio, multilingual code-switching. We
analyse three forms of linguistic redundancy that contribute to this
robustness: phonological (multiple cues per phoneme), morphological
(grammatical agreement signalling the same information twice), and pragmatic
(context filling in dropped or mis-heard tokens). The three forms appear to
trade off — languages that are phonologically richer tend to have lower
morphological redundancy, and vice versa. We propose this is a structural
feature of how communication systems balance reliability against
transmission cost.

## Limitations

The cross-linguistic sample is moderate (30 languages); the trade-off
relation is suggestive rather than statistically conclusive at this sample
size.
```

`applications/conversation/research_synthesis/sample_papers/linguistics/paper_2.md`:

```markdown
# Code-switching as redundancy management

## Abstract

Bilingual speakers code-switch in ways that classical sociolinguistics
treated as performative or identity-driven. We argue an additional functional
account: code-switching frequently introduces redundancy where the speaker
estimates the channel reliability is low. In a corpus of bilingual
conversation, code-switches concentrated at moments of repair, emphasis, or
the introduction of high-stakes content. The pattern is consistent with
speakers using their second language as an error-correction code over their
first. The phenomenon has analogues in distributed systems: when uncertainty
is high, encode the message twice in different formats.

## Caveats

The corpus is restricted to four bilingual communities; the redundancy
account does not exclude identity and performative accounts.
```

- [ ] **Step 6: Commit**

```bash
git add applications/conversation/
git commit -m "Add Perspective Debate and Research Synthesis demo runners + sample corpus"
```

---

## Phase 12: Documentation

### Task 24: Philosophy + architecture + tutorials docs

**Files:**
- Create: `docs/philosophy/{conversation-not-delegation,creativity-needs-variance-not-precision,confidence-is-not-competence}.md`
- Create: `docs/architecture/four-subsystems.md`
- Create: `docs/tutorials/{perspective-debate,research-synthesis}.md`
- Create: `docs/cli-reference.md`

- [ ] **Step 1: Create three philosophy docs**

`docs/philosophy/conversation-not-delegation.md`:

```markdown
# Conversation, not delegation

The harness's design pressure is "amplification through verified offloading."

Delegation is not binary. The question is what level you delegate at, and whether there is a verification loop.

| Delegate this | Keep this |
|---|---|
| The mechanical (compute, retrieve, draft, enumerate, summarise) | The judgment (taste, smell tests, strategic direction) |
| The breadth (scan 50 papers across disciplines) | The depth (which 3 actually matter) |
| The first draft | The voice |
| The "what are my options" | The "which one feels right" |

Cognitive offloading is acceptable when there is a verification loop. The cash-register example: when you buy items at a shop, you delegate the addition to the till; when the total seems wrong, you ask to check the prices. That is not surrender; it is amplification with a smell-test boundary.

The harness's job is to **make the verification loop cheap**:

- Surface uncertainty so the human's smell test has something to fire on
- Expose tool outputs so verification doesn't require re-running the work
- Maintain conversation state so the human can push back without re-explaining

The model handles the mechanical. The human keeps the judgment. The harness is the connective tissue that lets the human work at a higher level without losing oversight at any level they choose to inspect.
```

`docs/philosophy/creativity-needs-variance-not-precision.md`:

```markdown
# Creativity needs variance, not precision

Frontier models win on **precision tasks**: factual recall, math, code correctness, well-defined classification. Small local models lose those head-to-head — confabulation, mode collapse, narrower training distributions.

Frontier models lose on **creative tasks**: when 50 frontier models write a story about a boy and a dragon, there is a striking sameness across them. They converge to a safe centre. Variance — the soul of creativity — is what they erode.

Small local models, properly harnessed, can compete on creative tasks for the inverted reason: their imprecision *becomes variance*, and variance is what creativity needs. The design imperative is to **stop trying to make small models precise** and instead channel their variance through the harness.

This is the project's positioning claim. It justifies why a conversational harness on small local models is worth building rather than a frontier-API wrapper.

How the harness channels variance:

- The E primitive enforces `n >= 2` — singular outputs are forbidden at the primitive level
- `FrameStrategy` engineers variance deliberately — identity frames, discipline frames, temperature ladders, constraint inversions — rather than hoping for it from N samples of the same prompt
- Every `Variant` carries a rationale, so the variance is legible (the human can see *why* this variant differs from that one) instead of opaque
```

`docs/philosophy/confidence-is-not-competence.md`:

```markdown
# Confidence is not competence

Humans treat confidence as a synonym for competence. A confident-sounding model sounds correct. This is dangerous, especially with small local models whose confidence is famously miscalibrated.

The harness must surface uncertainty as a first-class signal. Every `Variant` carries a rationale and an `Uncertainty` marker, so the human's smell test has something to fire on. Refusing to surface uncertainty is a design defect, not a UX simplification. A confident-sounding output with no uncertainty markers is a primitive-contract violation.

The `Uncertainty` dataclass:

- **`flags`** — load-bearing. Known failure modes the model acknowledges ("citation may be hallucinated", "outside training cutoff"). UI surfaces these prominently.
- **`verification_hooks`** — load-bearing. What the human should check ("verify the price", "confirm the date"). Cheap-verification handoffs.
- **`confidence`** — auxiliary. 0.0–1.0 self-reported. Treated as a research artifact, not a load-bearing signal. Logged but not used to gate decisions, because small-model self-confidence is famously miscalibrated.
- **`relative_rank`** — optional rank within a batch (1 = highest). More informative than absolute confidence when available, because comparative judgments tend to be better calibrated than absolute ones.

Calibration capture starts on day one: every user pick / reject / edit on every variant is logged in machine-readable form. Over time this produces a labeled dataset (model self-confidence claim → human-validated outcome) that a downstream Context-layer study analyses for systematic miscalibration patterns.
```

- [ ] **Step 2: Create `docs/architecture/four-subsystems.md`**

```markdown
# The four subsystems

The harness has four named subsystems, mapped to the four-layer enterprise harness model from contemporary literature but using names rather than numbers because the diagram doesn't flow numerically and the project's research surfaces are not the enterprise framing's research surfaces.

```
ORCHESTRATION  — patterns that compose the E primitive
                  (DebatePattern, SynthesisPattern, IterativeRefinement, SinglePass)
                              ↓ uses
CONTEXT        — hierarchical instruction file system
                  (personality, rules, memory, tools)
                              ↓ uses
TOOLS          — MCP-based external capabilities
                  (Phase 1 stub; Phase 2 real)
                              ↓ uses
INFERENCE      — OpenAI-compatible HTTP backend
                  (Ollama default; loco-bench specifies which engine)
```

The four subsystems are not equally novel. **Orchestration and Context are where the project's research contribution lives.** Tools is "adopt MCP, don't reinvent." Inference is already covered by loco-bench (which engine to pick) and loco-puente (deployment).

## Orchestration

Each pattern implements `OrchestrationPattern` and produces a `Conversation` trace.

| Pattern | What it does | Demo using it |
|---|---|---|
| `SinglePass` | One call → N variants → return all | building block |
| `DebatePattern` | N framed agents → R rounds → optional moderator merge | Perspective Debate |
| `SynthesisPattern` | N source-grounded variants → dedup → attribute | Research Synthesis |
| `IterativeRefinement` | Variants → critique → re-variant with feedback | research-track use |

New patterns are first-class extension points — students can implement a new one in a few hundred lines.

## Context

Modular instruction files loaded by session hooks based on the current task:

```
profiles/<profile_name>/
├── personality/         # one file per frame: ceo.md, legal.md, marketing.md
├── rules/               # hard constraints: never_fabricate.md, cite_or_flag.md
├── memory/              # persistent: user_preferences.md, calibration_history.jsonl
└── tools/               # available MCP tool specs as markdown
```

No mega-system-prompt strings are permitted in code.

## Tools

Phase 1: stub MCP client. Phase 2: real MCP servers wired in. The harness never knows whether a `search_corpus` tool is RAG-backed, BM25, full-text scan, or human typing — it just sees the tool's input/output contract.

If MCP is replaced by a successor protocol, the Tools subsystem is the only place the harness needs to change. The other three are unaffected.

## Inference

The harness is inference-engine-agnostic. It speaks OpenAI-compatible HTTP. Defaults to Ollama for ergonomics; loco-bench specifies which engine is best for which VRAM tier.
```

- [ ] **Step 3: Create `docs/tutorials/perspective-debate.md`**

```markdown
# Tutorial: Perspective Debate

You will run the Perspective Debate demo end-to-end against a small local model. You will see how the harness produces structured `Conversation` traces, surface the verification hooks each frame produced, and understand how to swap profiles for different audiences.

## Prerequisites

1. Python 3.10+
2. [uv](https://github.com/astral-sh/uv) installed
3. [Ollama](https://ollama.ai) installed and running locally
4. A small model pulled into Ollama:

   ```bash
   ollama pull qwen3:4b
   ```

## Set up

```bash
cd loco-agente
uv venv .venv
uv pip install -e '.[dev]' --python .venv/bin/python
```

## Run a debate

```bash
.venv/bin/locoagente debate \
    --brief "Should we ship the GDPR-flagged feature on Tuesday?" \
    --frames "ceo,legal,marketing" \
    --rounds 3 \
    --profile-root ./profiles/business \
    --rules "never_fabricate,disagree_explicitly" \
    --output ./out/debate.json
```

## Inspect the trace

```bash
.venv/bin/locoagente log --trace ./out/debate.json
```

Output sections:
- **Run ID, Pattern, Profile, Brief** — basic provenance
- **Round 0** — three variants (CEO, legal, marketing). Each has text, rationale, confidence, flags, verification hooks
- **Round 1, Round 2** — same three frames, but each variant now sees the prior rounds and can push back
- **User decisions** — empty initially; populated when you start integrating the trace with a UI

## What to look for

- Are the three frames distinguishable? Read the rationales — does each one *sound* like that frame?
- Are the disagreements explicit? Look for places where round 1 or 2 calls back to round 0 and argues against it
- Did any variant produce a verification hook? Are those hooks things you would actually check?

## Try a different audience

Switch the profile bundle:

```bash
.venv/bin/locoagente debate \
    --brief "Is this study design adequate to support the claim?" \
    --frames "methodologist,theorist,practitioner" \
    --rounds 3 \
    --profile-root ./profiles/academic \
    --rules "cite_or_flag,explicit_uncertainty_on_citations" \
    --output ./out/academic_debate.json
```

The same `DebatePattern` runs; only the profile and frames change.

## Common issues

- **Model returns no `<variant>` tags.** Some models are bad at structured output. Try a different model (`LOCOAGENTE_MODEL=phi-4-mini` or larger). Watch for the parser raising `VariantParseError`.
- **All three frames sound the same.** Frame collapse — the model isn't following the personality files. Try a more capable model, or rewrite the personalities to be more distinct.
- **Connection refused.** Ollama isn't running, or the model isn't pulled. Check `ollama list` and `curl http://localhost:11434/v1/models`.
```

- [ ] **Step 4: Create `docs/tutorials/research-synthesis.md`**

```markdown
# Tutorial: Research Synthesis

You will run the Research Synthesis skeleton against the bundled sample corpus, then point it at your own corpus.

## Prerequisites

Same as the perspective debate tutorial.

## Run with the bundled corpus

```bash
.venv/bin/locoagente synthesise \
    --brief "How do other fields handle long-tail rare-event prediction?" \
    --disciplines "biology,operations,history,linguistics" \
    --corpus-root ./applications/conversation/research_synthesis/sample_papers \
    --profile-root ./profiles/academic \
    --output ./out/synthesis.json

.venv/bin/locoagente log --trace ./out/synthesis.json
```

## What to look for

- One variant per discipline
- Each variant should reference at least one paper from its discipline's corpus
- Each variant's verification hooks should call out where it might have confabulated a citation (the `cite_or_flag` rule)

## Run with your own corpus

Build a corpus directory with one subdirectory per discipline:

```bash
mkdir -p ./my_corpus/{economics,physics,sociology}
# Drop *.md files (papers, abstracts, notes) in each subdir
```

Then:

```bash
.venv/bin/locoagente synthesise \
    --brief "your research question" \
    --disciplines "economics,physics,sociology" \
    --corpus-root ./my_corpus \
    --profile-root ./profiles/academic \
    --output ./out/my_synthesis.json
```

## Methodological note: the stubbed corpus is a feature

Live retrieval (web search, Semantic Scholar API) makes demos non-deterministic — the same brief produces different results from one run to the next, because the underlying world has changed. The Phase 1 stubbed corpus is a methodological commitment, not a shortcut: same input → same output → reproducible across runs and across users.

Phase 2 wires real MCP retrieval behind the same orchestration code. Comparing Phase 1 (stubbed) and Phase 2 (real) results on the same brief will be one of the project's first calibration studies.
```

- [ ] **Step 5: Create `docs/cli-reference.md`**

```markdown
# CLI Reference

## `locoagente debate`

Run a perspective debate over multiple rounds.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--brief` | yes | — | The question or topic. |
| `--frames` | yes | — | Comma-separated frame names (e.g., `ceo,legal,marketing`). |
| `--rounds` | no | `3` | Number of debate rounds. |
| `--profile-root` | yes | — | Path to the profile bundle directory. |
| `--rules` | no | (none) | Comma-separated rule names from the profile. |
| `--output` | yes | — | Where to write the trace JSON. |

## `locoagente synthesise`

Run a cross-disciplinary synthesis. One variant per discipline.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--brief` | yes | — | The research question. |
| `--disciplines` | yes | — | Comma-separated disciplines. |
| `--corpus-root` | no | (none) | Optional corpus directory with one subdir per discipline. |
| `--profile-root` | yes | — | Path to the profile bundle. |
| `--rules` | no | `cite_or_flag,explicit_uncertainty_on_citations` | Comma-separated rule names. |
| `--output` | yes | — | Where to write the trace JSON. |

## `locoagente log`

Print a human-readable summary of a Conversation trace.

| Flag | Required | Description |
|---|---|---|
| `--trace` | yes | Path to a Conversation trace JSON. |

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LOCOAGENTE_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible endpoint. |
| `LOCOAGENTE_API_KEY` | `ollama-local` | API key (Ollama doesn't check it). |
| `LOCOAGENTE_MODEL` | `qwen3:4b` | Model name as the inference backend reports it. |
```

- [ ] **Step 6: Commit**

```bash
git add docs/philosophy/ docs/architecture/ docs/tutorials/ docs/cli-reference.md
git commit -m "Add philosophy docs, architecture overview, and tutorials"
```

---

## Phase 13: Final integration

### Task 25: README rewrite + repo-wide smoke test + coverage check

**Files:**
- Modify: `README.md`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Rewrite `README.md`**

Read the existing `README.md` (the prior track A/B/C/D narrative) and replace its content with the new conversational-harness mission. Keep the existing badges block (lines marked with `<!-- BADGES:START -->` and `<!-- BADGES:END -->`); replace everything else with:

```markdown
# LocoAgente

### Local Conversational Harness — Small Models as Thinking Partners, Not Substitutes

> *"Frontier models win on precision. Small local models, properly harnessed, can win on variance — and variance is what creativity needs."*

LocoAgente is a research project building a **conversational harness** for small local models — a four-subsystem architecture (Orchestration / Context / Tools / Inference) that amplifies human thinking through verified offloading rather than substituting for it.

The harness's foundational primitive: **always produce N alternatives with rationale and surfaced uncertainty.** Singular outputs are forbidden at the primitive level. Variance is engineered, not hoped for.

## The three load-bearing principles

1. **Creativity does not need precision; it needs ideas.** Stop trying to make small models precise; channel their variance.
2. **Conversation, not delegation — but verified offloading is fine.** The harness makes the verification loop cheap.
3. **Confidence is not competence.** Surface uncertainty as a first-class signal.

See [`docs/philosophy/`](docs/philosophy/) for the full statements.

## Two demos

- **Perspective Debate** (Phase 1, end-to-end): N business / academic / writing frames debate a brief over R rounds. See [`docs/tutorials/perspective-debate.md`](docs/tutorials/perspective-debate.md).
- **Research Synthesis** (Phase 1 skeleton, Phase 2 functional): cross-disciplinary scan of a folder-of-papers corpus. See [`docs/tutorials/research-synthesis.md`](docs/tutorials/research-synthesis.md).

## Quick start

```bash
git clone https://github.com/michael-borck/loco-agente && cd loco-agente
uv venv .venv
uv pip install -e '.[dev]' --python .venv/bin/python
ollama pull qwen3:4b

.venv/bin/locoagente debate \
    --brief "Should we ship without a GDPR review?" \
    --frames "ceo,legal,marketing" \
    --rounds 3 \
    --profile-root ./profiles/business \
    --rules "never_fabricate,disagree_explicitly" \
    --output ./out/debate.json

.venv/bin/locoagente log --trace ./out/debate.json
```

## Architecture

See [`docs/architecture/four-subsystems.md`](docs/architecture/four-subsystems.md) and the design spec at [`docs/superpowers/specs/2026-05-07-conversational-harness-design.md`](docs/superpowers/specs/2026-05-07-conversational-harness-design.md).

## Connection to the loco fleet

- **loco-llm** — provides base model + adapters; the harness consumes them
- **loco-bench** — answers "which inference engine + model fits my VRAM tier"
- **loco-puente** — deployment substrate (if/when the harness gets a web UI)

## License

MIT.

## Citation

```bibtex
@software{locoagente2026,
  title={LocoAgente: Conversational Harness for Small Local Models},
  author={Michael Borck and Contributors},
  year={2026},
  url={https://github.com/michael-borck/loco-agente}
}
```
```

- [ ] **Step 2: Create `tests/test_smoke.py` for end-to-end wiring against `FakeInferenceClient`**

```python
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
```

- [ ] **Step 3: Run the full test suite with coverage**

```bash
.venv/bin/python -m pytest tests/ -v --cov=harness --cov-report=term-missing
```

Expected: all tests pass; coverage on `harness/core.py` and `harness/orchestration/` ≥ 80%. If coverage is below threshold on those modules, add targeted tests.

- [ ] **Step 4: Run the linter sanity check (no lint config configured yet, but at least confirm imports resolve)**

```bash
.venv/bin/python -c "import harness; import cli.locoagente; import applications.conversation.perspective_debate.runner; import applications.conversation.research_synthesis.runner; print('all imports OK')"
```

Expected: `all imports OK`.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_smoke.py
git commit -m "Add end-to-end smoke test and rewrite README around conversational harness"
```

---

## Out of scope for this plan (parking lot)

These are documented in spec §6 and §7 but **not in this plan**. They become Phase 2+ work:

- Real MCP retrieval (replaces the stubbed `StubToolClient` and the folder-of-papers in Research Synthesis)
- Calibration analysis (Track C study on captured user-feedback data)
- Orchestration framework comparison (Track D study: hand-rolled vs LangGraph vs CrewAI)
- Delegation applications (autoresearch port, task agents) — Phase 3 work
- Web UI / chatbot UI as downstream consumers of the harness (out of repo)
- Adapter training (Phase 4 LocoLLM integration)

---

## Appendix: Implementer notes

- **Run `pytest` from the repo root.** Tests use `tmp_path` fixtures liberally; nothing in `tests/` writes to the real filesystem outside of `tmp_path`.
- **Use `.venv/bin/python -m pytest` rather than activating the venv.** Matches the loco-bench pattern.
- **Keep the harness library inference-engine-agnostic.** Anything specific to Ollama (default model name, default base URL) lives in `cli/commands/*.py`'s `build_inference_client()` factories, not in `harness/`.
- **The CLI commands' `build_inference_client()` is patched in tests.** That's the seam between the deterministic `FakeInferenceClient` (for tests) and the real `OpenAICompatibleClient` (for live runs).
- **No GPU is required to run the test suite.** The smoke test uses `FakeInferenceClient`. Real-model demo runs require an Ollama (or compatible) backend.
- **Phase 1 is intentionally CLI-only.** A web UI / chatbot is a downstream project that imports the harness, not a Phase 1 deliverable. See spec §6.

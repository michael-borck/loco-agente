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

    Each iteration of the per-discipline loop calls generate_variants with
    n=2 (primary + alternate angles); the pattern keeps the primary and
    discards the alternate. This respects the n>=2 primitive contract while
    presenting one variant per discipline at the orchestration level.
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
                "SynthesisPattern requires a DisciplineFrames strategy; got "
                f"{type(self.frame_strategy).__name__}"
            )

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
                rules=[],  # rules wired by CLI in Task 16
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

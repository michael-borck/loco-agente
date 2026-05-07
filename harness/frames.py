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

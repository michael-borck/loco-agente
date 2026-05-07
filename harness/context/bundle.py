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

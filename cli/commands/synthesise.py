"""locoagente synthesise — cross-disciplinary synthesis demo."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import click

from harness import __version__ as harness_version
from harness.context.bundle import load_bundle
from harness.frames import DisciplineFrames
from harness.inference.client import InferenceClient, OpenAICompatibleClient
from harness.orchestration.synthesis import SynthesisPattern


def _git_sha(path: Path) -> str | None:
    """Return the short git SHA at <path>, or None if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


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
    conv.metadata.update({
        "harness_version": harness_version,
        "model": os.environ.get("LOCOAGENTE_MODEL", "qwen3:4b"),
        "base_url": os.environ.get("LOCOAGENTE_BASE_URL", "http://localhost:11434/v1"),
        "profile_git_sha": _git_sha(profile_root),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(conv.to_json())
    click.echo(f"Wrote trace to {output}")

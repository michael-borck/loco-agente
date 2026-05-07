"""locoagente debate — runs a perspective-debate over N frames and R rounds."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import click

from harness import __version__ as harness_version
from harness.context.bundle import load_bundle
from harness.frames import IdentityFrames
from harness.inference.client import InferenceClient, OpenAICompatibleClient
from harness.orchestration.debate import DebatePattern


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
    conv.metadata.update({
        "harness_version": harness_version,
        "model": os.environ.get("LOCOAGENTE_MODEL", "qwen3:4b"),
        "base_url": os.environ.get("LOCOAGENTE_BASE_URL", "http://localhost:11434/v1"),
        "profile_git_sha": _git_sha(profile_root),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(conv.to_json())
    click.echo(f"Wrote trace to {output}")

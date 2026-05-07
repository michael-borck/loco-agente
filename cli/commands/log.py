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

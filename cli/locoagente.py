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

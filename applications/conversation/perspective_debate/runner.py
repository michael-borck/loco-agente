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

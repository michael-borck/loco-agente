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

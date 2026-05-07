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

# Research Synthesis Demo

Phase 1: skeleton with a stubbed corpus (folder of papers, deterministic).
Phase 2: real MCP retrieval (Semantic Scholar / arXiv / Zotero) wired in behind the same orchestration code.

## Run it

```bash
.venv/bin/python -m applications.conversation.research_synthesis.runner \
    "How do other fields handle long-tail rare-event prediction?"
```

The default corpus (`sample_papers/`) groups short paper stubs by discipline (biology, operations, history, linguistics). The harness pulls all `*.md` files in each discipline subdirectory into that discipline's prompt context.

## What success looks like

- Given the corpus, produces 1–2 connections per discipline with rationale + uncertainty
- Each connection has citation hooks for verification
- Runs without confabulating citations to papers not in the corpus (the `cite_or_flag` rule's real test)

## Adding your own corpus

```bash
mkdir -p ./my_corpus/{economics,physics,sociology}
cp ~/Zotero/exports/economics/*.md ./my_corpus/economics/
# (repeat per discipline)

.venv/bin/python -m applications.conversation.research_synthesis.runner \
    "your research question" \
    --corpus-root ./my_corpus
```

# The four subsystems

The harness has four named subsystems, mapped to the four-layer enterprise harness model from contemporary literature but using names rather than numbers because the diagram doesn't flow numerically and the project's research surfaces are not the enterprise framing's research surfaces.

```
ORCHESTRATION  — patterns that compose the E primitive
                  (DebatePattern, SynthesisPattern, IterativeRefinement, SinglePass)
                              ↓ uses
CONTEXT        — hierarchical instruction file system
                  (personality, rules, memory, tools)
                              ↓ uses
TOOLS          — MCP-based external capabilities
                  (Phase 1 stub; Phase 2 real)
                              ↓ uses
INFERENCE      — OpenAI-compatible HTTP backend
                  (Ollama default; loco-bench specifies which engine)
```

The four subsystems are not equally novel. **Orchestration and Context are where the project's research contribution lives.** Tools is "adopt MCP, don't reinvent." Inference is already covered by loco-bench (which engine to pick) and loco-puente (deployment).

## Orchestration

Each pattern implements `OrchestrationPattern` and produces a `Conversation` trace.

| Pattern | What it does | Demo using it |
|---|---|---|
| `SinglePass` | One call → N variants → return all | building block |
| `DebatePattern` | N framed agents → R rounds → optional moderator merge | Perspective Debate |
| `SynthesisPattern` | N source-grounded variants → dedup → attribute | Research Synthesis |
| `IterativeRefinement` | Variants → critique → re-variant with feedback | research-track use |

New patterns are first-class extension points — students can implement a new one in a few hundred lines.

## Context

Modular instruction files loaded by session hooks based on the current task:

```
profiles/<profile_name>/
├── personality/         # one file per frame: ceo.md, legal.md, marketing.md
├── rules/               # hard constraints: never_fabricate.md, cite_or_flag.md
├── memory/              # persistent: user_preferences.md, calibration_history.jsonl
└── tools/               # available MCP tool specs as markdown
```

No mega-system-prompt strings are permitted in code.

## Tools

Phase 1: stub MCP client. Phase 2: real MCP servers wired in. The harness never knows whether a `search_corpus` tool is RAG-backed, BM25, full-text scan, or human typing — it just sees the tool's input/output contract.

If MCP is replaced by a successor protocol, the Tools subsystem is the only place the harness needs to change. The other three are unaffected.

## Inference

The harness is inference-engine-agnostic. It speaks OpenAI-compatible HTTP. Defaults to Ollama for ergonomics; loco-bench specifies which engine is best for which VRAM tier.

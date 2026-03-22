---
title: "Track C: Scaffolding Research"
description: "Systematic study of which scaffolding strategies most improve small-model agent reliability."
---

# Track C: Scaffolding Research

The systematic study of what scaffolding strategies most improve small-model agent reliability. This is the most publishable track — it produces general findings about SLM agent design that apply beyond any single task domain.

---

## Research Questions

1. Which individual scaffolding strategies produce the largest per-step accuracy gains for SLM agents?
2. Do strategy gains compound (RE2 + voting > RE2 alone + voting alone)?
3. Are there diminishing returns — a point where adding more scaffolding stops helping?
4. Do strategies interact negatively (e.g., CoT consumes context needed for voting)?
5. Is there a minimal effective set — the fewest strategies that capture most of the reliability gain?

---

## Strategy Categories Under Study

### Prompting Strategies

- Chain-of-thought (CoT)
- RE2 (Re-Reading)
- Role prompting
- Structured output formats

### Sampling Strategies

- Self-consistency voting (vote on N proposals)
- Best-of-N selection

### Action Space Constraints

- Free-form generation
- Structured menus (choose from N options)
- Binary choices (yes/no, improve/revert)

### Loop Structures

- Single-pass (one attempt per step)
- Iterative refinement (try, check, retry)
- Human-in-the-loop checkpoints

### Memory Strategies

- Full context (everything stays in the prompt)
- Summarised history (compress prior turns)
- External state files (offload state to disk, read back what's needed)

For detailed descriptions of each strategy see [Scaffolding Strategies](scaffolding-strategies).

---

## Evaluation Method

Each strategy (and combination) is tested against the same agent task using the same model, hardware, and evaluation harness. Variables are isolated — one change at a time unless testing interactions.

**Metrics:**
- Per-step accuracy (did this step produce the right action?)
- Loop completion rate (did the agent reach the goal in N steps?)
- Steps to first error
- Drift rate (goal coherence measured at each step)
- Token cost per loop (context consumed by the strategy)

**Control:** Bare local model with no scaffolding, on the same task.

**Baseline:** Frontier model with no scaffolding, on the same task.

---

## Connection to Cognitive Strategy Transfer

The scaffolding strategies tested here are the same strategies humans use when delegating to junior team members:

- **Constrained choices** → "here are three options, pick one" rather than "do whatever you think is best"
- **Verification checkpoints** → "show me your work before moving on"
- **Iterative refinement** → "try it, and if it doesn't work, try something else"
- **Explicit reasoning** → "explain your thinking before you act"

The CST framework predicts these should transfer to AI agents for the same structural reasons they work with humans. LocoAgente is an empirical test of that prediction.

---

## Dependencies

Can start alongside Track A — uses the autoresearch task as the initial test harness. As Track B adds new task domains, scaffolding experiments extend to those domains for cross-task validation.

---

## Status

Phase 2 (Semester 1, 2027). Early experiments (CoT, RE2) in Phase 1.

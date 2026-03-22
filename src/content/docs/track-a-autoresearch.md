---
title: "Track A: Local Autoresearch"
description: "Porting Karpathy's autoresearch loop to a local small model."
---

# Track A: Local Autoresearch

The starting point. Port Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) loop to use a local small model as the agent brain. The training loop stays the same — the question is whether a 4B model can make productive research decisions.

---

## Why Start Here

The autoresearch loop is the simplest possible agent:

1. Read a metric (val_bpb)
2. Decide what to change in one file (train.py)
3. Run for 5 minutes
4. Check if it improved
5. Repeat

This constrained action space is deliberately chosen. It's small enough that a 4B model might handle it, and the evaluation is automatic — the metric either improved or it didn't. No subjective judgment, no human scoring. This makes it ideal for running many experiments quickly.

---

## Experiment Matrix

| Agent Brain | Cost | What We Learn |
|---|---|---|
| Frontier model (baseline) | API costs | Upper bound on this task |
| Local 4B, bare | Free | Raw small-model capability |
| Local 4B + chain-of-thought | Free | Does explicit reasoning help? |
| Local 4B + RE2 prompting | Free | Does re-reading context help? |
| Local 4B + self-consistency (vote on 3 proposals) | Free | Does sampling diversity help? |
| Local 4B + constrained actions (menu of changes) | Free | Does reducing action space beat improving reasoning? |
| Local 4B + agent adapter | Free | Does a specialist adapter outperform prompting? |

Every row after the first is free to run. This is the same LocoLLM advantage applied to agent research.

---

## What Success Looks Like

The frontier model baseline will produce some number of improving iterations out of N attempts. Success for the local model is not necessarily matching that number — it's understanding the gap and what closes it.

Key questions:

- Does the local model ever produce a genuine improvement, or does it drift immediately?
- Which scaffolding strategies most reduce the gap to the frontier baseline?
- Is there a consistent failure mode (e.g., always fails at the planning step, or always generates syntactically invalid changes)?
- Does an agent-trained adapter meaningfully outperform prompting strategies?

Even if the local model achieves 0% success on bare runs, that's a useful baseline for measuring the scaffolding strategies.

---

## Research Context

Karpathy's autoresearch demonstrated that LLM agents can autonomously conduct ML research by modifying code, running experiments, and iterating on results. The key design insight: constrain the action space (one file, one metric, fixed time budget) to make the loop tractable.

The original used frontier models. Nobody has published results from running the same loop with a sub-7B model.

---

## Status

Phase 1 (Semester 2, 2026). Not yet started.

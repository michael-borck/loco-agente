---
title: "Track D: Framework Evaluation"
description: "Does framework choice meaningfully affect small-model agent performance?"
---

# Track D: Framework Evaluation

Most comparisons between agentic frameworks are conducted on frontier models, where framework overhead is negligible. With SLMs, every token in the context window matters. LocoAgente asks whether framework choice meaningfully affects small-model agent performance.

---

## The Central Hypothesis

Minimalist orchestrators (NanoClaw-style, hand-rolled loops) outperform full-featured frameworks (LangGraph, CrewAI) with local models because they produce smaller, tighter system prompts and fewer injected tokens per loop iteration.

If true, the implication is that SLM agent deployments should lean toward minimal scaffolding rather than adopting frameworks designed for frontier models.

---

## Why This Matters

Framework comparisons are always done on frontier models. The overhead of a LangGraph system prompt or a CrewAI role description is a rounding error when you have 128K tokens of context. But a 4B model with an effective 4K-8K context window? A bloated system prompt could consume 20-30% of usable context before the agent even starts working.

This is a genuinely unstudied question. Nobody has measured it.

---

## Evaluation Dimensions

### Context Bloat

Tokens injected per loop iteration by the framework versus a minimal hand-rolled loop. Measured by instrumenting the prompt at each step.

### Tool-Call Accuracy

Does prompt inflation from framework boilerplate hurt tool selection? Measured by comparing tool-call correctness on identical tasks across frameworks.

### Drift Rate

How quickly does the agent lose goal coherence across turns under each framework? Measured using the drift metrics from [Agentic Drift](agentic-drift).

### Failure Modes

Do framework-managed agents fail differently than minimalist agents? Qualitative analysis of failure patterns — does LangGraph's structure prevent certain failures? Does CrewAI's role framing introduce new ones?

---

## Experiment Matrix

| Framework | Overhead | What We Learn |
|---|---|---|
| Hand-rolled loop (baseline) | Minimal | Lower bound on prompt bloat |
| NanoClaw | Low | Does a principled minimal framework match hand-rolled? |
| LangGraph | Medium | Does structured graph orchestration help or hurt SLMs? |
| CrewAI | High | Does multi-agent role framing help or hurt SLMs? |

All four run the same task, same model, same hardware. The only variable is the framework.

---

## The LocoBench Handoff

Track D produces qualitative findings and initial measurements. When results are mature enough for rigorous, reproducible benchmarking, the methodology and runs move to LocoBench. LocoAgente generates the experiments; LocoBench produces the numbers.

---

## Open Questions

- Is there a "LocoAgente-native" minimal framework that captures the best properties without depending on external projects?
- Does the overhead penalty change with model size? (Is it worse at 3B than 7B?)
- Can framework overhead be reduced by custom system prompts, or is it structural?

---

## Dependencies

Can start alongside Track A — requires only a working agent loop, which can be implemented independently in each framework. Benefits from Track C findings about which scaffolding strategies to include in each framework's configuration.

---

## Status

Phase 1-2. Framework comparison can begin as soon as the baseline hand-rolled loop exists.

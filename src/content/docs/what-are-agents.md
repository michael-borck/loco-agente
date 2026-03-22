---
title: "What Are Agents?"
description: "Background on agentic AI — what it is, how it works, and how it differs from single-turn interaction."
---

# What Are Agents?

A language model agent is a system that uses a language model to observe its environment, decide what to do, take action, and evaluate the result — in a loop.

This is fundamentally different from the way most people use language models. A chatbot answers one question. An agent pursues a goal across multiple steps, making decisions and adjusting based on outcomes.

---

## The Agent Loop

Every agentic system, from simple scripts to complex frameworks, follows the same basic pattern:

1. **Observe** — read the current state (files, metrics, outputs, error messages)
2. **Plan** — decide what to do next based on the goal and current state
3. **Act** — execute a change (modify code, run a command, call a tool)
4. **Evaluate** — check whether the action moved toward the goal
5. **Iterate** — loop back to step 1

The intelligence lives in the model. The framework provides the loop.

---

## What Makes This Hard

Each step in the loop is a reasoning task. The model must understand state, make a judgment, express that judgment as a structured action, and then interpret the result.

For frontier models with hundreds of billions of parameters, this works because they have massive capacity for in-context reasoning. They can hold the goal, the history, and the current state simultaneously, and they produce coherent next-actions most of the time.

For small models, the challenge is that **errors compound across the loop**. If a model makes the right decision 95% of the time on a single step, that sounds good. But across a 5-step loop:

| Per-step accuracy | 1 step | 3 steps | 5 steps | 10 steps |
|-------------------|--------|---------|---------|----------|
| 95% | 95% | 86% | 77% | 60% |
| 90% | 90% | 73% | 59% | 35% |
| 85% | 85% | 61% | 44% | 20% |

A model that seems reliable on single turns becomes unreliable across a loop. This is the core challenge LocoAgente investigates.

---

## Agents vs Chatbots

| | Chatbot | Agent |
|---|---------|-------|
| **Interaction** | Single turn or short conversation | Multi-step loop pursuing a goal |
| **State** | Conversation history | Environment state (files, metrics, systems) |
| **Actions** | Generates text | Calls tools, modifies systems, runs code |
| **Evaluation** | Human judges quality | Automatic (metric improved, test passed, etc.) |
| **Failure mode** | Bad answer | Compounding errors, drift, stuck loops |

---

## Tool Calling

Agents interact with their environment through **tool calls** — structured requests to read a file, run a command, query a database, or perform some other action. The model doesn't execute the action directly; it produces a structured output (typically JSON) describing what it wants to do, and the framework executes it.

For LocoAgente, tool calling is treated as a controlled variable — solved infrastructure, not a research question. We use guided decoding and flat JSON schemas to ensure reliable tool calls, and focus the research on the reasoning and planning layers above.

---

## Agentic Frameworks

Several frameworks exist to provide the loop structure, tool integration, and state management that agents need:

- **Hand-rolled loops** — custom Python scripts with minimal abstraction
- **NanoClaw** — minimalist, containerised, self-isolated agent architecture
- **LangGraph** — graph-based orchestration with structured state transitions
- **CrewAI** — multi-agent systems with role-based coordination
- **OpenClaw** — full-featured agent framework (assumes frontier models)

These frameworks differ in how much overhead they add — how many tokens they inject into prompts, how they manage state, and how much abstraction they provide. For small models where every token in the context window counts, this overhead may matter more than it does for frontier models. This is one of LocoAgente's research questions (see [Track D: Framework Evaluation](track-d-framework-eval)).

---

## Further Reading

- [Agentic Drift](agentic-drift) — why agents lose coherence and what to do about it
- [Scaffolding Strategies](scaffolding-strategies) — techniques for improving agent reliability
- [Why LocoAgente](why-locoagente) — why study this on small local models

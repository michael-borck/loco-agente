---
title: "Agentic Drift"
description: "How and why agents lose coherence — error compounding, hallucination amplification, and goal drift in multi-step loops."
---

# Agentic Drift

Agentic drift is what happens when a language model agent loses the plot. It stops pursuing the original goal, starts hallucinating actions or state, or gets stuck in a loop doing the same thing repeatedly. It is the central failure mode of agentic AI, and it is dramatically worse with small models.

---

## Why Drift Happens

### Error Compounding

Every step in an agent loop is a reasoning task. Every reasoning task has a nonzero error rate. Across a loop, those errors multiply.

A 5% error rate per step becomes a 23% chance of failure over 5 steps. A 15% error rate becomes a 56% chance. For small models with higher baseline error rates than frontier models, the compounding is brutal.

The implication: strategies that barely matter for single-turn quality can be critical for loops. A prompting technique that improves per-step accuracy from 85% to 92% changes 5-step success from 44% to 66%.

### Hallucination Amplification

In single-turn use, a hallucination is a wrong answer. In an agent loop, a hallucination becomes an action — and that action changes the state the model observes on the next turn. The model is now reasoning about a world it corrupted.

This is particularly dangerous when the model hallucinates successful outcomes. If the model "believes" its action worked when it didn't, it proceeds to the next step on a false foundation. Each subsequent step drifts further from reality.

### Goal Drift

Autoregressive language models generate left to right. In a long loop with accumulating context, the original goal gets pushed further from the model's attention window. The model starts optimising for local coherence (making the next action seem reasonable given recent context) rather than global coherence (making progress toward the original goal).

Small models are more susceptible because they have smaller effective context windows and weaker long-range attention. By turn 5, a 4B model may have functionally forgotten what it was trying to do.

### Loop Traps

The model gets stuck repeating an action that doesn't work but doesn't trigger an obvious failure. It retries the same command, re-reads the same file, or makes the same edit. Without an external mechanism to detect repetition, the loop continues indefinitely.

---

## Why Small Models Are More Vulnerable

Frontier models resist drift because they have:

- **Larger context windows** — the original goal stays in attention longer
- **Stronger instruction following** — they maintain goal adherence across turns
- **Better calibration** — they know when they don't know, reducing hallucinated actions
- **More robust reasoning** — each step is more likely to be correct, so compounding is slower

Small models lack all of these advantages. But the question LocoAgente asks is not "do small models drift less?" (they don't). The question is: **can scaffolding strategies reduce drift enough to make the loop viable?**

---

## Mitigation Strategies

Each of these is studied in detail in [Scaffolding Strategies](scaffolding-strategies). The summary:

| Strategy | What It Addresses |
|----------|-------------------|
| **RE2 (Re-Reading)** | Goal drift — forces the model to re-encode the original goal before each step |
| **Guided decoding** | Hallucination — physically constrains output to valid tool calls |
| **Atomic tasking** | Error compounding — fewer reasoning steps per agent invocation |
| **Self-consistency voting** | Per-step accuracy — samples multiple proposals and picks the consensus |
| **Constrained action space** | All of the above — fewer choices means fewer ways to go wrong |
| **Human-in-the-loop checkpoints** | Drift detection — a human verifies state at key points |
| **Temperature zero** | Hallucination — removes sampling randomness for deterministic output |

The research question is which combinations of these strategies produce the best reliability gains for the lowest complexity cost.

---

## Measuring Drift

LocoAgente measures drift across several dimensions:

- **Goal coherence** — is the agent still pursuing the original objective after N turns?
- **Action validity** — are the agent's tool calls syntactically and semantically correct?
- **State accuracy** — does the agent's implicit model of the world match reality?
- **Loop detection** — is the agent repeating actions without progress?
- **Steps to failure** — how many successful steps before the first error?

These metrics are tracked per-experiment and compared across scaffolding strategies and frameworks.

---

## Further Reading

- [What Are Agents?](what-are-agents) — background on the agent loop
- [Scaffolding Strategies](scaffolding-strategies) — techniques for reducing drift
- [Track C: Scaffolding Research](track-c-scaffolding) — the systematic study of what works

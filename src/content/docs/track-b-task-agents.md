---
title: "Track B: Task-Specific Agents"
description: "Extending the agent loop to domain-specific tasks beyond autoresearch."
---

# Track B: Task-Specific Agents

Once the autoresearch baseline is established (Track A), extend the agent loop to other well-scoped tasks. Each task domain gets its own agent adapter and its own evaluation harness.

---

## Target Domains

### Data Analysis Agent

- Read a CSV dataset
- Plan analyses (distributions, correlations, outliers)
- Write and execute Python code
- Check results for correctness
- Iterate on findings

**Evaluation:** Automatic — does the generated code run? Do the statistics match a reference analysis?

### Code Review Agent

- Read a diff
- Identify potential issues (bugs, style, security)
- Suggest specific fixes
- Verify suggestions are syntactically valid

**Evaluation:** Semi-automatic — compare identified issues against a labelled set of known bugs.

### Documentation Agent

- Read source code
- Generate or update documentation
- Verify accuracy against the code
- Check for completeness

**Evaluation:** Automatic for structure (does it cover all public functions?), manual for quality.

---

## Design Pattern

Each task-specific agent follows the same pattern:

1. **Same base model** — Qwen3-4B, Q4_K_M
2. **Task-specific adapter** — trained via LocoLLM's adapter pipeline on domain-relevant data
3. **Domain-constrained action space** — each agent only has tools relevant to its task
4. **Automatic evaluation** — where possible, remove the need for human scoring

The adapter training connects directly to LocoLLM's infrastructure. The adapter is what specialises the base model for the task; the scaffolding strategies (from Track C) are what make the loop reliable.

---

## Cross-Task Questions

- Do scaffolding strategies transfer across domains, or are they task-specific?
- Is adapter training more or less valuable than prompting for task-specific agents?
- Are some task types fundamentally easier for small-model agents than others?

---

## Dependencies

Requires Track A's working agent loop and experiment harness. Can benefit from Track C's scaffolding findings, but can also proceed in parallel using the baseline strategies (CoT, RE2).

---

## Status

Phase 3 (Semester 2, 2027). Waiting on Track A foundation.

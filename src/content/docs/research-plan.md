---
title: "Research Plan"
description: "Four research tracks, phasing, dependencies, and the LocoBench handoff."
---

# Research Plan

LocoAgente is organised into four research tracks and a phased roadmap. The tracks are independent lines of investigation that share infrastructure; the phases sequence the work across semesters.

---

## The Core Question

**The scaffolding hypothesis**: if a 4B model with RE2 + voting can match a 70B model on a single question, can a 4B agent with RE2 + voting + constrained actions match a frontier agent on a well-scoped task?

---

## Four Tracks

### [Track A: Local Autoresearch](track-a-autoresearch)

Port Karpathy's autoresearch loop to a local 4B model. The simplest possible agent — read a metric, change code, run, check, repeat. This is the starting point because the action space is small enough that a 4B model might handle it, and evaluation is automatic.

### [Track B: Task-Specific Agents](track-b-task-agents)

Extend to other well-scoped tasks: data analysis, code review, documentation. Each domain gets its own adapter and evaluation harness. Depends on Track A for the working agent loop.

### [Track C: Scaffolding Research](track-c-scaffolding)

Systematic study of which scaffolding strategies most improve small-model agent reliability. The most publishable track — connects to the Cognitive Strategy Transfer research and the keep-asking nudge work.

### [Track D: Framework Evaluation](track-d-framework-eval)

Does framework choice meaningfully affect SLM agent performance? The hypothesis: minimalist orchestrators outperform full-featured frameworks with local models because of lower prompt overhead. Results feed into LocoBench.

---

## Dependencies

```
Track A (autoresearch port)
   ├── Track B (new task domains — needs the working loop from A)
   ├── Track C (scaffolding — can start alongside A, uses A as test harness)
   └── Track D (framework eval — can start alongside A, compares loop implementations)
```

Track A is the foundation. Tracks B, C, and D build on it but are otherwise independent of each other.

---

## Design Decisions

**Tool calling is a controlled variable.** LocoAgente assumes tool calling works (guided decoding + flat schemas + Pydantic). The research questions are about reasoning in loops and framework overhead, not tool calling reliability. Improvements from LocoLLM or the broader community are imported, not built here.

**Local first.** Everything runs on consumer hardware. Cloud APIs are baseline comparisons, not the system under study.

**Constrained over capable.** Start with the smallest viable action space and expand.

**Measurably better.** Every strategy must demonstrate quantifiable improvement. "It feels smarter" doesn't count.

**Comparable experiments.** Fixed time budgets, fixed evaluation metrics, fixed hardware.

---

## Phased Roadmap

### Phase 1: Foundation (Semester 2, 2026)

- Port autoresearch loop to work with local models
- Implement baseline experiment: frontier model vs bare local model
- Build experiment harness (run, log, compare)
- First scaffolding experiments (CoT, RE2)
- Document findings

### Phase 2: Scaffolding (Semester 1, 2027)

- Systematic scaffolding strategy comparison (Track C)
- Self-consistency voting in agent loops
- Constrained action space experiments
- Train first agent adapter via LocoLLM pipeline
- First publication: scaffolding strategies for small-model agents

### Phase 3: Task Agents (Semester 2, 2027)

- Extend to task-specific agents (Track B)
- Data analysis agent, code review agent
- Human-in-the-loop checkpoint experiments
- Cross-task scaffolding comparison

### Phase 4: Integration (2028+)

- Agent adapters contributed to LocoLLM ecosystem
- Multi-agent coordination on consumer hardware
- Integration with LocoConvoy for multi-GPU agent swarms

---

## The LocoBench Handoff

LocoAgente generates experiments and qualitative findings. When results need rigorous, reproducible measurement — particularly from Track D's framework comparisons — the benchmarking methodology and runs move to LocoBench. LocoAgente asks the questions; LocoBench produces the numbers.

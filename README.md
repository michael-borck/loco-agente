# LocoAgente

<!-- BADGES:START -->
[![agentic](https://img.shields.io/badge/-agentic-blue?style=flat-square)](https://github.com/topics/agentic) [![consumer-hardware](https://img.shields.io/badge/-consumer--hardware-blue?style=flat-square)](https://github.com/topics/consumer-hardware) [![framework-evaluation](https://img.shields.io/badge/-framework--evaluation-blue?style=flat-square)](https://github.com/topics/framework-evaluation) [![language-model](https://img.shields.io/badge/-language--model-blue?style=flat-square)](https://github.com/topics/language-model) [![python](https://img.shields.io/badge/-python-3776ab?style=flat-square)](https://github.com/topics/python) [![research](https://img.shields.io/badge/-research-blue?style=flat-square)](https://github.com/topics/research) [![scaffolding](https://img.shields.io/badge/-scaffolding-blue?style=flat-square)](https://github.com/topics/scaffolding)
<!-- BADGES:END -->

### Local Agentic AI -- Can Small Models Think in Loops?

> *"The frontier labs will never bother asking this question. That's why it matters."*

LocoAgente is a research project investigating whether small language models (3-4B parameters), running on consumer hardware, can perform useful agentic tasks — and what scaffolding strategies make the difference between failure and viability.

Modern agentic AI systems (OpenClaw, Claude Code, Devin, Karpathy's autoresearch) assume frontier models with hundreds of billions of parameters. LocoAgente asks: how far can you get with a 4B model, the right prompting strategies, and a constrained action space? Where is the hard capability floor? And can the same scaffolding strategies that improve single-turn quality (chain-of-thought, RE2 prompting, self-consistency voting) compound to make small-model agents reliable enough to be useful?

This project is part of the [LocoLab](https://locolabo.org) research programme at Curtin University.

## The Problem

Agentic AI requires models that can:
1. **Observe** — read state (files, metrics, outputs)
2. **Plan** — decide what to do next
3. **Act** — execute a change (modify code, run a command, call a tool)
4. **Evaluate** — check whether the action improved things
5. **Iterate** — loop back to step 1

Each step is a reasoning task. Frontier models handle this because they have enormous capacity for in-context reasoning. Small models struggle — not because they can't do any of these steps, but because errors compound across the loop. A 5% error rate per step becomes a 23% chance of failure over 5 steps. A 15% error rate becomes a 56% chance.

The research question is whether scaffolding strategies — prompting techniques, constrained action spaces, voting mechanisms, human-in-the-loop checkpoints — can reduce per-step error rates enough to make the loop viable.

## Why This Matters

**For resource-constrained environments.** Not everyone can afford frontier API costs for agentic workflows. If small local models can handle well-scoped agent tasks, that opens agentic AI to students, small teams, and institutions without cloud budgets.

**For understanding.** The scaffolding strategies that make small agents work are the same strategies that make human delegation effective — clear task descriptions, constrained choices, verification checkpoints, iterative refinement. Studying where small agents fail illuminates what's actually hard about agentic AI vs what's just papered over by scale.

**For the research gap.** Nobody is systematically studying agentic capabilities at the sub-7B scale. The small-model community focuses on single-turn benchmarks. The agent community focuses on frontier models. The intersection is unstudied.

## Research Programme

### Track A: Local Autoresearch (Starting Point)

Port Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) loop to use a local small model as the agent brain. The training loop stays the same — the question is whether a 4B model can make productive research decisions.

The autoresearch loop is the simplest possible agent:
- Read a metric (val_bpb)
- Decide what to change in one file (train.py)
- Run for 5 minutes
- Check if it improved
- Repeat

This constrained action space is deliberately chosen — it's small enough that a 4B model might handle it, and the evaluation is automatic (the metric either improved or it didn't).

**Experiment matrix:**

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

### Track B: Task-Specific Agents

Once the autoresearch baseline is established, extend to other well-scoped agent tasks:

- **Data analysis agent** — Read a CSV, plan analyses, write Python, check results
- **Code review agent** — Read a diff, identify issues, suggest fixes
- **Documentation agent** — Read code, generate/update documentation, verify accuracy

Each task domain gets its own agent adapter (trained via LocoLLM's adapter pipeline) and its own evaluation harness.

### Track C: Scaffolding Strategy Research

The systematic study of what scaffolding strategies most improve small-model agent reliability:

- **Prompting strategies**: CoT, RE2, role prompting, structured output formats
- **Sampling strategies**: Self-consistency voting, best-of-N selection
- **Action space constraints**: Free-form vs structured menus vs binary choices
- **Loop structures**: Single-pass vs iterative refinement vs human-in-the-loop checkpoints
- **Memory strategies**: Full context vs summarised history vs external state files

This is the most publishable track — it connects to the [Cognitive Strategy Transfer](https://locolabo.org) research programme and the [keep-asking](https://locolabo.org) nudge research.

### Track D: Framework Evaluation

Most comparisons between agentic frameworks are conducted on frontier models, where the overhead of any given framework is negligible. With SLMs, every token in the context window matters. LocoAgente asks whether framework choice meaningfully affects small-model agent performance.

**The central hypothesis**: minimalist orchestrators (NanoClaw-style, hand-rolled loops) outperform full-featured frameworks (LangGraph, CrewAI) with local models because they produce smaller, tighter system prompts and fewer injected tokens per loop iteration. If true, the implication is that SLM agent deployments should lean toward minimal scaffolding rather than adopting frameworks designed for frontier models.

**Evaluation dimensions:**

- **Context bloat**: tokens injected per loop iteration by the framework vs a minimal hand-rolled loop
- **Tool-call accuracy**: does prompt inflation from framework boilerplate hurt tool selection?
- **Drift rate**: how quickly does the agent lose goal coherence across turns under each framework?
- **Failure modes**: do framework-managed agents fail differently than minimalist agents?

**Experiment matrix:**

| Framework | Overhead | What We Learn |
|---|---|---|
| Hand-rolled loop (baseline) | Minimal | Lower bound on prompt bloat |
| NanoClaw | Low | Does a principled minimal framework match hand-rolled? |
| LangGraph | Medium | Does structured graph orchestration help or hurt SLMs? |
| CrewAI | High | Does multi-agent role framing help or hurt SLMs? |

Results from Track D feed directly into LocoBench as a benchmarking methodology for agentic framework comparison at the sub-7B scale.

## How It Works

```
                    +-------------------+
   Task / Goal ---->|   Agent Loop      |
                    |                   |
                    +-------------------+
                           |
              +------------+-------------+
              |            |             |
              v            v             v
        +---------+  +-----------+  +----------+
        | Observe |  |   Plan    |  |   Act    |
        | (read   |  | (decide   |  | (modify  |
        |  state) |  |  next     |  |  code,   |
        |         |  |  action)  |  |  run     |
        |         |  |           |  |  command)|
        +---------+  +-----------+  +----------+
              |            |             |
              +------------+-------------+
                           |
                    +------+------+
                    |  Evaluate   |
                    | (did it     |
                    |  improve?)  |
                    +------+------+
                           |
                    +------+------+
                    |  Scaffolding |
                    | (RE2, CoT,   |
                    |  voting,     |
                    |  constraints)|
                    +--------------+
```

The agent brain is a LocoLLM model — the same Qwen3-4B base with task-specific adapters. Scaffolding strategies are applied at each step of the loop, not just at the prompting stage.

## Project Structure

```
loco-agente/
├── README.md
├── LICENSE
├── pyproject.toml
├── src/
│   ├── locoagente/
│   │   ├── __init__.py
│   │   ├── cli.py                  # Command-line interface
│   │   ├── loop.py                 # Core agent loop
│   │   ├── observer.py             # State observation
│   │   ├── planner.py              # Action planning (LLM calls)
│   │   ├── actor.py                # Action execution (sandboxed)
│   │   ├── evaluator.py            # Result evaluation
│   │   └── scaffolding/
│   │       ├── __init__.py
│   │       ├── cot.py              # Chain-of-thought prompting
│   │       ├── re2.py              # RE2 re-reading
│   │       ├── voting.py           # Self-consistency voting
│   │       └── constraints.py      # Action space constraints
│   └── content/
│       └── docs/                   # Astro/Starlight documentation
├── experiments/
│   ├── autoresearch/               # Track A experiments
│   ├── task-agents/                # Track B experiments
│   ├── scaffolding/               # Track C experiments
│   └── framework-eval/             # Track D experiments
├── scripts/
│   ├── run_experiment.py           # Run a single experiment
│   └── compare_results.py         # Compare experiment outcomes
└── tests/
```

## Design Principles

**Local first.** Everything runs on consumer hardware. If it needs a cloud API, it's a baseline comparison, not the system under study.

**Constrained over capable.** Start with the smallest viable action space and expand. A reliable agent with 5 possible actions beats an unreliable agent with 500.

**Measurably better.** Every scaffolding strategy must demonstrate quantifiable improvement. "It feels smarter" doesn't count.

**Comparable experiments.** Fixed time budgets, fixed evaluation metrics, fixed hardware. Every experiment is directly comparable to every other.

**Accumulative.** Each semester adds new scaffolding strategies, new task domains, new adapter types. The experiment harness is the source of truth.

## Connection to LocoLLM

LocoAgente uses LocoLLM's infrastructure:
- **Same base model** (Qwen3-4B, Q4_K_M)
- **Same adapter pipeline** (LoRA via QLoRA, trained on consumer hardware)
- **Same prompting strategies** (RE2, self-consistency voting)
- **Same evaluation philosophy** (measurably better or it doesn't ship)

The difference: LocoLLM asks "can small models answer questions well?" LocoAgente asks "can small models *think in loops*?"

## Roadmap

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

## Research Context

**Karpathy's autoresearch** demonstrated that LLM agents can autonomously conduct ML research by modifying code, running experiments, and iterating on results. The key design insight: constrain the action space (one file, one metric, fixed time budget) to make the loop tractable.

**OpenClaw and NanoClaw** show that agent frameworks are converging on a common pattern: observe → plan → act → evaluate. The intelligence lives in the model; the framework provides the loop.

**LocoLLM's scaffolding research** (RE2, self-consistency voting, adapter specialisation) demonstrates that prompting strategies can significantly close the gap between small and large models on single-turn tasks. The open question is whether these gains compound across multi-step agent loops.

**The scaffolding hypothesis**: if a 4B model with RE2 + voting can match a 70B model on a single question, can a 4B agent with RE2 + voting + constrained actions match a frontier agent on a well-scoped task? That's the question LocoAgente exists to answer.

## FAQ

**Why not just use Claude/GPT as the agent?**
That's the baseline comparison. The research question is whether local models can do useful agent work *without* frontier API costs. If the answer is "no, not yet," that's a publishable negative result.

**Isn't this just LocoLLM with extra steps?**
LocoLLM studies single-turn model quality. LocoAgente studies multi-step reasoning loops. The difference is that errors compound in loops — a model that's 90% accurate per turn is only 59% accurate over 5 turns. Scaffolding strategies that barely matter for single turns might be critical for loops.

**What if small models just can't do this?**
Then we characterise exactly where they fail and publish that. Understanding the capability floor is as valuable as pushing through it. The field needs systematic data on small-model agent limitations, and that data doesn't exist yet.

**How does this relate to the cognitive strategy transfer research?**
The scaffolding strategies we're testing (constrained choices, verification checkpoints, iterative refinement) are the same strategies humans use when delegating to junior team members. The CST framework predicts they should transfer to AI agents for the same structural reasons. LocoAgente is an empirical test of that prediction.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{locoagente2026,
  title={LocoAgente: Local Agentic AI on Consumer Hardware},
  author={Michael Borck and Contributors},
  year={2026},
  url={https://github.com/michael-borck/loco-agente}
}
```

---
title: "Scaffolding Strategies"
description: "Catalogue of strategies for improving small-model agent reliability, with SLM-specific considerations."
---

# Scaffolding Strategies

Scaffolding strategies are techniques applied around a language model to improve its reliability in agentic loops. They don't change the model — they change the environment the model operates in.

For frontier models, many of these strategies provide marginal gains. For small models, they can be the difference between a functional agent and a useless one.

---

## Prompting Strategies

### RE2 (Re-Reading)

**What it is:** Repeat the query within the prompt to force the model to encode it twice.

**The formula:** `[System Prompt] → [Query] → "Read the question again:" → [Query] → [Step-by-step elicitation]`

**Why it helps SLMs:** Autoregressive models generate left to right and can lose early context by the time they start generating. RE2 acts as a pseudo-bidirectional attention mechanism, forcing the model to re-attend to the original instruction without drastically inflating token count.

**Agentic extension:** Apply RE2 not just at the single-turn level but at each step of the agent loop. Before each action, re-present the original goal and the current state. This directly combats goal drift.

### Chain-of-Thought (CoT)

**What it is:** Prompt the model to reason step by step before producing an answer or action.

**Why it helps SLMs:** Explicit reasoning reduces the chance of the model jumping to a wrong conclusion. The intermediate steps serve as a form of working memory, keeping relevant context active during generation.

**Tradeoff:** CoT consumes tokens. In a constrained context window, the reasoning steps compete with history and state for space. The net benefit depends on whether the accuracy gain outweighs the context cost.

### Role Prompting

**What it is:** Assign the model a specific role in the system prompt ("You are a code review agent. Your only job is to identify bugs in the diff provided.").

**Why it helps SLMs:** Narrows the model's generative space. A model prompted as a "code reviewer" is less likely to drift into general conversation or unrelated suggestions than a model prompted as a "helpful assistant."

---

## Output Strategies

### Guided Decoding / Strict JSON Schema

**What it is:** Use the inference engine's grammar enforcement (e.g., `llama.cpp` via Ollama) to physically constrain the model's output to a valid JSON schema. Invalid tokens are masked at the logit level — the model literally cannot produce them.

**Why it helps SLMs:** This is the strongest available mechanism against agentic drift. If the model can only output a valid tool call, it cannot hallucinate a conversational tangent, an unstructured response, or a malformed action.

**Why structured output helps SLMs in general:** In free-form prose, every token is a wide-open prediction from the full vocabulary. In JSON, the prediction space collapses — structural tokens (`{`, `}`, `:`, `,`) are essentially free, so the model spends its limited capacity on content tokens. There is also a training data effect: SLMs have seen enormous amounts of well-formed JSON in code-heavy corpora, so their predictions within that structure are more confident.

**SLM-specific rules:**
1. **Temperature zero** — when you want structured data, you want zero creativity
2. **Prompt reinforcement** — still instruct "Respond in JSON" even with schema enforcement
3. **Keep schemas flat** — SLMs handle flat dictionaries well but struggle with deeply nested recursive structures

### Type-Safe Function Registries

**What it is:** Provide the model with rigid API signatures or TypeScript-style interfaces rather than prose descriptions of available tools.

**Why it helps SLMs:** Reduces the task from "understand what I want and figure out how to express it" to "map this input to one of these strict data types." SLMs excel at the latter.

**Pairs with:** Guided decoding. The schema defines the output shape; the function registry defines what actions are available.

---

## Sampling Strategies

### Self-Consistency Voting

**What it is:** Generate N candidate responses (typically 3-5) and select the most common answer or the consensus action.

**Why it helps SLMs:** Individual samples may be wrong, but if 3 out of 5 agree, the consensus is more likely correct. This is particularly effective for action selection, where the right action is often "obvious" to the model but occasionally disrupted by sampling noise.

**Tradeoff:** Multiplies inference cost by N. On local hardware this means N× latency per step. Whether the reliability gain justifies the time cost is an empirical question.

### Best-of-N Selection

**What it is:** Generate N candidates and select the one that scores highest on some quality metric (e.g., confidence score, output length, validation check).

**Why it differs from voting:** Voting picks the consensus; best-of-N picks the best individual. Useful when there's a quality signal beyond agreement.

---

## Architectural Strategies

### Atomic Tasking

**What it is:** Break multi-step workflows into micro-tasks with clear inputs and outputs. Each agent invocation does one thing.

**Why it helps SLMs:** Agentic drift happens when you give a model a massive, multi-step goal. Atomic tasks reduce the number of reasoning steps per invocation, keeping per-step error rates from multiplying. Each step is independently verifiable.

**Example:** Instead of "Research this company and write a report," use one agent to run the search, another to parse the results, and a third to write the summary.

### Constrained Action Space

**What it is:** Limit the set of actions the agent can take. Instead of "do anything to improve this code," offer a menu: "choose one: add a docstring, rename a variable, extract a function, add error handling."

**Why it helps SLMs:** Fewer choices means fewer ways to go wrong. A reliable agent with 5 possible actions beats an unreliable agent with 500. The model's job shifts from open-ended generation to selection from a defined set — a much easier task for small models.

### SLM-Default, LLM-Fallback

**What it is:** Use the local model for the majority of tasks. If the model fails a validation check or encounters ambiguity beyond its capability, escalate to a frontier model.

**Why it helps:** Captures the cost and privacy benefits of local inference for the 80% of tasks that are routine, while maintaining quality on the 20% that are genuinely hard.

**Open questions:** What triggers escalation? Can the threshold be learned? What's the cost profile of an 80/20 split vs 100% frontier?

---

## Loop Strategies

### Human-in-the-Loop Checkpoints

**What it is:** Pause the agent loop at defined points and present the current state to a human for verification before continuing.

**Why it helps:** Catches drift before it compounds. A human checking state after step 3 prevents 7 more steps of divergent execution.

**Tradeoff:** Breaks autonomy. Useful during development and for high-stakes tasks; less practical for fully autonomous workflows.

### Iteration Limits and Loop Detection

**What it is:** Set hard limits on loop iterations and detect repetitive actions.

**Why it helps SLMs:** Prevents the model from burning cycles in a loop trap — repeating the same failing action indefinitely. Simple to implement, essential for unattended operation.

---

## Strategy Interactions

These strategies are not independent. Some combinations reinforce each other:

- **Guided decoding + function registries** — the schema constrains shape; the registry constrains semantics
- **Atomic tasking + constrained actions** — each micro-task has a small action space
- **RE2 + human checkpoints** — re-reading maintains goal coherence between checkpoints
- **Self-consistency voting + temperature > 0** — voting requires diverse samples, which requires non-zero temperature (in tension with the "temperature zero" rule for structured output)

Understanding these interactions is part of [Track C: Scaffolding Research](track-c-scaffolding).

---

## Further Reading

- [Agentic Drift](agentic-drift) — the problems these strategies address
- [Track C: Scaffolding Research](track-c-scaffolding) — systematic evaluation of strategy effectiveness
- [Seed Ideas](seed-ideas) — raw discussion notes where many of these strategies were first captured

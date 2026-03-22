---
title: Seed Ideas
description: Captured ideas and strategies not yet placed in the project structure.
---

# Seed Ideas

Raw ideas captured from early discussions. These are not commitments — they are starting points for experiments, architecture decisions, or documentation as the project takes shape.

---

## Tool Calling Is a Controlled Variable, Not a Research Question

LocoAgente assumes tool calling is solved (or at least good enough). The research question is "can small models think in loops?" — not "can small models call tools?"

**In practice this means:**

1. Use guided decoding + flat schemas + Pydantic as settled infrastructure, not experimental variables
2. Constrain experiments to simple, reliable tool interfaces (e.g., autoresearch only needs "read metric" and "edit file")
3. Import improvements from elsewhere — when LocoLLM or the broader community improves SLM tool calling, pull those gains in rather than building them here

**Why this matters:** It sharpens the project scope. Scaffolding strategies (Track C) and framework overhead (Track D) are the genuine research contributions. Tool calling is plumbing. Holding it constant makes experiments cleaner and results more attributable to the variables we're actually studying.

---

## SLM-Default, LLM-Fallback

Use the local Ollama instance as the "worker" for the majority of repetitive agentic tasks: routing, formatting data, standard tool calling. If the SLM fails a validation check or encounters a highly ambiguous prompt, escalate to a frontier model.

This treats the local model as the fast, cheap default — not a lesser version of the frontier model. The fallback is a design choice, not an admission of failure.

**Open questions:**
- What validation checks trigger escalation?
- Can the escalation threshold be learned over time?
- What's the cost/latency profile of a 80/20 local/frontier split vs 100% frontier?

---

## Atomic Tasking

Agentic drift happens when you give an SLM a massive, multi-step overarching goal. The fix is to break workflows into micro-tasks with clear inputs and outputs.

Instead of "Research this company and write a report," have one agent run the search, another parse the HTML, and a third write the summary. Each step is independently verifiable.

This connects to the error-compounding problem already documented in the README — atomic tasks reduce the number of reasoning steps per agent invocation, keeping per-step error rates from multiplying.

---

## Guided Decoding / Strict JSON Schema Enforcement

SLMs drift when they are allowed to generate free-form prose. By using strict JSON Schema enforcement (supported by Ollama and vLLM), you physically constrain the model's generation space. If the model is forced at the system level to only output a valid JSON tool-call, it literally cannot hallucinate a conversational tangent.

This is the strongest available mechanism against agentic drift — it operates at the decoding level, not the prompting level.

**Open questions:**
- What's the quality/reliability tradeoff of strict vs flexible schemas?
- How does guided decoding interact with chain-of-thought (which needs free-form reasoning)?
- Can you alternate between constrained (action) and unconstrained (reasoning) generation steps?

---

## Type-Safe Function Registries

Instead of explaining desired behaviour in paragraphs of text, provide the model with rigid TypeScript interfaces or exact API signatures in the system prompt. SLMs perform exceptionally well when their primary job is mapping an input to a strict data type.

This pairs with guided decoding — the schema defines the output shape, and the function registry defines what actions are available and how to call them.

---

## RE2 for Agentic Loops (Extension)

RE2 (Re-Reading) is already documented as a prompting strategy. The extension idea: apply RE2 not just at the single-turn level, but at each step of the agentic loop. Before each action, re-present the original goal and the current state.

*The formula:* `[System Prompt] -> [Query] -> "Read the question again:" -> [Query] -> [Step-by-step elicitation]`

This acts as a pseudo-bidirectional attention mechanism — particularly effective for SLMs that lose early context during long autoregressive generation.

---

## The Assembly Line Framing

The mindset shift for local agentic AI: move away from "giving an open-ended assistant a complex job" and toward "building a highly structured digital assembly line."

Frontier models can figure things out on the fly. SLMs require strict guardrails and narrow scopes. This is not a limitation — it's a design constraint that produces more predictable, auditable, and debuggable systems.

This framing connects several strategies:
- **Atomic tasking** → each station on the line has one job
- **Guided decoding** → each station can only produce valid output
- **Function registries** → each station has a fixed set of tools
- **SLM-default/LLM-fallback** → escalation when a station can't handle the work

---

## NanoClaw over OpenClaw for Local Models

NanoClaw's containerised, self-isolated architecture makes it easier to sandbox local models and restrict their execution environment. For local agentic work, minimalist frameworks may outperform monolithic ones — not because they're more capable, but because they keep models focused and minimise blast radius when drift occurs.

**Open question:** Is there a "LocoAgente-native" minimal framework that captures these properties without depending on external projects?

---

## Why Structured Output Helps SLMs (Not Just How)

Empirical observation: small models produce noticeably better results when outputting JSON rather than free-form text, even before grammar enforcement is applied. The likely explanation is about prediction space.

In free-form prose, every token position is wide open — the model chooses from its entire vocabulary based on loose statistical patterns. Thousands of plausible next tokens at each step, each constraining the next only weakly. That's where ambiguity, drift, and hallucination accumulate.

In JSON, the prediction space collapses at each step:

- After `"name":` the model knows it needs `"` (a string value)
- After `{` it knows it needs a key from the schema
- After a value, it needs `,` or `}`
- Structural tokens (`{`, `}`, `:`, `,`, `"`) are essentially free — the model barely has to reason about them

The result: the model spends its limited reasoning capacity on **content tokens** (the actual values) rather than wasting it on sentence structure, word order, transitions, and all the other open-ended choices in prose. Less work per meaningful output token.

There is also likely a **training data effect** — JSON appears heavily in code-oriented training corpora with consistent, repetitive patterns. SLMs have seen enormous amounts of well-formed JSON, so their predictions within that structure are more confident and calibrated than in open prose.

This is the conceptual bridge between the practical guardrails (flat schemas, temperature zero) and the assembly-line framing. Structure doesn't just constrain the model — it *helps* the model. Grammar enforcement via guided decoding takes this one step further by guaranteeing the structural tokens, freeing the model's entire capacity for content prediction.

**Testable hypothesis:** For the same task, compare SLM output quality (correctness, relevance) in three conditions: (1) free-form prose, (2) JSON via prompt instruction only, (3) JSON via guided decoding. Expect improvement at each step, with the largest jump between (1) and (2).

---

## Guided Decoding in Practice (Ollama)

Concrete implementation notes for the guided decoding idea above. When Ollama receives a JSON schema via the `format` parameter, it uses `llama.cpp`'s grammar engine to physically mask token logits that would violate the schema. This is not prompt-level guidance — it is decoding-level constraint. The model cannot produce invalid output.

### Python + Pydantic (Recommended Path)

```python
from ollama import chat
from pydantic import BaseModel

class ToolCall(BaseModel):
    action: str
    target: str
    confidence: int

response = chat(
    model='llama3.2',
    messages=[
        {'role': 'system', 'content': 'You are a task router. Respond in JSON.'},
        {'role': 'user', 'content': 'Summarise the latest sales report.'}
    ],
    format=ToolCall.model_json_schema(),
    options={'temperature': 0}
)

result = ToolCall.model_validate_json(response.message.content)
```

Pydantic generates the JSON schema; Ollama enforces it at the token level; `model_validate_json` guarantees type-safe parsing. This is the pattern for every agent action step.

### Raw API (Any Language)

```bash
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" -d '{
  "model": "llama3.2",
  "messages": [
    {"role": "system", "content": "Respond using JSON."},
    {"role": "user", "content": "Route this task: summarise sales report."}
  ],
  "stream": false,
  "options": {"temperature": 0},
  "format": {
    "type": "object",
    "properties": {
      "action": {"type": "string"},
      "target": {"type": "string"},
      "confidence": {"type": "integer"}
    },
    "required": ["action", "target", "confidence"]
  }
}'
```

### SLM-Specific Guardrails

Three rules that matter more for small models than frontier models:

1. **Temperature zero** — when you want structured data, you want zero creativity. Prevents the model from sampling "interesting" tokens that disrupt the JSON.
2. **Prompt reinforcement** — even with schema enforcement at the decoding level, always instruct the model to "Respond in JSON" in the system prompt. The model's internal weights still need to be aligned with the task.
3. **Keep schemas flat** — an 8B model will easily map inputs to a flat dictionary, but loses reasoning capability with deeply nested, recursive schemas. Keep data structures shallow.

### Open Questions

- Guided decoding guarantees valid *structure*, but not valid *content*. The model can output `{"action": "fly_to_moon"}` and it's valid JSON. How do you validate semantic correctness?
- What's the performance overhead of grammar-constrained decoding on consumer GPUs?
- Can schemas be dynamically generated based on available tools (connecting this to the function registry idea)?

---
title: "Future Work"
description: "Ideas and open questions not yet part of the active research tracks."
---

# Future Work

Ideas captured from early discussions that are not yet part of the four research tracks. These are starting points for future experiments or documentation — kept here so they don't get lost.

When an idea matures into an experiment or gets absorbed into a track, it should be removed from this page.

---

## Why Structured Output Helps SLMs (Theory)

Empirical observation: small models produce noticeably better results when outputting JSON rather than free-form text, even before grammar enforcement is applied. The likely explanation is about prediction space.

In free-form prose, every token position is wide open — the model chooses from its entire vocabulary based on loose statistical patterns. Thousands of plausible next tokens at each step, each constraining the next only weakly. That's where ambiguity, drift, and hallucination accumulate.

In JSON, the prediction space collapses at each step:

- After `"name":` the model knows it needs `"` (a string value)
- After `{` it knows it needs a key from the schema
- After a value, it needs `,` or `}`
- Structural tokens (`{`, `}`, `:`, `,`, `"`) are essentially free — the model barely has to reason about them

The result: the model spends its limited reasoning capacity on **content tokens** (the actual values) rather than wasting it on sentence structure, word order, transitions, and all the other open-ended choices in prose. Less work per meaningful output token.

There is also likely a **training data effect** — JSON appears heavily in code-oriented training corpora with consistent, repetitive patterns. SLMs have seen enormous amounts of well-formed JSON, so their predictions within that structure are more confident and calibrated than in open prose.

Structure doesn't just constrain the model — it *helps* the model.

**Testable hypothesis:** For the same task, compare SLM output quality (correctness, relevance) in three conditions: (1) free-form prose, (2) JSON via prompt instruction only, (3) JSON via guided decoding. Expect improvement at each step, with the largest jump between (1) and (2). This could be a standalone experiment or fold into Track C.

---

## Guided Decoding Implementation Patterns (Ollama)

Concrete implementation notes for when the project reaches the coding phase. When Ollama receives a JSON schema via the `format` parameter, it uses `llama.cpp`'s grammar engine to physically mask token logits that would violate the schema. This is decoding-level constraint — the model cannot produce invalid output.

### Python + Pydantic

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

---

## Open Questions

Questions surfaced during early discussions that don't have a home in the current tracks yet:

- **Semantic validation:** Guided decoding guarantees valid *structure*, but not valid *content*. The model can output `{"action": "fly_to_moon"}` and it's valid JSON. How do you validate that the content is meaningful?
- **Grammar overhead:** What's the performance cost of grammar-constrained decoding on consumer GPUs? Does it slow generation enough to matter?
- **Dynamic schemas:** Can schemas be generated at runtime based on available tools, connecting guided decoding to a function registry automatically?
- **CoT + guided decoding tension:** Chain-of-thought needs free-form reasoning, but guided decoding constrains output to structured JSON. Can you alternate between unconstrained (reasoning) and constrained (action) generation steps within a single loop iteration?
- **Escalation learning:** In the SLM-default/LLM-fallback pattern, can the escalation threshold be learned from experience rather than hardcoded?

# LocoLLM as the Inference Backend

LocoAgente's `Inference` subsystem speaks OpenAI-compatible HTTP (see `harness/inference/client.py`). Any backend that exposes that contract drops in as the inference layer — Ollama, llama.cpp server, vLLM, SGLang, or **LocoLLM**.

LocoLLM serves models via Ollama (per ADR-0006, "GGUF/Ollama as inference standard"), which exposes an OpenAI-compatible endpoint. That makes the integration mechanical: point LocoAgente at LocoLLM's Ollama URL, name the served model, and the harness does not know — and does not need to know — that adapter routing or specialist selection is happening on the other side.

## Configuration

```python
from harness.inference.client import OpenAICompatibleClient

client = OpenAICompatibleClient(
    base_url="http://localhost:11434/v1",   # LocoLLM's Ollama endpoint
    api_key="ollama",                        # any non-empty string; Ollama ignores
    model="qwen3-4b-locollm-router",         # the LocoLLM-served model name
)
```

The [Inference subsystem](../architecture/four-subsystems.md#inference) treats this client like any other. Orchestration patterns, Context, and Tools are unaffected — they only see `client.complete(prompt, **sampling_params)`.

## Why this composition

LocoAgente's research surface is multi-turn orchestration (Debate, Perspective, Synthesis, Iterative Refinement). LocoLLM's research surface is single-turn specialisation (adapter training, routing, base model selection). They compose:

- **LocoAgente without LocoLLM** — harness on a generic Ollama-served base model. Default development setup.
- **LocoAgente with LocoLLM** — harness on a routed set of task-specific adapters. Each call inside an Orchestration pattern can be served by a different specialist, decided by LocoLLM.

The integration is a config change, not a code change. That is the point of treating Inference as a swappable subsystem.

## Boundaries

From LocoAgente's view, the backend is opaque:

- LocoAgente does **not** select adapters; LocoLLM does, by routing internally to the right specialist per request.
- LocoAgente does **not** decide quantisation or base model; that is LocoLLM's choice and is invisible at the HTTP boundary.
- LocoLLM does **not** see Orchestration state or conversation history; it sees individual prompt-completion pairs, one per harness call.

The clean boundary is the OpenAI-compatible chat-completions contract. Both sides can evolve independently as long as that contract holds.

## Further reading

- [LocoLLM](https://locollm.org) — project site; see ADR-0006 ("GGUF/Ollama inference standard") and ADR-0003 ("single evolving router")
- [Four subsystems](../architecture/four-subsystems.md) — where Inference sits in the LocoAgente harness
- [LocoLab: Technique before scale](https://locolabo.org/the-loco-thesis#technique-before-scale) — how the projects compose at the lab level

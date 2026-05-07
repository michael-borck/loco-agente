# LocoAgente

<!-- BADGES:START -->
[![agentic](https://img.shields.io/badge/-agentic-blue?style=flat-square)](https://github.com/topics/agentic) [![consumer-hardware](https://img.shields.io/badge/-consumer--hardware-blue?style=flat-square)](https://github.com/topics/consumer-hardware) [![framework-evaluation](https://img.shields.io/badge/-framework--evaluation-blue?style=flat-square)](https://github.com/topics/framework-evaluation) [![language-model](https://img.shields.io/badge/-language--model-blue?style=flat-square)](https://github.com/topics/language-model) [![python](https://img.shields.io/badge/-python-3776ab?style=flat-square)](https://github.com/topics/python) [![research](https://img.shields.io/badge/-research-blue?style=flat-square)](https://github.com/topics/research) [![scaffolding](https://img.shields.io/badge/-scaffolding-blue?style=flat-square)](https://github.com/topics/scaffolding)
<!-- BADGES:END -->

### Local Conversational Harness — Small Models as Thinking Partners, Not Substitutes

> *"Frontier models win on precision. Small local models, properly harnessed, can win on variance — and variance is what creativity needs."*

LocoAgente is a research project building a **conversational harness** for small local models — a four-subsystem architecture (Orchestration / Context / Tools / Inference) that amplifies human thinking through verified offloading rather than substituting for it.

The harness's foundational primitive: **always produce N alternatives with rationale and surfaced uncertainty.** Singular outputs are forbidden at the primitive level. Variance is engineered, not hoped for.

## The three load-bearing principles

1. **Creativity does not need precision; it needs ideas.** Stop trying to make small models precise; channel their variance.
2. **Conversation, not delegation — but verified offloading is fine.** The harness makes the verification loop cheap.
3. **Confidence is not competence.** Surface uncertainty as a first-class signal.

See [`docs/philosophy/`](docs/philosophy/) for the full statements.

## Two demos

- **Perspective Debate** (Phase 1, end-to-end): N business / academic / writing frames debate a brief over R rounds. See [`docs/tutorials/perspective-debate.md`](docs/tutorials/perspective-debate.md).
- **Research Synthesis** (Phase 1 skeleton, Phase 2 functional): cross-disciplinary scan of a folder-of-papers corpus. See [`docs/tutorials/research-synthesis.md`](docs/tutorials/research-synthesis.md).

## Quick start

```bash
git clone https://github.com/michael-borck/loco-agente && cd loco-agente
uv venv .venv
uv pip install -e '.[dev]' --python .venv/bin/python
ollama pull qwen3:4b

.venv/bin/locoagente debate \
    --brief "Should we ship without a GDPR review?" \
    --frames "ceo,legal,marketing" \
    --rounds 3 \
    --profile-root ./profiles/business \
    --rules "never_fabricate,disagree_explicitly" \
    --output ./out/debate.json

.venv/bin/locoagente log --trace ./out/debate.json
```

## Architecture

See [`docs/architecture/four-subsystems.md`](docs/architecture/four-subsystems.md) and the design spec at [`docs/superpowers/specs/2026-05-07-conversational-harness-design.md`](docs/superpowers/specs/2026-05-07-conversational-harness-design.md).

## Connection to the loco fleet

- **loco-llm** — provides base model + adapters; the harness consumes them
- **loco-bench** — answers "which inference engine + model fits my VRAM tier"
- **loco-puente** — deployment substrate (if/when the harness gets a web UI)

## License

MIT.

## Citation

```bibtex
@software{locoagente2026,
  title={LocoAgente: Conversational Harness for Small Local Models},
  author={Michael Borck and Contributors},
  year={2026},
  url={https://github.com/michael-borck/loco-agente}
}
```

# Tutorial: Perspective Debate

You will run the Perspective Debate demo end-to-end against a small local model. You will see how the harness produces structured `Conversation` traces, surface the verification hooks each frame produced, and understand how to swap profiles for different audiences.

## Prerequisites

1. Python 3.10+
2. [uv](https://github.com/astral-sh/uv) installed
3. [Ollama](https://ollama.ai) installed and running locally
4. A small model pulled into Ollama:

   ```bash
   ollama pull qwen3:4b
   ```

## Set up

```bash
cd loco-agente
uv venv .venv
uv pip install -e '.[dev]' --python .venv/bin/python
```

## Run a debate

```bash
.venv/bin/locoagente debate \
    --brief "Should we ship the GDPR-flagged feature on Tuesday?" \
    --frames "ceo,legal,marketing" \
    --rounds 3 \
    --profile-root ./profiles/business \
    --rules "never_fabricate,disagree_explicitly" \
    --output ./out/debate.json
```

## Inspect the trace

```bash
.venv/bin/locoagente log --trace ./out/debate.json
```

Output sections:
- **Run ID, Pattern, Profile, Brief** — basic provenance
- **Round 0** — three variants (CEO, legal, marketing). Each has text, rationale, confidence, flags, verification hooks
- **Round 1, Round 2** — same three frames, but each variant now sees the prior rounds and can push back
- **User decisions** — empty initially; populated when you start integrating the trace with a UI

## What to look for

- Are the three frames distinguishable? Read the rationales — does each one *sound* like that frame?
- Are the disagreements explicit? Look for places where round 1 or 2 calls back to round 0 and argues against it
- Did any variant produce a verification hook? Are those hooks things you would actually check?

## Try a different audience

Switch the profile bundle:

```bash
.venv/bin/locoagente debate \
    --brief "Is this study design adequate to support the claim?" \
    --frames "methodologist,theorist,practitioner" \
    --rounds 3 \
    --profile-root ./profiles/academic \
    --rules "cite_or_flag,explicit_uncertainty_on_citations" \
    --output ./out/academic_debate.json
```

The same `DebatePattern` runs; only the profile and frames change.

## Common issues

- **Model returns no `<variant>` tags.** Some models are bad at structured output. Try a different model (`LOCOAGENTE_MODEL=phi-4-mini` or larger). Watch for the parser raising `VariantParseError`.
- **All three frames sound the same.** Frame collapse — the model isn't following the personality files. Try a more capable model, or rewrite the personalities to be more distinct.
- **Connection refused.** Ollama isn't running, or the model isn't pulled. Check `ollama list` and `curl http://localhost:11434/v1/models`.

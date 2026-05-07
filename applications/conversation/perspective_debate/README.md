# Perspective Debate Demo

The Phase 1 headline demo. Three business frames (CEO, legal counsel, marketing director) debate a brief over multiple rounds. The harness produces a structured `Conversation` trace surfacing each frame's perspective, where they disagree, what each is uncertain about, and what the human should verify.

## Run it

Prerequisite: an OpenAI-compatible inference backend running locally. Default is Ollama at `http://localhost:11434/v1` with model `qwen3:4b`.

### Via CLI:

```bash
.venv/bin/locoagente debate \
    --brief "Should we ship the feature without GDPR review?" \
    --frames "ceo,legal,marketing" \
    --rounds 3 \
    --profile-root ./profiles/business \
    --rules "never_fabricate,disagree_explicitly" \
    --output ./out/debate.json

.venv/bin/locoagente log --trace ./out/debate.json
```

### Via demo runner:

```bash
.venv/bin/python -m applications.conversation.perspective_debate.runner \
    "Should we ship the feature without GDPR review?"
```

## What success looks like

- Three personalities are distinguishable (blind test: a reader can guess which frame produced which variant)
- Disagreements are surfaced explicitly, not glossed
- Uncertainty hooks fire when the model claims something verifiable

## Customisation

- Different frames: pass `--frames "founder,investor,customer"` (after creating those personality files in `profiles/business/personality/`)
- Different audience: switch profile bundle with `--profile-root ./profiles/academic` and use academic frame names (e.g., `methodologist,theorist,practitioner`)
- Different model: set `LOCOAGENTE_MODEL=phi-4-mini` etc.

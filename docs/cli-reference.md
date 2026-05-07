# CLI Reference

## `locoagente debate`

Run a perspective debate over multiple rounds.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--brief` | yes | — | The question or topic. |
| `--frames` | yes | — | Comma-separated frame names (e.g., `ceo,legal,marketing`). |
| `--rounds` | no | `3` | Number of debate rounds. |
| `--profile-root` | yes | — | Path to the profile bundle directory. |
| `--rules` | no | (none) | Comma-separated rule names from the profile. |
| `--output` | yes | — | Where to write the trace JSON. |

## `locoagente synthesise`

Run a cross-disciplinary synthesis. One variant per discipline.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--brief` | yes | — | The research question. |
| `--disciplines` | yes | — | Comma-separated disciplines. |
| `--corpus-root` | no | (none) | Optional corpus directory with one subdir per discipline. |
| `--profile-root` | yes | — | Path to the profile bundle. |
| `--rules` | no | `cite_or_flag,explicit_uncertainty_on_citations` | Comma-separated rule names. |
| `--output` | yes | — | Where to write the trace JSON. |

## `locoagente log`

Print a human-readable summary of a Conversation trace.

| Flag | Required | Description |
|---|---|---|
| `--trace` | yes | Path to a Conversation trace JSON. |

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LOCOAGENTE_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible endpoint. |
| `LOCOAGENTE_API_KEY` | `ollama-local` | API key (Ollama doesn't check it). |
| `LOCOAGENTE_MODEL` | `qwen3:4b` | Model name as the inference backend reports it. |

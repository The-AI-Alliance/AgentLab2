# Round 1 — budget + ordinal guidance validation

**Code delta since round 0**: session-local config only (`round_1/exp_config.py`)

## Hypothesis (before run)

If we raise `max_steps` to 20 and add explicit ordinal guidance in the agent prompt, we should eliminate avoidable failures seen in Round 0 (`choose-date`, `search-engine`) without regressing easy tasks.

## Experiment (planned)

- **exp_config**: `exp_config.py` (this dir)
- **Tasks**: click-button, click-checkboxes, enter-text, focus-text, choose-date, choose-list, search-engine, book-flight
- **Agent / model**: Genny default + `AUTO_CUBE_MODEL` (default `claude-haiku-4-5`)
- **Infra**: local miniwob server
- **Step budget**: `AUTO_CUBE_MAX_STEPS` (default 20)

## Planned run commands

```bash
SESSION_IN_REPO="/Users/shaileshnanisetty/cube-harness/.local/auto_cube/miniwob-local-r0"

# Set to a funded model/provider before running.
export AUTO_CUBE_MODEL="claude-haiku-4-5"
export AUTO_CUBE_MAX_STEPS=20

.venv/bin/python "$SESSION_IN_REPO/journal/round_1/exp_config.py" --limit 8
```

Then investigate:

```bash
EXP_DIR="<new round-1 experiment dir from output>"
unset ANTHROPIC_API_KEY

ch-investigate run "$EXP_DIR" \
  --driver claude-code-sdk \
  --journal-dir "$SESSION_IN_REPO/journal" \
  --context-dir "$SESSION_IN_REPO"
```

## Status

Prepared and ready to execute once Anthropic API billing/rate limits are available (or with another funded provider via `AUTO_CUBE_MODEL`).

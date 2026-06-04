# Round 0 — baseline miniwob sweep

**Code delta since round -1**: none (new session baseline)

## Hypothesis (before run)

A small miniwob slice will produce a usable baseline and at least one concrete failure diagnosis that can be actioned in the next round.

## Experiment

- **exp_config**: `exp_config.py` (this dir)
- **Tasks**: click-button, click-checkboxes, enter-text, focus-text, choose-date, choose-list, search-engine, book-flight
- **Agent / model**: Genny default + `claude-haiku-4-5`
- **Infra**: local miniwob server (no remote infra)
- **Step budget**: `max_steps=10`

## Results (after run)

- **Outcomes**: 6/8 success (75%), 2/8 failure
- **Experiment dir**: `/Users/shaileshnanisetty/cube-harness/.local/auto_cube/miniwob-local-r0/experiments/20260526_141939_Genny-claude-haiku-4-5_miniwob-cube_auto-cube-miniwob-r0_865281e6`
- **Investigator recipe**: `general_blame` (driver `claude-code-sdk`) -> `meta_analysis.md` identified two orthogonal failure modes

## Findings

- **Confirmed:** `choose-date_ep4` failed because `MAX_STEPS_REACHED` at 10, despite correct strategy; this is an agent scaffolding/budget issue, not a capability miss.
- **Likely (single-episode):** `search-engine_ep6` failed due to ordinal interpretation mismatch (natural language 1-index vs DOM 0-index attribute value), attributed to model capability.
- **Positive signal:** six tasks solved quickly with no notable loop instability, indicating baseline stack health.

## Conclusion & next

Round 0 met its objective: end-to-end Auto-CUBE experiment + investigation completed with actionable diagnoses. Round 1 should test:
1) higher step budget (`max_steps=20`), and
2) explicit ordinal guidance in benchmark prompt wrapper.

Round 1 config is prepared but execution is pending Anthropic API billing/rate-limit recovery.

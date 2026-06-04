# Session: miniwob-local-r0

**Objective**: Run Auto-CUBE debug Round 0 on miniwob, complete investigation, and stage next-round fixes
**Benchmark**: miniwob-cube
**Branch**: dev
**Base commit**: unknown (session artifacts only)
**Started**: 2026-05-26

## Rounds

| # | Hypothesis (1 line) | Outcome |
|---|---|---|
| 0 | Baseline miniwob slice will expose at least one scaffolding issue and one model issue | Complete: 6/8 success; 2 diagnosed failures with concrete interventions |
| 1 | Increasing step budget + ordinal clarification should reduce avoidable failures | Prepared (config ready), blocked on Anthropic API credits/rate limits |

## Narrative

- Round 0 experiment completed on 8 tasks with Genny + Claude Haiku 4.5 and `max_steps=10`.
- Investigator successfully processed all 8 episodes and produced `meta_analysis.{md,json}`.
- Findings:
  - `choose-date` failure was a hard `max_steps` truncation (strategy was correct, budget too low).
  - `search-engine` failure was ordinal indexing confusion (1-indexed task wording vs 0-indexed DOM attribute).
- Prepared Round 1 config to test these interventions:
  - increase `max_steps` to 20,
  - append explicit ordinal indexing hint in benchmark prompt,
  - keep task slice fixed for comparability.
- Session paused before Round 1 execution due to Anthropic API out-of-credits / rate-limit status.

## Open questions / next

- Execute Round 1 once billing limits are available and compare against Round 0 on the same task list.
- If `choose-date` flips to success after budget increase, evaluate whether a task-aware step-budget policy should be proposed upstream.
- Re-run `search-engine` with more episodes to verify whether ordinal clarification reduces this failure mode beyond n=1.

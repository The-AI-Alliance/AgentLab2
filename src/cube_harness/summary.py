"""Experiment-level summary model.

Per-episode stats live inside `EventStreamer` (folded incrementally as
events flow through) and are persisted on `TrajectoryMetadata.summary_stats`.

This module is what remains after the SummaryProcessor + per-event jsonl
were collapsed into the streamer. It carries one model:

- `ExperimentSummary` — the per-experiment rollup (counts, totals, avg
  reward). Written by `FileStorage.update_experiment_summary` and read
  by XRay's experiment table.
"""

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ExperimentSummary(BaseModel):
    """Cumulative experiment-level totals across all finalized episodes.

    Updated by `FileStorage.update_experiment_summary(meta)` on each
    `finalize_episode`. Persisted at `<output_dir>/summary.json`. XRay's
    experiment-table uses this for O(1) rendering (no per-episode load).
    """

    model_config = ConfigDict(strict=True, extra="forbid", populate_by_name=True)

    n_episodes: int = 0
    n_completed: int = 0
    n_errored: int = 0
    total_reward: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost: float = 0.0
    updated_at: str | None = None
    # Previously named "success_rate" — actually avg reward, not a success rate
    avg_reward: float = Field(0.0, validation_alias=AliasChoices("avg_reward", "success_rate"))

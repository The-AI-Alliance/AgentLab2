import threading
import time
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from cube.core import EnvironmentOutput
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from cube_harness.core import (
    AgentErrorEvent,
    AgentOutput,
    EvaluationEvent,
    LLMCallEvent,
    ToolCallEvent,
    TrajectoryEvent,
    TrajectoryMetadata,
    TrajectoryStep,
)

if TYPE_CHECKING:
    from cube_harness.storage import FileStorage


class EpisodeStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class StepSummary(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    turn: int
    timestamp: float
    status: EpisodeStatus
    n_env_steps: int
    n_agent_steps: int
    total_actions: int
    total_llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    tokens: int
    cost_usd: float
    reward: float
    done: bool


class ExperimentSummary(BaseModel):
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


class SummaryProcessor:
    def __init__(self, episode_dir: Path) -> None:
        self._summary_path = episode_dir / "episode_summary.jsonl"
        self._n_env_steps = 0
        self._n_agent_steps = 0
        self._n_evaluations = 0
        self._total_actions = 0
        self._total_llm_calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cached_tokens = 0
        self._cache_creation_tokens = 0
        self._cost_usd = 0.0
        self._reward = 0.0
        self._done = False
        self._error_type: str | None = None
        # Parallel tool dispatch (GennyParallel + asyncio.to_thread)
        # fires `on_event` from multiple worker threads concurrently
        # for the same SummaryProcessor instance. Without this lock the
        # read-modify-write of counters races (undercounted steps,
        # dropped error_type, garbled / duplicate rows in the
        # episode_summary.jsonl stream).
        self._lock = threading.Lock()

    def _build_entry(self, turn: int, status: EpisodeStatus) -> StepSummary:
        return StepSummary(
            turn=turn,
            timestamp=time.time(),
            status=status,
            n_env_steps=self._n_env_steps,
            n_agent_steps=self._n_agent_steps,
            total_actions=self._total_actions,
            total_llm_calls=self._total_llm_calls,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            tokens=self._prompt_tokens + self._completion_tokens,
            cost_usd=self._cost_usd,
            reward=self._reward,
            done=self._done,
        )

    def _append(self, entry: StepSummary) -> None:
        with open(self._summary_path, "a") as f:
            f.write(entry.model_dump_json() + "\n")

    def _fold_llm_call(self, call) -> None:
        """Accumulate one LLM call's usage. Must be called under `self._lock`.

        `call` is `None` only on legacy V1/V2 decode (the on-disk step
        doesn't carry an LLMCall record); count the turn but skip usage.
        """
        self._n_agent_steps += 1
        self._total_llm_calls += 1
        if call is None:
            return
        if call.usage:
            self._prompt_tokens += call.usage.prompt_tokens
            self._completion_tokens += call.usage.completion_tokens
            self._cached_tokens += call.usage.cached_tokens
            self._cache_creation_tokens += call.usage.cache_creation_tokens
            self._cost_usd += call.usage.cost

    def on_step(self, step_num: int, step: TrajectoryStep) -> None:
        """DEPRECATED: legacy `TrajectoryStep` accumulator. No live caller
        — replaced by `on_event` (event-stream model). Retained only for
        the legacy V2 read path that materializes `TrajectoryStep`s from
        events. Will be removed alongside `_events_to_legacy_steps` in
        the XRay-rewrite follow-up PR."""
        with self._lock:
            if isinstance(step.output, AgentOutput):
                self._n_agent_steps += 1
                self._total_actions += len(step.output.actions)
                if step.output.error is not None and self._error_type is None:
                    self._error_type = step.output.error.error_type
            elif isinstance(step.output, EnvironmentOutput):
                self._n_env_steps += 1
                self._reward = step.output.reward
                self._done = step.output.done
                if step.output.error is not None and self._error_type is None:
                    self._error_type = step.output.error.error_type

            self._append(self._build_entry(step_num, EpisodeStatus.RUNNING))

    def on_event(self, event: TrajectoryEvent) -> None:
        """Accumulate per-event stats for the streaming event model.

        LLMCallEvent → counts as one agent turn; folds token / cost
                       tally from `event.output.call.usage`.
        ToolCallEvent → counts as one env step.
        EvaluationEvent → records terminal reward.
        AgentErrorEvent → captures the first agent-side failure.

        Thread-safe: the lock guards the read-modify-write on the
        counters AND the append-to-jsonl. GennyParallel dispatches N
        tool calls via asyncio.to_thread → each worker thread calls
        on_event concurrently. Without the lock, counter increments
        race and the jsonl gets garbled rows.
        """
        with self._lock:
            # The summary-jsonl `turn` counter walks through events 1:1.
            turn_n = self._n_agent_steps + self._n_env_steps + self._n_evaluations
            out = event.output
            if isinstance(out, LLMCallEvent):
                self._fold_llm_call(out.call)
                if out.error is not None and self._error_type is None:
                    self._error_type = out.error.error_type
            elif isinstance(out, ToolCallEvent):
                self._n_env_steps += 1
                if out.action is not None:
                    self._total_actions += 1
                if out.error is not None and self._error_type is None:
                    self._error_type = out.error.error_type
            elif isinstance(out, EvaluationEvent):
                self._n_evaluations += 1
                self._reward = out.reward
                if out.is_terminal:
                    self._done = True
            elif isinstance(out, AgentErrorEvent):
                if self._error_type is None:
                    self._error_type = out.error.error_type
            self._append(self._build_entry(turn_n, EpisodeStatus.RUNNING))

    @property
    def has_error(self) -> bool:
        return self._error_type is not None

    @property
    def final_reward(self) -> float:
        """Reward of the most recent environment step (the trajectory's final reward)."""
        return self._reward

    def summary_stats(self, *, duration: float | None, final_reward: float) -> dict:
        """Final per-episode stats, accumulated incrementally — the single source of
        truth (replaces the old end-of-run walk over ``trajectory.steps``)."""
        return {
            "n_env_steps": self._n_env_steps,
            "n_agent_steps": self._n_agent_steps,
            "total_actions": self._total_actions,
            "total_llm_calls": self._total_llm_calls,
            "duration": duration,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "cached_tokens": self._cached_tokens,
            "cache_creation_tokens": self._cache_creation_tokens,
            "cost": self._cost_usd,
            "final_reward": final_reward,
            "error_type": self._error_type,
        }

    def on_episode_complete(self, meta: TrajectoryMetadata, storage: "FileStorage") -> None:
        """Finalize the per-episode summary stream and roll into the
        experiment-level summary. `meta` carries the just-finalized
        TrajectoryMetadata whose `summary_stats` was filled by
        `summary_stats(...)` above."""
        status = EpisodeStatus.FAILED if self.has_error else EpisodeStatus.DONE
        self._append(self._build_entry(-1, status))
        storage.update_experiment_summary(meta)

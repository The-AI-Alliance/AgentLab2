import threading
import time
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from cube.core import EnvironmentOutput
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from cube_harness.core import (
    AgentEvent,
    AgentOutput,
    EvaluationEvent,
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

    def on_step(self, step_num: int, step: TrajectoryStep) -> None:
        with self._lock:
            # Capture the first step-level error so EpisodeRecord can report it without
            # re-walking the (now un-retained) step list.
            err = getattr(step.output, "error", None)
            if err is not None and self._error_type is None:
                self._error_type = err.error_type

            if isinstance(step.output, AgentOutput):
                self._n_agent_steps += 1
                self._total_actions += len(step.output.actions)
                self._total_llm_calls += len(step.output.llm_calls)
                for llm_call in step.output.llm_calls:
                    if llm_call.usage:
                        self._prompt_tokens += llm_call.usage.prompt_tokens
                        self._completion_tokens += llm_call.usage.completion_tokens
                        self._cached_tokens += llm_call.usage.cached_tokens
                        self._cache_creation_tokens += llm_call.usage.cache_creation_tokens
                        self._cost_usd += llm_call.usage.cost
            elif isinstance(step.output, EnvironmentOutput):
                self._n_env_steps += 1
                self._reward = step.output.reward
                self._done = step.output.done

            self._append(self._build_entry(step_num, EpisodeStatus.RUNNING))

    def on_event(self, event: TrajectoryEvent) -> None:
        """Accumulate per-event stats for the agent-owns-loop event model.

        AgentEvent  → counts as one agent step (and folds llm/cost tally).
        ToolCallEvent → counts as one env step; carries reward from the
                        underlying EnvironmentOutput (zero at the tool
                        boundary unless task.step wrapped it).
        EvaluationEvent → records the terminal reward.

        Errors on AgentEvent / ToolCallEvent are captured the same way
        as in `on_step`.

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
            if isinstance(out, AgentEvent):
                self._n_agent_steps += 1
                self._total_actions += len(out.actions)
                self._total_llm_calls += len(out.llm_calls)
                for llm_call in out.llm_calls:
                    if llm_call.usage:
                        self._prompt_tokens += llm_call.usage.prompt_tokens
                        self._completion_tokens += llm_call.usage.completion_tokens
                        self._cached_tokens += llm_call.usage.cached_tokens
                        self._cache_creation_tokens += llm_call.usage.cache_creation_tokens
                        self._cost_usd += llm_call.usage.cost
                if out.error is not None and self._error_type is None:
                    self._error_type = out.error.error_type
            elif isinstance(out, ToolCallEvent):
                # ToolCallEvent now carries only obs + error (reward lives on
                # the sibling EvaluationEvent; done is a TaskDone signal).
                self._n_env_steps += 1
                if out.error is not None and self._error_type is None:
                    self._error_type = out.error.error_type
            elif isinstance(out, EvaluationEvent):
                self._n_evaluations += 1
                # Terminal evaluation: overrides reward, marks done.
                # Step-wise evaluation: also surfaces the reward so the
                # summary tracks the latest validate_per_step result.
                self._reward = out.reward
                if out.is_terminal:
                    self._done = True
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

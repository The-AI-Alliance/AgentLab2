"""EventStreamer — the trajectory's event fan-out + per-episode stats.

The streamer is attached to each event-producing component:

  - `LLM.attach_recorder(streamer)` — every `.call()` emits an
    `LLMCallEvent`. The streamer stashes the latest `LLMCallEvent.id`
    so subsequent `ToolCallEvent`s can parent under it.
  - `MonitoredTool` (Episode-installed) — every `.execute_action()`
    emits a `ToolCallEvent` with `parent_event_id` resolved via the
    streamer's `current_parent_event_id()` getter.

From the agent's POV the streamer is invisible: code is just
`await self.llm.call(prompt)` + `await env_tool.execute_action(action)`.

Episode-only helpers (`record_reset`, `record_failure`,
`record_evaluation`) remain on this object — they emit synthetic events
at trajectory boundaries that no component naturally owns.

Per-episode stats (n_llm_calls, n_tool_calls, n_evaluations,
total_actions, tokens, cost, first error, ...) are folded INSIDE the
streamer as events flow through. `summary_stats()` returns the final
dict — what previously lived on a separate `SummaryProcessor`. No
per-event jsonl is written; the dict in `TrajectoryMetadata` is the
single source of truth.

`EventStreamerConfig` is the forward seam for multi-sink fan-out:
FileStorage today; OTel + RL HTTP + custom sinks land via additive
config fields without changing this surface.
"""

import logging
import threading
import time
from typing import Protocol, runtime_checkable

from cube.core import Action, EnvironmentOutput, StepError, TypedBaseModel

from cube_harness.core import (
    AgentErrorEvent,
    EvaluationEvent,
    LLMCallEvent,
    ToolCallEvent,
    TrajectoryEvent,
)
from cube_harness.llm import LLMCall
from cube_harness.tool import BudgetExceeded

logger = logging.getLogger(__name__)

# A sentinel parent_event_id we attach to events emitted before any
# agent turn (the synthetic reset event recorded by Episode).
RESET_PARENT_EVENT_ID = "reset"


@runtime_checkable
class EventSink(Protocol):
    """The forward seam for trajectory event consumers.

    Anything that receives `TrajectoryEvent`s as the agent runs (today:
    `FileStorage`; tomorrow: an OTel span emitter, an RL trainer HTTP
    pump, a Kafka publisher) implements this Protocol. `EventStreamer`
    fans every event out to its registered sinks; sinks read what they
    understand and ignore the rest.

    Sinks MUST be cheap and non-blocking. `emit()` is called on the agent
    loop's hot path under the streamer's stats-lock; a slow sink stalls
    every parallel tool dispatch in `Genny[parallel_actions=True]`. If a sink needs I/O
    (HTTP, disk-fsync), do it in a background queue.

    The structural typing here matches `FileStorage.save_event` exactly,
    which is why the streamer didn't need an interface to ship the first
    sink. Declaring the Protocol makes future sinks self-documenting.
    """

    def save_event(self, te: TrajectoryEvent, trajectory_id: str) -> None: ...


class EventStreamerConfig(TypedBaseModel):
    """Configuration for the per-episode `EventStreamer`.

    Forward seam — empty today. Phase 1 ships FileStorage as the only
    sink (always on). Future fields:

      * `enable_otel: bool` — emit each event as an OTel span.
      * `rl_http_endpoint: str | None` — POST events to an RL trainer.
      * `extra_sinks: list[SinkConfig]` — user-defined sinks.
    """


class EventStreamer:
    """The trajectory's event fan-out + stats accumulator.

    Built by Episode; attached to event producers (LLM, env_tool) via
    their respective `attach_recorder` methods. Producers call
    `emit(te)` which:

      1. Folds stats counters (under a lock to be safe under parallel
         tool dispatch from Genny[parallel_actions=True]'s asyncio.to_thread workers).
      2. Forwards to every sink (currently FileStorage; OTel + RL HTTP
         land via `EventStreamerConfig`).
      3. Returns the event id.

    `current_parent_event_id()` returns the id of the most recent
    `LLMCallEvent`, or `RESET_PARENT_EVENT_ID` if no LLM call has
    fired yet. Used by `MonitoredTool`'s `parent_event_id_getter` so
    each ToolCallEvent parents under the LLM call that spawned it.
    """

    def __init__(
        self,
        trajectory_id: str,
        storage: object | None = None,
        budget: object | None = None,
        metadata_updates: dict | None = None,
    ) -> None:
        self.trajectory_id = trajectory_id
        self.storage = storage
        # `cube_harness.tool.Budget`; loose-typed to avoid circular import.
        self.budget = budget
        # Mutable side-channel dict passed from Episode; merged into
        # TrajectoryMetadata.metadata at finalize. Writes from inside a
        # parallel-dispatch worker (e.g. an asyncio.to_thread tool call
        # via Genny[parallel_actions=True]) MUST hold `self._lock` — the dict itself
        # has no internal synchronization. Single-threaded callers can
        # write directly.
        self.metadata_updates = metadata_updates if metadata_updates is not None else {}
        # Sinks list — storage is the canonical sink-0, additional sinks
        # (OTel emitter, RL HTTP pump, ...) register post-construction
        # via `streamer._sinks.append(sink)`. Tests pass duck-typed
        # objects with `save_event`; the Protocol check is structural,
        # not nominal.
        self._sinks: list[EventSink] = []
        if storage is not None and hasattr(storage, "save_event"):
            self._sinks.append(storage)
        self._current_parent_event_id: str | None = None
        # Last tool call event id stamped on the terminal EvaluationEvent
        # so the final reward attaches to the agent turn that ended the
        # episode (the agent's last action) instead of by trailing
        # position. None until the first tool call fires.
        self._last_tool_call_event_id: str | None = None
        # Stats counters folded as events flow through. Lock guards the
        # multi-counter read-modify-write under parallel dispatch.
        self._lock = threading.Lock()
        self._n_llm_calls = 0
        self._n_tool_calls = 0
        self._n_evaluations = 0
        self._total_actions = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cached_tokens = 0
        self._cache_creation_tokens = 0
        self._cost_usd = 0.0
        self._reward = 0.0
        self._done = False
        self._error_type: str | None = None

    # ----- single fan-out entry point ---------------------------------------

    def emit(self, te: TrajectoryEvent) -> str:
        """Fold stats + forward to sinks. Returns the event's id.

        Sole event-flow entry point. LLM producers, MonitoredTool, and
        the Episode-only boundary helpers all funnel through here so
        every event flows through the same stats fold + sink fan-out.

        Sink fan-out is sequential and best-effort: a sink that raises
        is logged and skipped so a misbehaving downstream consumer
        (slow HTTP, full disk) can't kill the trajectory. Sinks SHOULD
        be cheap; for blocking I/O, queue inside the sink.
        """
        with self._lock:
            self._fold_stats(te.output)
        for sink in self._sinks:
            try:
                sink.save_event(te, self.trajectory_id)
            except Exception:
                logger.exception("EventStreamer sink %r raised; continuing.", sink)
        # EvaluationEvent doesn't carry an `id` field (parent_event_id
        # links it to a ToolCallEvent or it's terminal). Return empty
        # string for those so producers that don't need the id don't
        # have to special-case.
        return getattr(te.output, "id", "")

    def _fold_stats(self, out: object) -> None:
        """Update per-episode counters from one event. Must hold `self._lock`."""
        if isinstance(out, LLMCallEvent):
            self._n_llm_calls += 1
            if out.call is not None and out.call.usage is not None:
                u = out.call.usage
                self._prompt_tokens += u.prompt_tokens
                self._completion_tokens += u.completion_tokens
                self._cached_tokens += u.cached_tokens
                self._cache_creation_tokens += u.cache_creation_tokens
                self._cost_usd += u.cost
            if out.error is not None and self._error_type is None:
                self._error_type = out.error.error_type
        elif isinstance(out, ToolCallEvent):
            self._n_tool_calls += 1
            if out.action is not None:
                self._total_actions += 1
            if out.error is not None and self._error_type is None:
                self._error_type = out.error.error_type
            # Track the latest tool call so the terminal EvaluationEvent
            # can attach exactly to it.
            self._last_tool_call_event_id = out.id
        elif isinstance(out, EvaluationEvent):
            self._n_evaluations += 1
            self._reward = out.reward
            if out.is_terminal:
                self._done = True
        elif isinstance(out, AgentErrorEvent):
            if self._error_type is None:
                self._error_type = out.error.error_type

    # ----- producer-facing hook (called by LLM.call() auto-emit) ------------

    def on_llm_call(
        self,
        call: LLMCall,
        profiling: dict[str, tuple[float, float]] | None = None,
        error: StepError | None = None,
    ) -> str:
        """Emit one `LLMCallEvent`, bump Budget LLM-usage counters
        (cost + tokens), enforce caps. Returns the event id (also
        stashed as the active parent_event_id for subsequent tool calls).

        Does NOT bump `budget.agent_steps` — turn-counting is per agent step,
        not per LLM call (one step may make 0..N calls). The agent loop
        calls `on_step` per iteration instead.
        """
        event = LLMCallEvent(call=call, profiling=dict(profiling or {}), error=error)
        self._current_parent_event_id = event.id
        start, end = self._llm_window(profiling)
        if self.budget is not None and call.usage is not None:
            self.budget.bump_llm_usage(
                cost=call.usage.cost,
                prompt=call.usage.prompt_tokens,
                completion=call.usage.completion_tokens,
            )
        self.emit(TrajectoryEvent(output=event, start_time=start, end_time=end))
        # Enforce AFTER emit so the LLM call that crossed the cap is on
        # disk before we abort. Mirrors what MonitoredTool does on
        # tool dispatch.
        if self.budget is not None and self.budget.exhausted:
            raise BudgetExceeded()
        return event.id

    def on_step(self) -> None:
        """Bump `budget.agent_steps` and enforce. Called by the agent loop
        once per `self.step(obs)` iteration. Turn-counting is per-step,
        NOT per-LLM-call — a step that makes 3 LLM calls (Genny:
        compact + summarize + act) bumps `agent_steps` by exactly 1.

        Enforcement happens here so a `max_agent_steps` cap kicks in cleanly
        at the agent-step boundary."""
        if self.budget is not None:
            self.budget.bump_agent_step()
            if self.budget.exhausted:
                raise BudgetExceeded()

    @staticmethod
    def _llm_window(profiling: dict[str, tuple[float, float]] | None) -> tuple[float, float]:
        if profiling and "llm" in profiling:
            return profiling["llm"]
        now = time.time()
        return now, now

    # ----- Episode-only boundary helpers ------------------------------------

    def record_reset(self, initial: EnvironmentOutput) -> None:
        """Synthetic ToolCallEvent capturing the initial observation from `task.reset()`."""
        synthetic_action = Action(id=RESET_PARENT_EVENT_ID, name="_reset", arguments={})
        event = ToolCallEvent(
            parent_event_id=RESET_PARENT_EVENT_ID,
            action_id=RESET_PARENT_EVENT_ID,
            action=synthetic_action,
            obs=initial.obs,
            error=initial.error,
        )
        ts = time.time()
        self.emit(TrajectoryEvent(output=event, start_time=ts, end_time=ts))

    def record_failure(self, exc: BaseException) -> None:
        """Capture an Episode-level failure as an `AgentErrorEvent`.

        `parent_event_id` attaches the error to the agent turn that
        crashed: the most recent LLM call (if any), else the most
        recent tool call (covers the case where the agent crashed
        post-tool-dispatch before the next LLM call), else None.
        """
        if isinstance(exc, Exception):
            err = StepError.from_exception(exc)
        else:
            err = StepError(
                error_type=type(exc).__name__,
                exception_str=str(exc),
                stack_trace="",
            )
        parent = self._current_parent_event_id or self._last_tool_call_event_id
        event = AgentErrorEvent(error=err, parent_event_id=parent)
        ts = time.time()
        self.emit(TrajectoryEvent(output=event, start_time=ts, end_time=ts))

    def record_evaluation(self, reward: float, info: dict | None = None, *, is_terminal: bool = True) -> None:
        """Record a `task.evaluate()` result.

        Terminal flavor (default): Episode emits exactly one in
        `finally`. `parent_event_id` is the id of the most recent
        `ToolCallEvent` so the final reward attaches to the agent's
        last action; `None` only when no tool call has fired yet.

        The step-wise flavor (`is_terminal=False`) is emitted by
        `MonitoredTool` directly — this API surfaces the terminal path
        only.
        """
        ev = EvaluationEvent(
            reward=float(reward),
            info=dict(info or {}),
            is_terminal=is_terminal,
            parent_event_id=self._last_tool_call_event_id if is_terminal else None,
        )
        ts = time.time()
        self.emit(TrajectoryEvent(output=ev, start_time=ts, end_time=ts))

    # ----- read-only state surfaced for MonitoredTool's getter --------------

    def current_parent_event_id(self) -> str:
        """The id of the most recently emitted LLMCallEvent, or
        `RESET_PARENT_EVENT_ID` if no LLM call has fired yet."""
        return self._current_parent_event_id or RESET_PARENT_EVENT_ID

    # ----- per-episode summary (folded incrementally; queried at finalize) --

    @property
    def has_error(self) -> bool:
        return self._error_type is not None

    @property
    def final_reward(self) -> float:
        """Reward of the most recent EvaluationEvent (the trajectory's final reward)."""
        return self._reward

    def summary_stats(self, *, duration: float | None, final_reward: float) -> dict:
        """Final per-episode stats. Written to `TrajectoryMetadata.summary_stats`
        at finalize. The single source of truth for XRay's per-trajectory
        table (no separate jsonl file is written under the auto-recorder
        model)."""
        with self._lock:
            return {
                "n_env_steps": self._n_tool_calls,
                "n_agent_steps": self._n_llm_calls,
                "total_actions": self._total_actions,
                "total_llm_calls": self._n_llm_calls,
                "duration": duration,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "cached_tokens": self._cached_tokens,
                "cache_creation_tokens": self._cache_creation_tokens,
                "cost": self._cost_usd,
                "final_reward": final_reward,
                "error_type": self._error_type,
            }

"""EventStreamer — the trajectory's event fan-out.

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

`EventStreamerConfig` is the forward seam for multi-sink fan-out:
FileStorage today; OTel + RL HTTP + custom sinks land via additive
config fields without changing this surface.
"""

import logging
import time
from typing import TYPE_CHECKING

from cube.core import Action, EnvironmentOutput, StepError, TypedBaseModel

from cube_harness.core import (
    AgentErrorEvent,
    EvaluationEvent,
    LLMCallEvent,
    ToolCallEvent,
    TrajectoryEvent,
)
from cube_harness.llm import LLMCall
from cube_harness.tool import BudgetExceeded, _stream_event

if TYPE_CHECKING:
    from cube_harness.summary import SummaryProcessor


logger = logging.getLogger(__name__)

# A sentinel parent_event_id we attach to events emitted before any
# agent turn (the synthetic reset event recorded by Episode).
RESET_PARENT_EVENT_ID = "reset"


class EventStreamerConfig(TypedBaseModel):
    """Configuration for the per-episode `EventStreamer`.

    Pydantic config object — follows the same "Python is the config"
    philosophy as `LLMConfig` / `AgentConfig`. Lives on `EpisodeConfig`
    so a recipe can opt sinks in or out without touching Episode.

    Phase 1 (this PR): only the file-storage + summary sinks exist;
    they're always on and don't need configuration. The class exists
    as a forward seam — future fields:

      * `enable_otel: bool` — emit each event as an OTel span.
      * `rl_http_endpoint: str | None` — POST events to an RL trainer
        HTTP endpoint (see `~/dev/cube-harness/docs/rl-integration.md`).
      * `extra_sinks: list[SinkConfig]` — user-defined sinks for
        third-party telemetry pipelines.

    Default `EventStreamerConfig()` is the current Phase-1 behavior:
    FileStorage + SummaryProcessor, nothing else.
    """


class EventStreamer:
    """The trajectory's event sink. Built by Episode; attached to event
    producers (LLM, env_tool) via their respective `attach_recorder`
    methods.

    Cross-turn state (storage, summary, budget) lives on Episode and is
    bound here at construction. Producers stream events through
    `_stream_event` (storage + summary); the recorder also bumps the
    Budget counters and enforces caps after each LLM call.

    `current_parent_event_id()` returns the id of the most recent LLMCallEvent,
    or `RESET_PARENT_EVENT_ID` if no LLM call has fired yet. Used by
    MonitoredTool's parent_event_id_getter.
    """

    def __init__(
        self,
        trajectory_id: str,
        storage: object | None = None,
        summary: "SummaryProcessor | None" = None,
        budget: object | None = None,
        metadata_updates: dict | None = None,
    ) -> None:
        self.trajectory_id = trajectory_id
        self.storage = storage
        self.summary = summary
        # `cube_harness.tool.Budget`; loose-typed to avoid circular import.
        self.budget = budget
        # Mutable side-channel dict passed from Episode; merged into
        # TrajectoryMetadata.metadata at finalize. Connectors that need a
        # back-channel write here.
        self.metadata_updates = metadata_updates if metadata_updates is not None else {}
        self._current_parent_event_id: str | None = None
        self._n_llm_calls_emitted = 0

    # --- producer-facing hook (called by LLM.call() auto-emit) ---

    def on_llm_call(
        self,
        call: LLMCall,
        profiling: dict[str, tuple[float, float]] | None = None,
        error: StepError | None = None,
    ) -> str:
        """Emit one `LLMCallEvent`, bump LLM-usage counters
        (cost + tokens), enforce caps. Returns the event id (also
        stashed as the active turn id for subsequent tool calls).

        Note: does NOT bump `budget.turns` — turn-counting is per
        agent step, not per LLM call (one step may make 0..N calls).
        The agent loop calls `on_step` per iteration instead.
        """
        event = LLMCallEvent(call=call, profiling=dict(profiling or {}), error=error)
        self._current_parent_event_id = event.id
        self._n_llm_calls_emitted += 1
        start, end = self._llm_window(profiling)
        if self.budget is not None and call.usage is not None:
            self.budget.bump_llm_usage(
                cost=call.usage.cost,
                prompt=call.usage.prompt_tokens,
                completion=call.usage.completion_tokens,
            )
        _stream_event(
            TrajectoryEvent(output=event, start_time=start, end_time=end),
            self.trajectory_id,
            self.storage,
            self.summary,
        )
        # Enforce AFTER stream so the LLM call that crossed the cap is
        # on disk before we abort. Mirrors what MonitoredTool does on
        # tool dispatch.
        if self.budget is not None and self.budget.exhausted:
            raise BudgetExceeded()
        return event.id

    def on_step(self) -> None:
        """Bump `budget.turns` and enforce. Called by the agent loop
        once per `self.step(obs)` iteration. Turn-counting is per-step,
        NOT per-LLM-call — a step that makes 3 LLM calls (Genny:
        compact + summarize + act) bumps `turns` by exactly 1.

        Enforcement happens here so a `max_turns` cap kicks in cleanly
        at the agent-step boundary."""
        if self.budget is not None:
            self.budget.bump_turn()
            if self.budget.exhausted:
                raise BudgetExceeded()

    @staticmethod
    def _llm_window(profiling: dict[str, tuple[float, float]] | None) -> tuple[float, float]:
        if profiling and "llm" in profiling:
            return profiling["llm"]
        now = time.time()
        return now, now

    # --- Episode-only helpers (trajectory boundaries) ---

    def record_reset(self, initial: EnvironmentOutput) -> None:
        """Synthetic ToolCallEvent capturing the initial observation from `task.reset()`."""
        synthetic_action = Action(id=RESET_PARENT_EVENT_ID, name="_reset", arguments={})
        event = ToolCallEvent(
            parent_event_id=RESET_PARENT_EVENT_ID,
            action_id=RESET_PARENT_EVENT_ID,
            action=synthetic_action,
            obs=initial.obs,
            error=initial.error,
            turn_id=RESET_PARENT_EVENT_ID,
        )
        ts = time.time()
        _stream_event(
            TrajectoryEvent(output=event, start_time=ts, end_time=ts),
            self.trajectory_id,
            self.storage,
            self.summary,
        )

    def record_failure(self, exc: BaseException) -> None:
        """Capture an Episode-level failure as an `AgentErrorEvent`.

        Accepts BaseException because the Budget/TaskDone signals
        extend BaseException. Does NOT bump budget — we are already
        past the failure point.
        """
        if isinstance(exc, Exception):
            err = StepError.from_exception(exc)
        else:
            err = StepError(
                error_type=type(exc).__name__,
                exception_str=str(exc),
                stack_trace="",
            )
        event = AgentErrorEvent(error=err)
        ts = time.time()
        _stream_event(
            TrajectoryEvent(output=event, start_time=ts, end_time=ts),
            self.trajectory_id,
            self.storage,
            self.summary,
        )

    def record_evaluation(self, reward: float, info: dict | None = None, *, is_terminal: bool = True) -> None:
        """Record a `task.evaluate()` result.

        Terminal flavor (default): Episode emits exactly one in
        `finally`. The step-wise flavor (`is_terminal=False`) is emitted
        by `MonitoredTool` directly through `_record_step_evaluation`
        — this API surfaces the terminal path only.
        """
        ev = EvaluationEvent(reward=float(reward), info=dict(info or {}), is_terminal=is_terminal)
        ts = time.time()
        _stream_event(
            TrajectoryEvent(output=ev, start_time=ts, end_time=ts),
            self.trajectory_id,
            self.storage,
            self.summary,
        )

    # --- getter consumed by MonitoredTool.parent_event_id_getter ---

    def current_parent_event_id(self) -> str:
        """The id of the most recently emitted LLMCallEvent, or
        `RESET_PARENT_EVENT_ID` if no LLM call has fired yet."""
        return self._current_parent_event_id or RESET_PARENT_EVENT_ID

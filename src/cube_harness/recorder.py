"""TurnRecorder — the trajectory's event sink.

Part of RFC `agent-owns-loop`. After the `auto-recorder` follow-up,
agents never call the recorder directly. The recorder is attached to
each event-producing component:

  - `LLM.attach_recorder(recorder)` — every `.call()` emits an
    `LLMCallEvent`. The recorder stashes the latest LLMCallEvent.id as
    the active turn id.
  - `MonitoredTool` (Episode-installed) — every `.execute_action()`
    emits a `ToolCallEvent` with `parent_event_id` / `turn_id` resolved
    via the recorder's `current_turn_id()` getter.

Agent code is reduced to `await self.llm.call(prompt)` +
`await env_tool.execute_action(action)`. Recorder is invisible.

Episode-only helpers (`record_reset`, `record_failure`,
`record_evaluation`) remain on this object — they emit synthetic events
at trajectory boundaries that no component naturally owns.
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


class RecorderConfig(TypedBaseModel):
    """Configuration for the per-episode `TurnRecorder`.

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

    Default `RecorderConfig()` is the current Phase-1 behavior:
    FileStorage + SummaryProcessor, nothing else.
    """


class TurnRecorder:
    """The trajectory's event sink. Built by Episode; attached to event
    producers (LLM, env_tool) via their respective `attach_recorder`
    methods.

    Cross-turn state (storage, summary, budget) lives on Episode and is
    bound here at construction. Producers stream events through
    `_stream_event` (storage + summary); the recorder also bumps the
    Budget counters and enforces caps after each LLM call.

    `current_turn_id()` returns the id of the most recent LLMCallEvent,
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
        self._current_turn_id: str | None = None
        self._n_llm_calls_emitted = 0

    # --- producer-facing hook (called by LLM.call() auto-emit) ---

    def on_llm_call(
        self,
        call: LLMCall,
        profiling: dict[str, tuple[float, float]] | None = None,
        error: StepError | None = None,
    ) -> str:
        """Emit one `LLMCallEvent`, bump Budget (turn + LLM usage), then
        enforce caps. Returns the event id (also stashed as the active
        turn id for subsequent tool calls)."""
        event = LLMCallEvent(call=call, profiling=dict(profiling or {}), error=error)
        self._current_turn_id = event.id
        self._n_llm_calls_emitted += 1
        start, end = self._llm_window(profiling)
        if self.budget is not None:
            usage = call.usage
            self.budget.bump_turn_and_usage(
                cost=usage.cost if usage is not None else 0.0,
                prompt=usage.prompt_tokens if usage is not None else 0,
                completion=usage.completion_tokens if usage is not None else 0,
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

    def current_turn_id(self) -> str:
        """The id of the most recently emitted LLMCallEvent, or
        `RESET_PARENT_EVENT_ID` if no LLM call has fired yet."""
        return self._current_turn_id or RESET_PARENT_EVENT_ID

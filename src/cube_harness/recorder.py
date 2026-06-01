"""TurnRecorder — the agent's outbound telemetry sink.

Part of RFC `agent-owns-loop`. Constructed by `Episode` per-episode and
passed to `agent.run`. The agent reports its internals (LLM calls,
thoughts, response text, profiling, agent-side errors) through this
recorder; the toolbox (`MonitoredTool`) handles tool-call events
separately.

Two complementary APIs on `TurnRecorder`:

* **Coarse** — `record(agent_output)`. One call per LLM cycle, all
  fields at once. The default `Agent.run` (Phase D) uses this — matches
  today's step-style.
* **Granular** — `begin_turn() -> Turn` context manager. Use when you
  want to add events incrementally during a turn (streaming LLM
  responses, mid-turn profiling). On `__exit__` the accumulated state
  flushes as one `AgentEvent`.

Episode-only helpers on the same object: `record_reset`,
`record_failure`, `record_evaluation`. Agents shouldn't call these; the
convention is documented but not actively prevented in v1.

A lossy `record_external_run(final_text, usage, raw_events)` path is
provided for Phase 2 connectors that wrap opaque external frameworks
(CLI agents, A2A endpoints) where per-turn decomposition isn't
possible.
"""

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cube.core import Action, EnvironmentOutput, StepError

from cube_harness.core import (
    AgentEvent,
    AgentOutput,
    EvaluationEvent,
    ToolCallEvent,
    TrajectoryEvent,
)
from cube_harness.llm import LLMCall, Usage

if TYPE_CHECKING:
    from cube_harness.summary import SummaryProcessor


@dataclass
class EventCounter:
    """Monotonic event numbering shared by `TurnRecorder` and `MonitoredTool`.

    Both writers emit `TrajectoryEvent`s onto the same event stream;
    each needs a globally-unique sequence number so files land at
    `events/000_*.msgpack.zst`, `events/001_*.msgpack.zst`, … without
    overwriting each other. Episode constructs one counter per episode
    and hands it to both writers.
    """

    n: int = 0

    def next(self) -> int:
        """Return the current value and increment for the next call."""
        n = self.n
        self.n += 1
        return n


logger = logging.getLogger(__name__)

# A sentinel parent_event_id we attach to events emitted before any
# agent turn (the synthetic reset event recorded by Episode).
RESET_PARENT_EVENT_ID = "reset"


class Turn:
    """Context manager for granular streaming-style event emission.

    Returned from `TurnRecorder.begin_turn()`. Accumulate LLM calls,
    thoughts, response text, profiling, and agent errors on the
    instance; on `__exit__` the recorder flushes them as one
    `AgentEvent`.

    Coding contract: never reuse a Turn after `__exit__`. Re-entering
    is a bug — open a new one for the next turn.
    """

    def __init__(self, recorder: "TurnRecorder") -> None:
        self._recorder = recorder
        self._event = AgentEvent()
        self._start_time = time.time()
        self._closed = False

    @property
    def id(self) -> str:
        """The AgentEvent.id that this turn will flush as — exposed so
        callers (and MonitoredTool's parent_event_id_getter) can read
        the active turn id mid-turn."""
        return self._event.id

    # --- granular adders ---

    def add_llm_call(self, call: LLMCall) -> None:
        """Append one LLMCall record (full prompt / response / usage) to
        the in-progress turn."""
        self._event.llm_calls.append(call)

    def add_thought(self, text: str) -> None:
        """Append a reasoning chunk. Multiple add_thought calls within
        one turn concatenate so streaming reasoning isn't lost."""
        if self._event.thoughts is None:
            self._event.thoughts = text
        else:
            self._event.thoughts += text

    def add_response_text(self, text: str) -> None:
        """Append assistant prose. Concatenates across multiple calls in
        the same turn (streaming-friendly)."""
        if self._event.response_text is None:
            self._event.response_text = text
        else:
            self._event.response_text += text

    def add_action(self, action: Action) -> None:
        """Append one action to the turn's outbound action list."""
        self._event.actions.append(action)

    def add_profile(self, label: str, start: float, end: float) -> None:
        """Record a labelled timing span (used by XRay's profiling tab)."""
        self._event.profiling[label] = (start, end)

    def add_error(self, err: StepError) -> None:
        """Mark this turn as failed with the supplied StepError."""
        self._event.error = err

    # --- context manager ---

    def __enter__(self) -> "Turn":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._closed:
            return
        self._closed = True
        end_time = time.time()
        if exc is not None and self._event.error is None and isinstance(exc, Exception):
            # Capture agent-side exceptions as an event error so the
            # trajectory remains coherent even when the agent crashes
            # mid-turn. We don't try to swallow it — re-raise the original
            # exception after recording.
            self._event.error = StepError.from_exception(exc)
        self._recorder._flush_agent_event(self._event, self._start_time, end_time)


class TurnRecorder:
    """The agent's outbound telemetry sink. Built by Episode; passed to agent.run.

    Cross-turn state (storage, summary, episode metadata) lives on Episode
    and is bound here at construction. The agent never reads any of that
    state directly — only writes to it through this recorder.

    `current_turn_id` is a read-only view onto the active turn's id, used
    by `MonitoredTool`'s parent_event_id_getter so tool calls fired
    inside a turn record the right parent. Outside a turn, it returns
    `RESET_PARENT_EVENT_ID` (anything fired during reset or
    finalization is attributed to that boundary, not to an absent
    agent turn).

    Events stream to disk via `storage.save_event(event, trajectory_id, n)`
    where `n` comes from the shared `EventCounter`. There is no in-memory
    accumulation — `Episode.run` returns an `TrajectoryView` and consumers
    read from storage.
    """

    def __init__(
        self,
        trajectory_id: str,
        storage: object | None = None,
        summary: "SummaryProcessor | None" = None,
        budget: object | None = None,
        event_counter: EventCounter | None = None,
        metadata_updates: dict | None = None,
    ) -> None:
        self.trajectory_id = trajectory_id
        self.storage = storage
        self.summary = summary
        # Budget is `cube_harness.tool.Budget` but we keep the type loose
        # to avoid the circular import (tool imports recorder transitively
        # through MonitoredTool's state). When set, every flushed
        # AgentEvent bumps budget.turns so MonitoredTool's
        # max-turns check fires correctly.
        self.budget = budget
        # Shared with MonitoredTool so both writers emit unique event nums.
        # Default: a recorder-private counter (used by tests that don't
        # install monitoring on a task).
        self.event_counter = event_counter if event_counter is not None else EventCounter()
        # Mutable side-channel dict passed from Episode. record_external_run
        # writes connector-specific data here; Episode merges it into the
        # final TrajectoryMetadata.metadata at finalize_episode time.
        self.metadata_updates = metadata_updates if metadata_updates is not None else {}
        self._current_turn_id: str | None = None
        self._n_turns_emitted = 0

    # --- agent-facing API ---

    def record(self, output: AgentOutput, response_text: str | None = None) -> str:
        """Coarse path — emit one AgentEvent from a complete AgentOutput.

        Returns the AgentEvent.id so callers (Episode's default
        Agent.run loop) can correlate downstream tool calls.
        """
        event = AgentEvent.from_agent_output(output, response_text=response_text)
        start = time.time()
        self._current_turn_id = event.id
        self._flush_agent_event(event, start, start)
        return event.id

    def begin_turn(self) -> Turn:
        """Granular path — open a turn, add events incrementally, flush on close."""
        turn = Turn(self)
        self._current_turn_id = turn.id
        return turn

    # --- lossy capture path for opaque external frameworks (Phase 2) ---

    def record_external_run(
        self,
        final_text: str | None,
        usage: Usage | None = None,
        raw_events: list[dict] | None = None,
    ) -> str:
        """Emit one synthetic AgentEvent summarising an opaque external run.

        Used by Phase-2 connectors (LangGraph, Pydantic AI, Codex CLI,
        A2A …) that can't decompose the agent's execution into per-turn
        events. Carries:

          * `final_text` — the agent's final assistant response.
          * `usage` — token / cost receipt (best-effort, kept in the
            trajectory's metadata under `external_run_usage` because
            `LLMCall` requires a full `llm_config` we don't have here).
          * `raw_events` — opaque blob preserved on the trajectory's
            metadata under `external_run_raw_events`.

        Connectors that CAN observe per-turn events (Pydantic AI,
        LangGraph, OpenAI Agents SDK, Inspect AI) should use `record()`
        or `begin_turn()` instead — this method is the fallback for
        truly opaque frameworks.
        """
        event = AgentEvent(response_text=final_text)
        # Stash side-channel data on the metadata_updates dict — Episode
        # merges it into the final TrajectoryMetadata.metadata at
        # finalize_episode time. Connectors can re-read it post-run
        # (XRay, scoring scripts, ADP export, ...).
        if usage is not None:
            self.metadata_updates.setdefault("external_run_usage", []).append(usage.model_dump(mode="json"))
        if raw_events:
            self.metadata_updates.setdefault("external_run_raw_events", []).extend(raw_events)
        start = time.time()
        self._current_turn_id = event.id
        self._flush_agent_event(event, start, start)
        return event.id

    # --- Episode-only helpers ---

    def record_reset(self, initial: EnvironmentOutput) -> None:
        """Synthetic ToolCallEvent capturing the initial observation from `task.reset()`."""
        synthetic_action = Action(id=RESET_PARENT_EVENT_ID, name="_reset", arguments={})
        event = ToolCallEvent(
            parent_event_id=RESET_PARENT_EVENT_ID,
            action_id=RESET_PARENT_EVENT_ID,
            obs=initial.obs,
            error=initial.error,
            turn_id=RESET_PARENT_EVENT_ID,
        )
        ts = time.time()
        self._append_event(TrajectoryEvent(output=event, start_time=ts, end_time=ts))
        # synthetic_action is only used for debugging logs.
        logger.debug("record_reset action=%r", synthetic_action)

    def record_failure(self, exc: BaseException) -> None:
        """Capture an agent-side or budget failure as a final AgentEvent.

        Episode calls this from its `except` clauses so the failure
        appears in the trajectory and the post-mortem (XRay, summary).
        Accepts BaseException because Budget/Episode-level signals
        (BudgetExceeded, EpisodeDone) extend BaseException.

        A failure event is NOT a normal turn — it doesn't bump
        budget.turns (we're already past the budget) and it doesn't
        update current_turn_id. We append directly.
        """
        # StepError.from_exception requires Exception, not BaseException.
        # Wrap BaseException-only signals so the trajectory still records
        # their type + message without breaking the StepError contract.
        if isinstance(exc, Exception):
            err = StepError.from_exception(exc)
        else:
            err = StepError(
                error_type=type(exc).__name__,
                exception_str=str(exc),
                stack_trace="",
            )
        event = AgentEvent(error=err)
        ts = time.time()
        self._append_event(TrajectoryEvent(output=event, start_time=ts, end_time=ts))

    def record_evaluation(self, reward: float, info: dict | None = None, *, is_terminal: bool = True) -> None:
        """Record a `task.evaluate()` result.

        - Terminal flavor (`is_terminal=True`, default): Episode emits
          exactly one of these in `finally` after `agent.run` returns.
        - Step-wise flavor: `MonitoredTool` emits via the lower-level
          `_record_step_evaluation` helper directly (skips this method)
          so it can pass `parent_event_id`. This API surfaces the
          terminal-eval path only."""
        ev = EvaluationEvent(reward=float(reward), info=dict(info or {}), is_terminal=is_terminal)
        ts = time.time()
        self._append_event(TrajectoryEvent(output=ev, start_time=ts, end_time=ts))

    # --- read-only state surfaced for MonitoredTool's getter ---

    def current_turn_id(self) -> str:
        """The id of the most recently opened AgentEvent. Used by
        MonitoredTool's `parent_event_id_getter` so tool calls fired
        inside a turn record that turn's id as parent. Returns the
        RESET sentinel when no turn has been opened yet."""
        return self._current_turn_id or RESET_PARENT_EVENT_ID

    # --- internals ---

    def _flush_agent_event(self, event: AgentEvent, start: float, end: float) -> None:
        """Persist a finished AgentEvent — bumps `budget.turns` and
        `budget.cost_usd` from the event's LLM calls, raises
        BudgetExceeded if that puts us past any cap, then writes
        through `_append_event` (storage + summary).

        Cost accumulation mirrors what SummaryProcessor does for the
        per-episode summary — same `llm_call.usage.cost` source — but
        also feeds `Budget.cost_usd` so `Budget.exhausted` enforces
        `max_cost_usd` end-to-end."""
        self._n_turns_emitted += 1
        if self.budget is not None:
            # Bump budget.turns so MonitoredTool's exhausted check fires
            # on the right boundary (one LLM turn = one increment).
            self.budget.turns += 1
            # Bump budget.cost_usd from this turn's LLM calls so the
            # max_cost_usd ceiling actually trips. Each LLMCall.usage.cost
            # is the dollar cost the LiteLLM wrapper recorded for that call.
            for call in event.llm_calls:
                if call.usage is not None:
                    self.budget.cost_usd += call.usage.cost
        self._append_event(TrajectoryEvent(output=event, start_time=start, end_time=end))
        # Enforce budget AFTER the flush so the AgentEvent that took us
        # past the cap is recorded before we abort the run. This
        # mirrors what MonitoredTool does on tool dispatch, but covers
        # the case where the task's step() bypasses the toolbox entirely
        # (e.g. tests with hand-rolled task.step that doesn't dispatch
        # to tool.execute_action).
        if self.budget is not None and self.budget.exhausted:
            from cube_harness.tool import BudgetExceeded  # local import: avoid recorder<->tool cycle

            raise BudgetExceeded()

    def _append_event(self, te: TrajectoryEvent) -> None:
        """Stream one event to storage + summary. Never keeps a copy in
        memory — the TrajectoryView is the read interface, this writes."""
        if self.storage is not None:
            save_event = getattr(self.storage, "save_event", None)
            if save_event is not None:
                save_event(te, self.trajectory_id, self.event_counter.next())
        if self.summary is not None:
            on_event = getattr(self.summary, "on_event", None)
            if on_event is not None:
                on_event(te)


def equivalent_agent_events(a: AgentEvent, b: AgentEvent) -> bool:
    """Field-by-field equality ignoring `id` and `profiling`.

    Used by tests asserting `record()` and `begin_turn()` produce the
    same `AgentEvent` for the same payload.
    """
    return (
        a.actions == b.actions
        and a.llm_calls == b.llm_calls
        and a.thoughts == b.thoughts
        and a.response_text == b.response_text
        and a.error == b.error
    )

from typing import Callable
from uuid import uuid4

from cube.core import Action, EnvironmentOutput, Observation, StepError, TypedBaseModel
from pydantic import BaseModel, Field

from cube_harness.llm import LLMCall


def _new_event_id() -> str:
    """Allocate a fresh per-event id used as the AgentEvent.id and the
    `turn_id` of its child ToolCallEvent siblings."""
    return uuid4().hex


class AgentOutput(TypedBaseModel):
    """What `Agent.step()` returns — the actions the agent intends to dispatch
    this turn, and (optionally) an agent-side error.

    Under the post-`agent-owns-loop-auto-recorder` model, the LLM emits its
    own `LLMCallEvent` to the recorder (auto-emit at `LLM.call()` time), so
    the agent does NOT bundle LLM calls or prose into its return value.
    `AgentOutput` is now the minimal "what to dispatch next" payload —
    everything else streams as events.
    """

    actions: list[Action] = Field(default_factory=list)
    error: StepError | None = None

    def __str__(self) -> str:
        return self.model_dump_json()


class TrajectoryStep(TypedBaseModel):
    """DEPRECATED — legacy `EnvironmentOutput | AgentOutput` step; removal
    target: `agent-owns-loop-xray`. New code uses `TrajectoryEvent` and
    `TrajectoryView`; this class is materialized on read by
    `_events_to_legacy_steps` for XRay / investigator until they migrate."""

    output: EnvironmentOutput | AgentOutput
    start_time: float | None = None
    end_time: float | None = None


# --- Event-stream model (RFC: agent-owns-loop + auto-recorder) -------------
# LLMCallEvent / ToolCallEvent / EvaluationEvent are the canonical trajectory
# stream. LLM auto-emits LLMCallEvent on every `.call()` / `.acall()` (when
# a recorder is attached); MonitoredTool auto-emits ToolCallEvent on every
# dispatch. The agent never explicitly records anything — its loop is just
# `llm.acall(...) + env_tool.execute_action(...)`.


class AgentErrorEvent(TypedBaseModel):
    """An agent-side or framework-side failure event.

    Used by Episode + EventStreamer.record_failure to capture exceptions
    that don't naturally belong on an LLMCallEvent or ToolCallEvent
    (BudgetExceeded, agent-side crashes outside an LLM/tool call, etc).
    Carries just the StepError payload — no need for a turn id since the
    failure terminates the trajectory.
    """

    id: str = Field(default_factory=_new_event_id)
    error: StepError


class LLMCallEvent(TypedBaseModel):
    """One LLM API call — the canonical "agent turn" event.

    Replaces the prior batched `AgentEvent`. The shift to one event per LLM
    call drops the explicit Turn context manager: events stream as they
    happen instead of being bundled and flushed at the end of a turn.

    Back-references:

    - `id` becomes the `turn_id` of any `ToolCallEvent` dispatched as a
      direct consequence of this LLM call. The recorder stashes the most
      recent `LLMCallEvent.id` as the current turn id; subsequent
      `MonitoredTool.execute_action` calls inherit it via the recorder's
      `parent_event_id_getter`.
    """

    id: str = Field(default_factory=_new_event_id)
    # Optional only to support the legacy V1/V2-steps decode path, which
    # synthesizes an LLMCallEvent from a step record that has no
    # corresponding LLMCall on disk. Live emissions always set this.
    call: LLMCall | None = None
    # Maps label → (start_time, end_time) as absolute Unix timestamps.
    profiling: dict[str, tuple[float, float]] = Field(default_factory=dict)
    error: StepError | None = None


class ToolCallEvent(TypedBaseModel):
    """One tool invocation — agent's action and what came back to the agent.

    The agent receives `obs` (or `error`) from `MonitoredTool.execute_action`.
    Reward / done / info are NOT part of this event:

    - `done` propagates as a `TaskDone(BaseException)` raised by
      `MonitoredTool` when `task.finished()` returns True. There is no
      `done` field anywhere in the trajectory.
    - Step-wise reward (when `task.validate_per_step=True`) lives on a
      separate `EvaluationEvent` whose `parent_event_id` references this
      `ToolCallEvent.id` and whose `is_terminal` is False.

    Back-references:

    - `parent_event_id` references the originating `LLMCallEvent.id`.
    - `action_id` echoes the `Action.id` emitted by the LLM.
    - `turn_id` groups parallel siblings of a single LLM turn (it equals
      the parent `LLMCallEvent.id` by default — agents emitting N parallel
      tool calls in one turn share that `turn_id`).
    - `action` carries the full `Action` payload so the on-disk trajectory
      is self-contained — no need to cross-reference back to an
      `LLMCallEvent` to know what was dispatched.
    """

    id: str = Field(default_factory=_new_event_id)
    parent_event_id: str
    action_id: str | None = None  # echoes Action.id; nullable for legacy actions
    action: Action | None = None  # full action payload (nullable for legacy decode)
    obs: Observation = Field(default_factory=Observation)  # empty when error is set
    error: StepError | None = None
    turn_id: str


class EvaluationEvent(TypedBaseModel):
    """`task.evaluate()` result. Step-wise OR terminal.

    Emitted in two flavors:

    - **Terminal** (`is_terminal=True`, `parent_event_id=None`): Episode
      emits exactly one of these in `finally`, regardless of how
      `agent.run` returned.
    - **Step-wise** (`is_terminal=False`, `parent_event_id=<ToolCallEvent.id>`):
      `MonitoredTool` emits one after each tool call when
      `task.validate_per_step=True`. Carries the per-step reward / info
      so step-eval data is preserved on disk without bleeding back to
      the agent (the agent only ever sees `obs` from `execute_action`).
    """

    reward: float
    info: dict = Field(default_factory=dict)
    is_terminal: bool = False
    parent_event_id: str | None = None


TrajectoryEventOutput = LLMCallEvent | ToolCallEvent | EvaluationEvent | AgentErrorEvent


class TrajectoryEvent(TypedBaseModel):
    output: TrajectoryEventOutput
    start_time: float | None = None
    end_time: float | None = None


class Trajectory(TypedBaseModel):
    """DEPRECATED — legacy trajectory shape; removal target: `agent-owns-loop-xray`.

    Legacy trajectory shape — what `Storage.load_trajectory(id)` returns.

    Consumed by XRay, the investigator, and `inspect_results` until they
    migrate to `TrajectoryView` directly (planned follow-up PR
    `agent-owns-loop-xray`). On the production write-path NOTHING constructs
    a `Trajectory` anymore — `Episode.run` builds an `TrajectoryMetadata`,
    streams events through `storage.save_event`, and returns an
    `TrajectoryView`. The Trajectory you see came from
    `_events_to_legacy_steps(view)` materializing the legacy step list at
    load time.

    Drops vs. the agent-owns-loop draft form:
      - `events` field removed (events live on the TrajectoryView; this class
        is steps-only for legacy consumers).
      - `streaming` flag removed (events ALWAYS stream now; the flag was
        a transition artefact).
      - `last_env_step` / `last_env_output` / `events_of_turn` /
        `n_agent_events` / `n_tool_calls` / `n_evaluations` methods removed
        (live on TrajectoryView; this class only carries what XRay needs).
      - `n_agent_steps` / `n_env_steps` properties stay because XRay's
        legacy step-walking UI counts them.
    """

    id: str
    steps: list[TrajectoryStep] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    start_time: float | None = None
    end_time: float | None = None
    reward_info: dict = Field(default_factory=dict)
    summary_stats: dict | None = None

    def last_env_step(self) -> EnvironmentOutput:
        """Most recent `EnvironmentOutput` in the steps list.

        Raises `ValueError` if the trajectory has no env step on disk.
        Used by the legacy XRay loader; new code should use
        `TrajectoryView.last_env_output()` (returns `None` instead of raising).
        """
        for step in reversed(self.steps):
            if isinstance(step.output, EnvironmentOutput):
                return step.output
        raise ValueError("No EnvironmentOutput found in the trajectory.")

    def last_env_output(self) -> EnvironmentOutput | None:
        """Like `last_env_step()` but returns `None` instead of raising."""
        try:
            return self.last_env_step()
        except ValueError:
            return None

    @property
    def n_agent_steps(self) -> int:
        """Number of agent turns in this trajectory's legacy step list."""
        return sum(1 for step in self.steps if isinstance(step.output, AgentOutput))

    @property
    def n_env_steps(self) -> int:
        """Number of env interactions in this trajectory's legacy step list."""
        return sum(1 for step in self.steps if isinstance(step.output, EnvironmentOutput))


class TrajectoryMetadata(BaseModel):
    """The scalar metadata of an episode — persisted as `episode.metadata.json`.

    Replaces the metadata half of the legacy `Trajectory` class (RFC
    `agent-owns-loop` scope expansion). The event list itself never lives
    here — events stream to `events/*.msgpack.zst` and are read back lazily
    via `TrajectoryView` (cube_harness.storage).

    Plain `BaseModel` (not `TypedBaseModel`) — TrajectoryMetadata is never
    polymorphic, and the `_type` discriminator that `TypedBaseModel`
    injects would shadow the legacy on-disk format that `load_trajectory`
    consumers still read.

    Written twice per episode:

    1. At episode START with `end_time=None` and stub summary fields.
       Makes crashed-mid-run episodes loadable: the file exists on disk
       and `TrajectoryView.is_complete` returns False until step 2 happens.
    2. At episode END with the final `end_time`, `summary_stats`, and
       `reward_info`. Overwrites the same file.
    """

    id: str
    metadata: dict = Field(default_factory=dict)
    start_time: float | None = None
    end_time: float | None = None
    summary_stats: dict | None = None
    reward_info: dict = Field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """True once `finalize_episode` has filled `end_time`."""
        return self.end_time is not None


class ActionSpace(frozenset[Callable]):
    """A set of action callables representing a subset of an action space.

    Supports set operations (&, -, |) for composing action subsets.
    """

    def __new__(cls, *actions: Callable) -> "ActionSpace":
        return super().__new__(cls, actions)

    @property
    def names(self) -> frozenset[str]:
        return frozenset(action.__name__ for action in self)

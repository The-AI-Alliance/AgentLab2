from typing import Callable
from uuid import uuid4

from cube.core import Action, EnvironmentOutput, StepError, TypedBaseModel
from pydantic import Field

from cube_harness.llm import LLMCall


def _new_event_id() -> str:
    """Allocate a fresh per-event id used as the AgentEvent.id and the
    `turn_id` of its child ToolCallEvent siblings."""
    return uuid4().hex


class AgentOutput(TypedBaseModel):
    actions: list[Action] = Field(default_factory=list)
    # All LLM calls made during this step. Set LLMCall.tag to label each call (e.g. "act", "summary").
    llm_calls: list[LLMCall] = Field(default_factory=list)
    error: StepError | None = None
    # Maps label → (start_time, end_time) as absolute Unix timestamps.
    # Used by the XRay viewer to render a profiling breakdown inside each timeline segment.
    profiling: dict[str, tuple[float, float]] = Field(default_factory=dict)
    # Agent's chain-of-thought, rationale, or extended thinking for this step.
    thoughts: str | None = None

    def __str__(self) -> str:
        return self.model_dump_json(exclude={"llm_calls"})


class TrajectoryStep(TypedBaseModel):
    output: EnvironmentOutput | AgentOutput
    start_time: float | None = None
    end_time: float | None = None


# --- New event-stream model (RFC: agent-owns-loop) -------------------------
# AgentEvent / ToolCallEvent / EvaluationEvent replace the binary
# `EnvironmentOutput | AgentOutput` union in trajectories. They live
# alongside the legacy `TrajectoryStep` during migration; Phase F (storage)
# converts on read and Phase E (Episode) flips the writer.


class AgentEvent(TypedBaseModel):
    """One agent turn — LLM call(s), thoughts, and the actions emitted.

    Each tool call spawned from this turn lives in a separate
    `ToolCallEvent` that references `AgentEvent.id` via
    `ToolCallEvent.parent_event_id` and one of `AgentEvent.actions[i].id`
    via `ToolCallEvent.action_id`.
    """

    id: str = Field(default_factory=_new_event_id)
    actions: list[Action] = Field(default_factory=list)
    llm_calls: list[LLMCall] = Field(default_factory=list)
    thoughts: str | None = None
    # Assistant response text alongside the tool calls (the prose the LLM
    # emitted on top of the structured actions). Lets XRay render what the
    # agent "said" even when several tool calls land in parallel.
    response_text: str | None = None
    # Maps label → (start_time, end_time) as absolute Unix timestamps.
    profiling: dict[str, tuple[float, float]] = Field(default_factory=dict)
    error: StepError | None = None

    @classmethod
    def from_agent_output(cls, output: AgentOutput, response_text: str | None = None) -> "AgentEvent":
        """Build from a legacy `AgentOutput`. Used by the default Agent.run
        and by the storage migration shim."""
        return cls(
            actions=list(output.actions),
            llm_calls=list(output.llm_calls),
            thoughts=output.thoughts,
            response_text=response_text,
            profiling=dict(output.profiling),
            error=output.error,
        )


class ToolCallEvent(TypedBaseModel):
    """One tool invocation — the action and its env response.

    `parent_event_id` references the originating `AgentEvent.id`.
    `action_id` references one of that agent event's `actions[i].id`.
    `turn_id` groups parallel siblings of a single agent turn (it equals
    the parent `AgentEvent.id` by default — agents emitting N parallel
    tool calls in one turn share that `turn_id`).
    """

    parent_event_id: str
    action_id: str | None = None  # echoes Action.id; nullable for legacy actions
    output: EnvironmentOutput
    turn_id: str


class EvaluationEvent(TypedBaseModel):
    """Terminal `task.evaluate()` result.

    Episode emits exactly one of these in `finally`, regardless of how
    `agent.run` returned.
    """

    reward: float
    info: dict = Field(default_factory=dict)


TrajectoryEventOutput = AgentEvent | ToolCallEvent | EvaluationEvent


class TrajectoryEvent(TypedBaseModel):
    output: TrajectoryEventOutput
    start_time: float | None = None
    end_time: float | None = None


class Trajectory(TypedBaseModel):
    """
    Stores history of the previous interaction.

    Metadata contains info about agent, env and task.
    reward_info represents episode level reward data.

    During the agent-owns-loop migration (RFC `agent-owns-loop`), this
    type carries BOTH the legacy `steps` field (binary union of
    `EnvironmentOutput | AgentOutput`, written by today's Episode loop)
    AND the new `events` field (the AgentEvent / ToolCallEvent /
    EvaluationEvent stream that the refactored Episode will write).
    Phase F (storage) lands the migration shim that converts old `steps`
    to `events` on read. Phase E flips Episode to write `events`. Tools
    that consume the trajectory should prefer `events` when non-empty.
    """

    id: str
    steps: list[TrajectoryStep] = Field(default_factory=list)
    events: list[TrajectoryEvent] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    start_time: float | None = None
    end_time: float | None = None
    reward_info: dict = Field(default_factory=dict)
    summary_stats: dict | None = None
    # When True, MonitoredTool / TurnRecorder skip the in-memory
    # `events.append(...)` step — events stream to storage + summary
    # ONLY. Preserves the stream-trajectory-steps invariant: driver /
    # worker RAM stays flat regardless of trajectory size (the
    # OSWorld-style 20 GB-per-job fix). Unit tests build trajectories
    # with streaming=False (the default) so they can inspect events
    # in-memory; Episode flips it to True for production runs.
    streaming: bool = False

    def last_env_step(self) -> EnvironmentOutput:
        """Return the most recent EnvironmentOutput in the trajectory.

        Prefers the new event stream (ToolCallEvent.output) over the
        legacy steps view. Raises ValueError if neither stream has any
        env output. With `streaming=True` the in-memory streams are
        empty during a run — call this on a loaded trajectory, not a
        live one."""
        # Prefer events stream when present.
        for event in reversed(self.events):
            if isinstance(event.output, ToolCallEvent):
                return event.output.output
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

    def events_of_turn(self, turn_id: str) -> list[TrajectoryEvent]:
        """All `ToolCallEvent`s sharing a `turn_id`. Returns `[]` if none.

        Useful for XRay to render parallel tool calls of one agent turn
        as siblings.
        """
        return [e for e in self.events if isinstance(e.output, ToolCallEvent) and e.output.turn_id == turn_id]

    @property
    def n_agent_steps(self) -> int:
        """Number of agent turns in this trajectory (legacy alias)."""
        # When the events stream is populated, it is authoritative —
        # `steps` may be a synthesized legacy view (see storage
        # _events_to_legacy_steps) so adding both would double-count.
        if self.events:
            return self.n_agent_events
        return sum(1 for step in self.steps if isinstance(step.output, AgentOutput))

    @property
    def n_env_steps(self) -> int:
        """Number of env interactions in this trajectory (legacy alias)."""
        if self.events:
            return self.n_tool_calls
        return sum(1 for step in self.steps if isinstance(step.output, EnvironmentOutput))

    @property
    def n_agent_events(self) -> int:
        """Number of AgentEvent entries in the event stream."""
        return sum(1 for e in self.events if isinstance(e.output, AgentEvent))

    @property
    def n_tool_calls(self) -> int:
        """Number of ToolCallEvent entries in the event stream."""
        return sum(1 for e in self.events if isinstance(e.output, ToolCallEvent))

    @property
    def n_evaluations(self) -> int:
        """Number of EvaluationEvent entries (≤1 per trajectory in practice)."""
        return sum(1 for e in self.events if isinstance(e.output, EvaluationEvent))


class ActionSpace(frozenset[Callable]):
    """A set of action callables representing a subset of an action space.

    Supports set operations (&, -, |) for composing action subsets.
    """

    def __new__(cls, *actions: Callable) -> "ActionSpace":
        return super().__new__(cls, actions)

    @property
    def names(self) -> frozenset[str]:
        return frozenset(action.__name__ for action in self)

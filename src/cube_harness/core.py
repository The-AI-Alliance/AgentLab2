from typing import Callable
from uuid import uuid4

from cube.core import Action, EnvironmentOutput, StepError, TypedBaseModel
from pydantic import BaseModel, Field

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
    """Legacy trajectory shape — what `Storage.load_trajectory(id)` returns.

    Consumed by XRay, the investigator, and `inspect_results` until they
    migrate to `EpisodeView` directly (planned follow-up PR
    `agent-owns-loop-xray`). On the production write-path NOTHING constructs
    a `Trajectory` anymore — `Episode.run` builds an `EpisodeMetadata`,
    streams events through `storage.save_event`, and returns an
    `EpisodeView`. The Trajectory you see came from
    `_events_to_legacy_steps(view)` materializing the legacy step list at
    load time.

    Drops vs. the agent-owns-loop draft form:
      - `events` field removed (events live on the EpisodeView; this class
        is steps-only for legacy consumers).
      - `streaming` flag removed (events ALWAYS stream now; the flag was
        a transition artefact).
      - `last_env_step` / `last_env_output` / `events_of_turn` /
        `n_agent_events` / `n_tool_calls` / `n_evaluations` methods removed
        (live on EpisodeView; this class only carries what XRay needs).
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
        `EpisodeView.last_env_output()` (returns `None` instead of raising).
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


class EpisodeMetadata(BaseModel):
    """The scalar metadata of an episode — persisted as `episode.metadata.json`.

    Replaces the metadata half of the legacy `Trajectory` class (RFC
    `agent-owns-loop` scope expansion). The event list itself never lives
    here — events stream to `events/*.msgpack.zst` and are read back lazily
    via `EpisodeView` (cube_harness.storage).

    Plain `BaseModel` (not `TypedBaseModel`) — EpisodeMetadata is never
    polymorphic, and the `_type` discriminator that `TypedBaseModel`
    injects would shadow the legacy on-disk format that `load_trajectory`
    consumers still read.

    Written twice per episode:

    1. At episode START with `end_time=None` and stub summary fields.
       Makes crashed-mid-run episodes loadable: the file exists on disk
       and `EpisodeView.is_complete` returns False until step 2 happens.
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

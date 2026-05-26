"""Agent abstraction."""

from abc import ABC, abstractmethod

from cube.core import ActionSchema, Observation, ValidatedConfig
from pydantic import Field

from cube_harness.core import AgentOutput, Trajectory


def apply_description_overrides(encoded_tools: list[dict], overrides: dict[str, str]) -> None:
    """Replace tool-schema descriptions in place from ``{action_name: description}``.

    ``encoded_tools`` are dicts in LLM tool-schema form (``ActionSchema.as_dict()``).
    Raises ``ValueError`` on a key that matches no tool in ``encoded_tools`` — this
    catches typos and keys left stale after an action was renamed. No-op when empty.
    """
    if not overrides:
        return
    by_name = {t["function"]["name"]: t for t in encoded_tools}
    unknown = set(overrides) - set(by_name)
    if unknown:
        raise ValueError(
            f"description_overrides target unknown actions {sorted(unknown)}; the action space has {sorted(by_name)}"
        )
    for name, description in overrides.items():
        by_name[name]["function"]["description"] = description


class AgentConfig(ValidatedConfig, ABC):
    """Configuration for creating an Agent."""

    description_overrides: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Experiment-time overrides for action descriptions, keyed by action name "
            "(the unique name the LLM sees). Replaces the docstring-derived description "
            "before the tool schema is built — a toggleable knob for testing better "
            "wording without editing the tool. A proven override graduates into the "
            "tool's docstring at the source via a PR."
        ),
    )

    @property
    def agent_name(self) -> str:
        """Human-readable name for this agent configuration, used in xray and logging."""
        return type(self).__name__

    @abstractmethod
    def make(self, action_set: list[ActionSchema] | None = None, **kwargs) -> "Agent":
        pass


class Agent(ABC):
    name: str
    description: str
    input_content_types: list[str]
    output_content_types: list[str]

    def __init__(self, config: AgentConfig):
        self.config = config

    @abstractmethod
    def step(self, obs: Observation) -> AgentOutput:
        """
        Perform a step given an observation and return the agent's output with actions.
        """
        pass

    def reflect(self, trajectory: Trajectory, final_reward: float) -> AgentOutput | None:
        """Hook called between episodes within a Rollout. Default: no-op.

        Implementations may inspect the just-finished trajectory and final reward,
        then update internal state read by subsequent ``step()`` calls (e.g. append
        a reflection to a memory list spliced into future prompts).

        When an implementation runs an LLM call as part of reflecting, it SHOULD
        return an ``AgentOutput`` carrying the reflection's ``LLMCall`` (with
        ``tag="reflection"``) and ``thoughts``. The Rollout then appends that
        output as a synthetic trajectory step so the reflection is observable in
        XRay, billing, and training-data extraction. Returning ``None`` means
        "nothing to record" and the trajectory is unchanged.

        Single-episode runs (``Episode.run``) never invoke this method; existing
        agents that don't override it behave identically inside or outside a Rollout.
        """
        _ = trajectory, final_reward
        return None

    def __repr__(self) -> str:
        return self.config.model_dump_json(indent=2, serialize_as_any=True)

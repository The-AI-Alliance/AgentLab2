"""Agent abstraction."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from cube.core import ActionSchema, Observation, ValidatedConfig
from pydantic import Field

from cube_harness.core import AgentOutput

if TYPE_CHECKING:
    from cube.task import Task

    from cube_harness.recorder import TurnRecorder

logger = logging.getLogger(__name__)


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
        """Instantiate the live Agent from this config + the task's action_set.

        Called by Episode after task.reset; subclasses wire in the
        action schemas, model handle, and any per-task overrides."""


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

    async def run(
        self,
        initial_obs: Observation,
        task: "Task",
        recorder: "TurnRecorder",
    ) -> None:
        """Default gym-style loop on top of `self.step` — the canonical
        entry point invoked by `Episode` (RFC `agent-owns-loop`).

        Sync `step()` agents get this for free. Agents that want
        parallel tool calls, async LLM dispatch, or streaming
        observability override this method instead.

        Termination:

          * Natural: `env_output.done == True` from `task.step`, or
            `agent.step` returns empty actions with no error.
          * Hard: `BudgetExceeded` raised by a monitored tool propagates
            up to `Episode`, which captures it in `finally`.

        The agent does NOT call `task.reset` or `task.evaluate` — those
        are Episode's. `task.step` is wrapped in `asyncio.to_thread`
        because cube-standard's `Task.step` is sync today.
        """
        obs = initial_obs
        while True:
            agent_output = await asyncio.to_thread(self.step, obs)
            recorder.record(agent_output)
            if not agent_output.actions and agent_output.error is None:
                # Graceful "done" by the agent itself (the convention
                # today's Episode._run_loop honours).
                return
            env_output = await asyncio.to_thread(task.step, agent_output.actions)
            if env_output.done:
                return
            obs = env_output.obs

    def __repr__(self) -> str:
        return self.config.model_dump_json(indent=2, serialize_as_any=True)

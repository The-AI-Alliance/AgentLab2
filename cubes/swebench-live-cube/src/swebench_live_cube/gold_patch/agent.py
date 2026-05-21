"""Gold-patch oracle agent for SWE-bench Live.

Applies the gold patch written by SWEBenchLiveTask.reset() (oracle_mode=True)
then immediately calls final_step. Used to validate the eval pipeline and
identify which tasks are solvable before running any real agent.

Requires cube-harness (not a swebench-live-cube runtime dependency).
Install with: pip install swebench-live-cube[eval]
"""

from cube.core import Action, ActionSchema, Observation

from cube_harness.agent import Agent, AgentConfig
from cube_harness.core import AgentOutput

_APPLY = Action(
    name="bash",
    arguments={
        "command": (
            "git apply /tmp/gold_patch.diff 2>&1"
            " || git apply --reject /tmp/gold_patch.diff 2>&1"
            " || patch --batch --forward --fuzz=5 -p1 -i /tmp/gold_patch.diff 2>&1"
        )
    },
)
_STOP = Action(name="final_step", arguments={})


class GoldPatchAgent(Agent):
    """Deterministic oracle: apply the gold patch then stop. No LLM."""

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)
        self._turn = 0

    def step(self, obs: Observation) -> AgentOutput:
        action = _APPLY if self._turn == 0 else _STOP
        self._turn += 1
        return AgentOutput(actions=[action])


class GoldPatchAgentConfig(AgentConfig):
    """Config for GoldPatchAgent — no parameters, no LLM."""

    def make(self, action_set: list[ActionSchema] | None = None, **kwargs: object) -> GoldPatchAgent:
        return GoldPatchAgent(config=self)

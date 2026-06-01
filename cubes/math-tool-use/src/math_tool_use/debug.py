"""Deterministic debug agent for testing MathToolUseBenchmark end-to-end without an LLM.

Public API
----------
get_debug_benchmark()         -> MathToolUseBenchmark
make_debug_agent(task_id)     -> DebugAgent
"""

from __future__ import annotations

import logging
import sys

from cube.benchmark import Benchmark
from cube.core import Action, ActionSchema, Observation
from cube.task import TaskConfig
from cube.testing import run_debug_suite
from math_tool_use.benchmark import MathToolUseBenchmark

logger = logging.getLogger(__name__)

_TASK_ACTIONS: dict[str, list[Action]] = {
    "add-3-4": [
        Action(name="run_python_code", arguments={"code": "result = 3 + 4"}),
        Action(name="MathAnswer", arguments={"answer": "\\boxed{7}"}),
    ],
    "sub-10-3": [
        Action(name="run_python_code", arguments={"code": "result = 10 - 3"}),
        Action(name="MathAnswer", arguments={"answer": "\\boxed{7}"}),
    ],
    "mul-6-7": [
        Action(name="run_python_code", arguments={"code": "result = 6 * 7"}),
        Action(name="MathAnswer", arguments={"answer": "\\boxed{42}"}),
    ],
    "add-100-1": [
        Action(name="run_python_code", arguments={"code": "result = 100 + 1"}),
        Action(name="MathAnswer", arguments={"answer": "\\boxed{101}"}),
    ],
}


class DebugAgent:
    """Deterministic agent that replays a fixed action sequence."""

    def __init__(self, task_id: str) -> None:
        if task_id not in _TASK_ACTIONS:
            raise ValueError(f"No debug actions for {task_id!r}. Known: {list(_TASK_ACTIONS)}")
        self._task_id = task_id
        self._step = 0
        self._actions = list(_TASK_ACTIONS[task_id])

    def get_action(self, obs: Observation) -> Action:
        if self._step >= len(self._actions):
            raise StopIteration(f"All actions exhausted for task {self._task_id!r}")
        action = self._actions[self._step]
        self._step += 1
        return action

    def __call__(self, obs: Observation, action_set: list[ActionSchema]) -> Action:
        return self.get_action(obs)


def get_debug_benchmark() -> Benchmark:
    return MathToolUseBenchmark().subset_from_list(list(_TASK_ACTIONS.keys()))


def get_debug_task_configs() -> list[TaskConfig]:
    return list(get_debug_benchmark().get_task_configs())


def make_debug_agent(task_id: str) -> DebugAgent:
    return DebugAgent(task_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

    module = sys.modules.get("math_tool_use.debug", sys.modules[__name__])
    results = run_debug_suite("math-tool-use", module)
    failed = [r for r in results if r["error"] or not r["done"] or r["reward"] < 1.0]
    sys.exit(1 if failed else 0)

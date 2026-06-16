"""Deterministic debug agent for testing swegym-cube end-to-end without an LLM.

Public API
----------
get_debug_benchmark()         -> SWEGymBenchmarkConfig
make_debug_agent(task_id)     -> DebugAgent
"""

from __future__ import annotations

import logging

from cube.core import Action, ActionSchema, Observation

from swegym_cube.benchmark import SWEGymBenchmarkConfig

logger = logging.getLogger(__name__)

# Each debug task runs in oracle_mode: the gold patch is written to
# /tmp/gold_patch.diff during reset(). The debug agent applies it
# and calls final_step, which triggers evaluate() → tests → reward == 1.0.
_APPLY_PATCH = Action(name="bash", arguments={"command": "git apply /tmp/gold_patch.diff 2>&1"})
_FINAL = Action(name="final_step", arguments={})

# Two small, fast tasks from different repos (moto + dvc), each with a single
# FAIL_TO_PASS test and a tiny gold patch — chosen so the oracle debug run is quick.
_TASK_ACTIONS: dict[str, list[Action]] = {
    "getmoto__moto-5699": [_APPLY_PATCH, _FINAL],
    "iterative__dvc-5822": [_APPLY_PATCH, _FINAL],
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


def get_debug_benchmark() -> SWEGymBenchmarkConfig:
    """Return a ``SWEGymBenchmarkConfig`` scoped to the debug tasks.

    Pure factory — the harness owns ``config.install()`` and ``config.make(infra)``.
    Debug tasks run in ``oracle_mode`` so reset() writes the gold patch to
    /tmp/gold_patch.diff.
    """
    return SWEGymBenchmarkConfig(oracle_mode=True).subset_from_list(list(_TASK_ACTIONS))


def make_debug_agent(task_id: str) -> DebugAgent:
    return DebugAgent(task_id)


if __name__ == "__main__":
    import sys

    import swegym_cube.debug as _this_module
    from cube.testing import run_debug_suite

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

    results = run_debug_suite("swegym-cube", _this_module)
    failed = [r for r in results if r["error"] or not r["done"] or r["reward"] < 1.0]
    sys.exit(1 if failed else 0)

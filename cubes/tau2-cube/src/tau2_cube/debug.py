"""Deterministic debug agent for testing CubeBenchmarkConfig end-to-end without an LLM.

Each debug task has a hardcoded action sequence that completes it successfully.
Used to validate the CUBE task loop in CI or local development.

Public API
----------
get_debug_benchmark()        → CubeBenchmarkConfig
make_debug_agent(task_id)    → DebugAgent

Usage::

    # Run all debug tasks and print a JSON report
    python -m tau2_cube.debug
"""

from __future__ import annotations

import logging

from cube.core import Action, ActionSchema, Observation
from cube.task import STOP_ACTION

from tau2_cube.benchmark import CubeBenchmarkConfig


logger = logging.getLogger(__name__)
_DEBUG_TASK_IDS = ["mock/create_task_1", "mock/update_task_1"]
# ---------------------------------------------------------------------------
# Hardcoded action sequences per task ID
#
# Each sequence must drive the task to done=True, reward=1.0.
# These also serve as a minimal sanity check that the task logic is correct.
# ---------------------------------------------------------------------------


def _gold_actions(task_id: str) -> list[Action]:
    """Pull the tau2 reference actions and convert to cube Actions."""
    meta = CubeBenchmarkConfig.task_metadata[task_id]
    return [
        Action(name=a.name, arguments=a.arguments)
        for a in meta.tau2_task.evaluation_criteria.actions
        if a.requestor == "assistant"  # only ones our agent can issue
    ]


# ---------------------------------------------------------------------------
# DebugAgent
# ---------------------------------------------------------------------------


class DebugAgent:
    """Deterministic debug agent that replays a fixed action sequence.

    Interface is compatible with cube.testing:
        agent = make_debug_agent(task_id)
        action = agent(obs, action_set)   # __call__ shorthand

    Args:
        task_id: Must match a key in _TASK_ACTIONS.

    Raises:
        ValueError: If task_id has no registered action sequence.
    """

    def __init__(self, task_id: str) -> None:
        # Replay the gold actions, then STOP so the episode ends with AGENT_STOP
        # (evaluate_simulation requires it; finished() no longer auto-stops).
        self._actions = [*_gold_actions(task_id), Action(name=STOP_ACTION.name, arguments={})]
        self._step = 0

    def __call__(self, obs: Observation, action_set: list[ActionSchema]) -> Action:
        action = self._actions[self._step]
        self._step += 1
        return action


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_debug_benchmark() -> CubeBenchmarkConfig:
    """Return a CubeBenchmarkConfig scoped to the debug tasks.

    Called once by cube.testing before any debug episodes run. The harness
    calls config.install(), config.make() to get a live Benchmark, then
    benchmark.close() at the end.

    If only a subset of tasks has debug sequences, use subset_from_list::

        return CubeBenchmarkConfig().subset_from_list(list(_TASK_ACTIONS))
    """
    return CubeBenchmarkConfig().subset_from_list(_DEBUG_TASK_IDS)


def make_debug_agent(task_id: str) -> DebugAgent:
    """Return a fresh DebugAgent for the given task_id."""
    return DebugAgent(task_id)


# ---------------------------------------------------------------------------
# __main__ — run all debug tasks, print JSON report
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    import tau2_cube.debug as _mod
    from cube.testing import run_debug_suite

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    results = run_debug_suite("tau2-cube", _mod)

    failed = [r for r in results if r["error"] or not r["done"] or r["reward"] <= 0]
    sys.exit(1 if failed else 0)

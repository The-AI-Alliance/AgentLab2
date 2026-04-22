"""Deterministic debug module for DRBench CUBE integration testing.

Provides ``get_debug_benchmark()`` and ``make_debug_agent()`` as required by
``cube.testing``. The debug agent replays a hardcoded action sequence per task
that exercises the full CUBE task loop without an LLM.

For DRBench the only structurally meaningful sequence is:
  1. One information-gathering action (exercises the tool → container path)
  2. submit_report() (triggers finished() → done=True)

Note on reward: DRBench's evaluate() calls an LLM judge (score_report), so
reward != 1.0 in CI. The debug tests validate structural compliance
(container starts, reset works, action_set is valid, done triggers) rather
than reward value.

Public API
----------
get_debug_benchmark()        → DrBenchBenchmark (scoped to sanity subset)
make_debug_agent(task_id)    → DebugAgent

Usage::

    python -m drbench_cube.debug
"""

from __future__ import annotations

import logging

from cube.core import Action, ActionSchema, Observation

from drbench_cube.benchmark import DrBenchBenchmark
from drbench_cube.container import DrBenchContainerBackend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded action sequences per task ID
#
# Each sequence exercises the tool→container path then calls submit_report()
# to drive the task to done=True. Reward will be low (no real report) — that
# is expected. The tests check structural compliance, not reward value.
# ---------------------------------------------------------------------------

_TASK_ACTIONS: dict[str, list[Action]] = {
    "DR0001": [
        # Gather something from Nextcloud to exercise the container path
        Action(name="list_nextcloud_directory", arguments={"path": "/"}),
        # Submit a minimal report — drives finished() → True → done=True
        Action(
            name="submit_report",
            arguments={
                "report_text": (
                    "Lee's Market can leverage FSMA 204 regulations by implementing "
                    "enhanced traceability systems for food safety compliance."
                )
            },
        ),
    ],
}


# ---------------------------------------------------------------------------
# DebugAgent
# ---------------------------------------------------------------------------


class DebugAgent:
    """Deterministic debug agent that replays a fixed action sequence."""

    def __init__(self, task_id: str) -> None:
        if task_id not in _TASK_ACTIONS:
            raise ValueError(
                f"No debug actions registered for task {task_id!r}. "
                f"Known tasks: {list(_TASK_ACTIONS)}"
            )
        self._task_id = task_id
        self._step = 0
        self._actions = list(_TASK_ACTIONS[task_id])

    def get_action(self, obs: Observation) -> Action:
        if self._step >= len(self._actions):
            raise StopIteration(
                f"[DebugAgent] task={self._task_id!r}: all {len(self._actions)} actions exhausted"
            )
        action = self._actions[self._step]
        logger.info(
            "[DebugAgent] task=%r  step=%d/%d  action=%s",
            self._task_id,
            self._step + 1,
            len(self._actions),
            action.name,
        )
        self._step += 1
        return action

    def __call__(self, obs: Observation, action_set: list[ActionSchema]) -> Action:
        return self.get_action(obs)


# ---------------------------------------------------------------------------
# Public helpers (required by cube.testing)
# ---------------------------------------------------------------------------


def get_debug_benchmark() -> DrBenchBenchmark:
    """Return a DrBenchBenchmark scoped to the debug tasks (DR0001 from val subset)."""
    debug_ids = list(_TASK_ACTIONS.keys())
    benchmark = DrBenchBenchmark(container_backend=DrBenchContainerBackend())
    return benchmark.subset_from_list(debug_ids)


def make_debug_agent(task_id: str) -> DebugAgent:
    """Return a fresh DebugAgent for the given task_id."""
    return DebugAgent(task_id)


# ---------------------------------------------------------------------------
# __main__ — run all debug tasks, print JSON report
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    import drbench_cube.debug as _mod
    from cube.testing import run_debug_suite

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    results = run_debug_suite("drbench-cube", _mod)
    failed = [r for r in results if r["error"] or not r["done"]]
    sys.exit(1 if failed else 0)

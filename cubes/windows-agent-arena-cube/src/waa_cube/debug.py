"""Deterministic debug agent for testing WAATask end-to-end with a live VM.

Public API (required by cube.testing.run_debug_suite)
-----------------------------------------------------
make_debug_agent(task_id)    → DebugAgent
get_debug_benchmark()        → DebugWAABenchmark

Usage::

    # Via cube CLI
    cube test waa-cube

    # Directly
    python -m waa_cube.debug
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from cube.benchmark import BenchmarkConfig
from cube.core import Action, ActionSchema, Observation
from cube.task import TaskConfig, TaskMetadata

from waa_cube.benchmark import WAABenchmark, WAATaskConfig
from waa_cube.computer import ComputerConfig
from waa_cube.task import WAATask

logger = logging.getLogger(__name__)

_DEBUG_TASK_METADATA_JSON = Path(__file__).parent / "debug_task_metadata.json"


# ---------------------------------------------------------------------------
# Debug benchmark and task config
# ---------------------------------------------------------------------------


class DebugWAATaskConfig(WAATaskConfig):
    def make(
        self,
        runtime_context: dict | None = None,
    ) -> WAATask:
        metadata = DebugWAABenchmark.task_metadata[self.task_id]
        return WAATask(
            metadata=metadata,
            tool_config=self.tool_config or ComputerConfig(),
            infra=self.infra,
            runtime_context=runtime_context,
        )


class DebugWAABenchmark(WAABenchmark):
    """WAABenchmark scoped to debug tasks only.

    Loads task_metadata from debug_task_metadata.json — config and evaluator are
    embedded in extra_info, so no WAA repo or evaluation_examples_windows/ is needed.
    """

    benchmark_metadata = WAABenchmark.benchmark_metadata.model_copy(update={"name": "waa-cube-debug", "num_tasks": 2})
    task_metadata: ClassVar[dict[str, TaskMetadata]] = BenchmarkConfig.task_metadata_from_json(
        _DEBUG_TASK_METADATA_JSON
    )
    task_config_class: ClassVar[type[TaskConfig]] = DebugWAATaskConfig

    def install(self) -> None:
        """No-op: debug task data is embedded in debug_task_metadata.json."""
        logger.info("DebugWAABenchmark.install() — nothing to do")


# ---------------------------------------------------------------------------
# Hardcoded action sequences per task ID
# ---------------------------------------------------------------------------

_TASK_ACTIONS: dict[str, list[Action]] = {
    "waa-debug-notepad": [
        # Win+R → type notepad → Enter → wait → done
        Action(name="hotkey", arguments={"keys": ["win", "r"]}),
        Action(name="wait", arguments={}),
        Action(name="typing", arguments={"text": "notepad"}),
        Action(name="press", arguments={"key": "enter"}),
        Action(name="wait", arguments={}),
        Action(name="done", arguments={}),
    ],
    "waa-debug-infeasible": [
        Action(name="fail", arguments={}),
    ],
}


# ---------------------------------------------------------------------------
# DebugAgent
# ---------------------------------------------------------------------------


class DebugAgent:
    """Deterministic debug agent that replays a fixed action sequence for a task."""

    def __init__(self, task_id: str) -> None:
        if task_id not in _TASK_ACTIONS:
            raise ValueError(f"No debug actions registered for task {task_id!r}. Known tasks: {list(_TASK_ACTIONS)}")
        self._task_id = task_id
        self._step = 0
        self._actions = list(_TASK_ACTIONS[task_id])

    def get_action(self, obs: Observation) -> Action:
        if self._step >= len(self._actions):
            raise StopIteration(f"[DebugAgent] task={self._task_id!r}: all {len(self._actions)} actions exhausted")
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
# Public helpers (required by cube.testing.run_debug_suite)
# ---------------------------------------------------------------------------


def get_debug_benchmark() -> DebugWAABenchmark:
    """Return a WAABenchmark scoped to the debug tasks."""
    return DebugWAABenchmark(
        default_tool_config=ComputerConfig(),
    )


def make_debug_agent(task_id: str) -> DebugAgent:
    return DebugAgent(task_id)


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    from cube.testing import run_debug_suite

    import waa_cube.debug as _mod

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    results = run_debug_suite("waa-cube", _mod)
    failed = [r for r in results if r["error"] or not r["done"] or r["reward"] <= 0]
    sys.exit(1 if failed else 0)

"""Deterministic debug agent for testing SokobanBenchmarkConfig without an LLM.

Each level ships with a reference ``solution`` (a move sequence) on its
metadata, so the debug action sequences are derived directly from the levels —
no hand-authoring. Replaying a level's solution drives it to ``done=True,
reward=1.0``, which is exactly what ``cube test sokoban-cube`` and the pytest
suite assert.

Actions are resolved lazily per task (not at import time) so that importing this
module never depends on ``task_metadata.json`` being well-formed.

Public API
----------
get_debug_benchmark()        → SokobanBenchmarkConfig
make_debug_agent(task_id)    → DebugAgent
"""

from __future__ import annotations

import logging

from cube.core import Action, ActionSchema, Observation

from sokoban_cube.benchmark import SokobanBenchmarkConfig
from sokoban_cube.task import SokobanTaskMetadata

logger = logging.getLogger(__name__)


def _actions_for(task_id: str) -> list[Action]:
    """Replayable action for ``task_id``: its whole stored solution as one multi-move call."""
    meta = SokobanBenchmarkConfig.task_metadata.get(task_id)
    if meta is None:
        raise ValueError(f"Unknown task {task_id!r}. Known tasks: {list(SokobanBenchmarkConfig.task_metadata)}")
    assert isinstance(meta, SokobanTaskMetadata), f"task {task_id!r} metadata is not a SokobanTaskMetadata"
    return [Action(name="move", arguments={"directions": ",".join(meta.solution)})]


class DebugAgent:
    """Deterministic debug agent that replays a level's reference solution."""

    def __init__(self, task_id: str) -> None:
        self._task_id = task_id
        self._step = 0
        self._actions = _actions_for(task_id)

    def get_action(self, obs: Observation) -> Action:
        if self._step >= len(self._actions):
            raise StopIteration(f"[DebugAgent] task={self._task_id!r}: all {len(self._actions)} actions exhausted")
        action = self._actions[self._step]
        self._step += 1
        return action

    def __call__(self, obs: Observation, action_set: list[ActionSchema]) -> Action:
        return self.get_action(obs)


def get_debug_benchmark() -> SokobanBenchmarkConfig:
    """Full config — every shipped level has a reference solution to replay."""
    return SokobanBenchmarkConfig()


def make_debug_agent(task_id: str) -> DebugAgent:
    return DebugAgent(task_id)


if __name__ == "__main__":
    import sys

    import sokoban_cube.debug as _mod
    from cube.testing import run_debug_suite

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s", datefmt="%H:%M:%S"
    )

    results = run_debug_suite("sokoban-cube", _mod)
    failed = [r for r in results if r["error"] or not r["done"] or r["reward"] < 1.0]
    sys.exit(1 if failed else 0)

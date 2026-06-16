"""
Deterministic debug agent for testing OSWorldTask end-to-end without an LLM.

Each debug task in debug_tasks.json has a hardcoded action sequence that
completes it successfully. Used to validate the CUBE task loop in CI or
local development without requiring an LLM.

Public API
----------
make_debug_agent(task_id)    → DebugAgent
get_debug_benchmark()        → OSWorldBenchmarkConfig

Usage:
    # Run all debug tasks and print a JSON report
    python -m osworld_cube.debug
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Generator
from pathlib import Path
from typing import ClassVar

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.core import Action, ActionSchema, Observation
from cube.resource import InfraConfig
from cube.task import RuntimeContext, TaskConfig, TaskMetadata
from cube.testing import run_debug_suite
from cube import LocalInfraConfig

from osworld_cube.benchmark import OSWorldBenchmark, OSWorldBenchmarkConfig, OSWorldTaskConfig
from osworld_cube.computer import ComputerConfig
from osworld_cube.infra_loader import load_runtime_infra_from_config_file
from osworld_cube.task import (
    OSWorldExecutionInfo,
    OSWorldTask,
)


logger = logging.getLogger(__name__)

_DEBUG_TASK_METADATA_JSON = Path(__file__).parent / "debug_task_metadata.json"


# ---------------------------------------------------------------------------
# Embedded execution info — debug tasks bypass the per-task cache because
# they don't require the OSWorld repo clone.
# ---------------------------------------------------------------------------

_DEBUG_EXECUTION_INFO: dict[str, OSWorldExecutionInfo] = {
    "simple-create-file": OSWorldExecutionInfo(
        config=[],
        evaluator={
            "func": "check_include_exclude",
            "result": {"type": "vm_command_line", "command": "cat ~/Desktop/hello.txt"},
            "expected": {"type": "rule", "rules": {"include": ["Hello World"], "exclude": []}},
        },
    ),
    "simple-make-directory": OSWorldExecutionInfo(
        config=[],
        evaluator={
            "func": "check_include_exclude",
            "result": {"type": "vm_command_line", "command": "ls ~/Desktop/"},
            "expected": {"type": "rule", "rules": {"include": ["my_folder"], "exclude": []}},
        },
    ),
}


# ---------------------------------------------------------------------------
# DebugOSWorldTaskConfig — bypasses the per-task cache, embeds execution_info
# ---------------------------------------------------------------------------


class DebugOSWorldTaskConfig(OSWorldTaskConfig):
    """OSWorldTaskConfig variant for debug tasks.

    Uses the embedded ``_DEBUG_EXECUTION_INFO`` mapping instead of loading
    from the per-task execution cache, so the OSWorld repo clone is not
    required to run debug tasks.
    """

    def verify_installed(self) -> None:
        """No-op: debug execution data is embedded in this module."""

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
    ) -> OSWorldTask:
        execution_info = _DEBUG_EXECUTION_INFO.get(self.task_id)
        if execution_info is None:
            raise RuntimeError(
                f"No debug execution info for task {self.task_id!r}. Known debug tasks: {sorted(_DEBUG_EXECUTION_INFO)}"
            )
        return OSWorldTask(
            metadata=self.metadata,
            execution_info=execution_info,
            tool_config=self.tool_config or ComputerConfig(),
            runtime_context=runtime_context,
            use_som=self.use_som,
        )


# ---------------------------------------------------------------------------
# DebugOSWorldBenchmarkConfig
# ---------------------------------------------------------------------------


class DebugOSWorldBenchmarkConfig(OSWorldBenchmarkConfig):
    """OSWorldBenchmarkConfig variant for the two hardcoded debug tasks.

    This is a separate subclass rather than ``OSWorldBenchmarkConfig().subset_from_list(...)``
    because debug differs from production along three axes that subsetting cannot express:

    1. **Different ``task_metadata`` source.** The debug task IDs
       (``simple-create-file``, ``simple-make-directory``) are not present in
       the main ``task_metadata.json``; they live in ``debug_task_metadata.json``.
       Subsetting can only narrow to IDs that already exist in the parent
       registry.

    2. **Different ``task_config_class`` (``DebugOSWorldTaskConfig``).**
       ``OSWorldTaskConfig.make()`` calls ``verify_installed()`` and reads the
       per-task execution cache from disk, so it requires the OSWorld repo
       clone and a populated ``~/.cube/.../tasks_execution_info/`` directory.
       ``DebugOSWorldTaskConfig.make()`` reads ``_DEBUG_EXECUTION_INFO``
       embedded in this module instead — no repo, no cache.

    3. **No-op ``install()`` / ``uninstall()``.** Base ``install()`` clones
       the OSWorld repo and writes per-task cache files; debug needs neither
       because all execution data is embedded in this module.

    ``resources = []`` and a tweaked ``benchmark_metadata`` are minor add-ons
    on top of those three structural differences.
    """

    benchmark_metadata: ClassVar[BenchmarkMetadata] = OSWorldBenchmarkConfig.benchmark_metadata.model_copy(
        update={"name": "osworld-cube-debug", "num_tasks": 2, "named_subsets": {}}
    )
    task_metadata: ClassVar[dict[str, TaskMetadata]] = BenchmarkConfig.task_metadata_from_json(
        _DEBUG_TASK_METADATA_JSON
    )
    task_config_class: ClassVar[type[TaskConfig]] = DebugOSWorldTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = OSWorldBenchmark

    @classmethod
    def install(cls) -> None:
        """No-op: debug task execution data is embedded in this module."""
        logger.info("DebugOSWorldBenchmarkConfig.install() — nothing to do")

    @classmethod
    def uninstall(cls) -> None:
        """No-op: debug task execution data is embedded in this module."""
        logger.info("DebugOSWorldBenchmarkConfig.uninstall() — nothing to do")

    def make(self, infra: InfraConfig | None = None) -> OSWorldBenchmark:
        """Resolve infra from OSWORLD_CUBE_TEST_INFRA_CONFIG_FILE if not provided."""
        return super().make(infra or _get_default_infra())

    def get_task_configs(self) -> Generator[OSWorldTaskConfig, None, None]:
        """Yield DebugOSWorldTaskConfig objects."""
        for tm in self.tasks().values():
            yield DebugOSWorldTaskConfig(
                metadata=tm,
                tool_config=self.tool_config,
                use_som=self.use_som,
            )


# ---------------------------------------------------------------------------
# Hardcoded action sequences per task ID
# ---------------------------------------------------------------------------

_TASK_ACTIONS: dict[str, list[Action]] = {
    "simple-create-file": [
        Action(name="hotkey", arguments={"keys": ["ctrl", "alt", "t"]}),
        Action(name="wait", arguments={}),
        Action(name="typing", arguments={"text": "echo 'Hello World' > ~/Desktop/hello.txt"}),
        Action(name="press", arguments={"key": "enter"}),
        Action(name="wait", arguments={}),
        Action(name="done", arguments={}),
    ],
    "simple-make-directory": [
        Action(name="hotkey", arguments={"keys": ["ctrl", "alt", "t"]}),
        Action(name="wait", arguments={}),
        Action(name="typing", arguments={"text": "mkdir ~/Desktop/my_folder"}),
        Action(name="press", arguments={"key": "enter"}),
        Action(name="wait", arguments={}),
        Action(name="done", arguments={}),
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
        logger.debug(
            "[DebugAgent] Initialised for task=%r with %d actions",
            task_id,
            len(self._actions),
        )

    def get_action(self, obs: Observation) -> Action:
        if self._step >= len(self._actions):
            raise StopIteration(f"[DebugAgent] task={self._task_id!r}: all {len(self._actions)} actions exhausted")
        action = self._actions[self._step]
        logger.debug(
            "[DebugAgent] task=%r  step=%d/%d  action=%s  args=%s",
            self._task_id,
            self._step + 1,
            len(self._actions),
            action.name,
            action.arguments or "",
        )
        self._step += 1
        return action

    def __call__(self, obs: Observation, action_set: list[ActionSchema]) -> Action:
        return self.get_action(obs)


def _get_default_infra() -> InfraConfig:
    """Resolve the default infra for debug runs.

    Priority:
      1. ``OSWORLD_CUBE_TEST_INFRA_CONFIG_FILE=/path/to/infra.json``
      2. ``LocalInfraConfig()`` for the standard zero-arg ``cube test`` path
    """
    return load_runtime_infra_from_config_file() or LocalInfraConfig()


def get_debug_benchmark() -> DebugOSWorldBenchmarkConfig:
    """Return a ``DebugOSWorldBenchmarkConfig`` for the run_debug_suite path.

    Uses ``debug_task_metadata.json`` as the task source — no OSWorld repo
    clone or per-task cache required. ``infra`` is wired in by the harness
    via ``config.make(infra)``.
    """
    return DebugOSWorldBenchmarkConfig()


def make_debug_agent(task_id: str) -> DebugAgent:
    return DebugAgent(task_id)


if __name__ == "__main__":
    import osworld_cube.debug as _mod

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    results = run_debug_suite("osworld-cube", _mod, infra=_get_default_infra())
    failed = [r for r in results if r["error"] or not r["done"] or r["reward"] <= 0]
    sys.exit(1 if failed else 0)

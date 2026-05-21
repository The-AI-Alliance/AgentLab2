"""Deterministic debug agent for testing BrowseCompBenchmark end-to-end without an LLM.

Public API
----------
get_debug_benchmark()         → DebugBrowseCompBenchmark
make_debug_agent(task_id)     → DebugAgent
"""

from __future__ import annotations

import logging
from typing import ClassVar, Generator

from cube.benchmark import Benchmark, BenchmarkMetadata, RuntimeContext
from cube.core import Action, ActionSchema, Observation
from cube.resource import InfraConfig
from cube.task import TaskConfig, TaskMetadata
from cube.tool import ToolboxConfig

from browsercomp_cube.benchmark import BrowseCompBenchmark, BrowseCompBenchmarkConfig
from browsercomp_cube.task import (
    BrowseCompExecutionInfo,
    BrowseCompTask,
    BrowseCompTaskConfig,
    BrowseCompTaskMetadata,
)
from browsercomp_cube.tool import SubmitAnswerToolConfig

logger = logging.getLogger(__name__)

_DEBUG_RECORDS: list[dict[str, str]] = [
    {"problem": "What is 2 + 2?", "answer": "4", "topic": "debug"},
    {"problem": "What is the capital of France?", "answer": "Paris", "topic": "debug"},
]


def _debug_task_id(idx: int) -> str:
    return f"browsecomp-debug-{idx:04d}"


def _debug_task_metadata() -> dict[str, TaskMetadata]:
    return {
        _debug_task_id(i): BrowseCompTaskMetadata(
            id=_debug_task_id(i),
            recommended_max_steps=5,
            topic=record["topic"],
        )
        for i, record in enumerate(_DEBUG_RECORDS)
    }


_TASK_ACTIONS: dict[str, list[Action]] = {
    "browsecomp-debug-0000": [
        Action(
            name="submit_answer",
            arguments={"answer": "Explanation: debug\nExact Answer: 4\nConfidence: 100"},
        )
    ],
    "browsecomp-debug-0001": [
        Action(
            name="submit_answer",
            arguments={"answer": "Explanation: debug\nExact Answer: Paris\nConfidence: 100"},
        )
    ],
}


class DebugBrowseCompTask(BrowseCompTask):
    """BrowseCompTask with a deterministic grader — no LLM calls."""

    def _call_grader(self, prompt: str, scorer_model: str) -> tuple[bool, str]:
        submitted = self._submit_tool().last_answer or ""
        is_correct = self._exec.answer.lower() in submitted.lower()
        return is_correct, f"correct: {'yes' if is_correct else 'no'} (debug grader)"


class DebugBrowseCompTaskConfig(BrowseCompTaskConfig):
    """TaskConfig that produces DebugBrowseCompTask instances."""

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
    ) -> DebugBrowseCompTask:
        idx = int(self.metadata.id.rsplit("-", 1)[-1])
        record = _DEBUG_RECORDS[idx]

        tool_cfg = self.tool_config or ToolboxConfig(tool_configs=[SubmitAnswerToolConfig()])
        return DebugBrowseCompTask(
            metadata=self.metadata,
            execution_info=BrowseCompExecutionInfo(problem=record["problem"], answer=record["answer"]),
            tool_config=tool_cfg,
            scorer_model=self.scorer_model,
            runtime_context=runtime_context,
        )


class DebugBrowseCompBenchmark(BrowseCompBenchmark):
    """Lightweight debug benchmark — 2 tasks, no network calls."""


class DebugBrowseCompBenchmarkConfig(BrowseCompBenchmarkConfig):
    """Serializable debug benchmark configuration."""

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BrowseCompBenchmarkConfig.benchmark_metadata
    task_metadata = _debug_task_metadata()
    task_config_class: ClassVar[type[TaskConfig]] = DebugBrowseCompTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = DebugBrowseCompBenchmark

    scorer_model: str = "debug-grader-unused"

    @classmethod
    def install(cls) -> None:
        """Debug benchmark uses inline records — no installation needed."""
        return

    @classmethod
    def uninstall(cls) -> None:
        return

    def get_task_configs(self) -> Generator[DebugBrowseCompTaskConfig, None, None]:
        for tm in self.tasks().values():
            yield DebugBrowseCompTaskConfig(
                metadata=tm,
                tool_config=ToolboxConfig(tool_configs=[SubmitAnswerToolConfig()]),
                scorer_model=self.scorer_model,
            )


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


def get_debug_benchmark(infra: InfraConfig | None = None) -> DebugBrowseCompBenchmarkConfig:
    return DebugBrowseCompBenchmarkConfig()


def make_debug_agent(task_id: str) -> DebugAgent:
    return DebugAgent(task_id)


if __name__ == "__main__":
    import sys

    import browsercomp_cube.debug as _mod
    from cube.testing import run_debug_suite

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

    results = run_debug_suite("browsercomp-cube", _mod)
    failed = [r for r in results if r["error"] or not r["done"] or r["reward"] < 1.0]
    sys.exit(1 if failed else 0)

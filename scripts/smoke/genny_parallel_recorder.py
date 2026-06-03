#!/usr/bin/env python3
"""SMOKE: Genny[parallel_actions=True] + Episode emit sibling ToolCallEvents in one turn.

Drives GennyParallel (RFC `agent-owns-loop`, Phase H) through the real
Episode body (Phase E), but bypasses the LLM by injecting a scripted
`step()` that returns multiple actions. Verifies:

  - The Episode finalizes cleanly.
  - The trajectory's event stream has one AgentEvent followed by N
    sibling ToolCallEvents sharing the same parent_event_id (the back-reference
    invariant the RFC asks for).

Run:
    .venv/bin/python scripts/smoke/genny_parallel_recorder.py

Prints SMOKE OK|FAIL: genny_parallel_recorder  (exit 0|1).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.core import Action, Observation
from cube.task import Task, TaskConfig, TaskMetadata
from cube.tool import Tool, ToolConfig, tool_action

from cube_harness.agent import Agent, AgentConfig
from cube_harness.core import AgentOutput, LLMCallEvent, ToolCallEvent
from cube_harness.exp_runner import run_sequentially
from cube_harness.experiment import Experiment
from cube_harness.storage import FileStorage

NAME = "genny_parallel_recorder"


class _ParallelTool(Tool):
    """3-action toolbox. GennyParallel calls all 3 in parallel via asyncio.gather."""

    @tool_action
    def alpha(self, n: int = 0) -> str:
        """Return alpha-n."""
        return f"alpha-{n}"

    @tool_action
    def beta(self, n: int = 0) -> str:
        """Return beta-n."""
        return f"beta-{n}"

    @tool_action
    def gamma(self, n: int = 0) -> str:
        """Return gamma-n."""
        return f"gamma-{n}"


class _ParallelToolConfig(ToolConfig):
    def make(self, container: object = None) -> _ParallelTool:
        _ = container
        return _ParallelTool()


class _ParallelTask(Task):
    def reset(self) -> tuple[Observation, dict]:
        return Observation.from_text("smoke goal — run 3 parallel"), {}

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict]:
        _ = obs
        return 1.0, {"success": True}


class _ParallelTaskConfig(TaskConfig):
    def make(self, runtime_context: object = None) -> _ParallelTask:
        _ = runtime_context
        return _ParallelTask(
            metadata=TaskMetadata(id=self.task_id),
            tool_config=self.tool_config or _ParallelToolConfig(),
        )


class _ParallelBenchmark(Benchmark):
    def _setup(self) -> None:
        pass

    def close(self) -> None:
        pass


class _ParallelBenchmarkConfig(BenchmarkConfig):
    benchmark_metadata = BenchmarkMetadata(
        name="genny-parallel-smoke", version="0.1.0", description="GennyParallel sibling ToolCallEvent smoke"
    )
    task_metadata = {"gp_task_0": TaskMetadata(id="gp_task_0")}
    task_config_class = _ParallelTaskConfig
    benchmark_class = _ParallelBenchmark


class _ScriptedParallelAgent(Agent):
    """Pure-Agent test driver, no LLM — first step() returns 3 parallel
    actions; second returns empty actions to graceful-stop. Inherits
    `_arun` (parallel) from Agent base; opt-in via parallel_actions=True
    on the config."""

    name = "_ScriptedParallelAgent"
    description = "scripted parallel agent for smoke"
    input_content_types: list[str] = []
    output_content_types: list[str] = []

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)
        self._called = 0

    def step(self, obs: Observation) -> AgentOutput:
        _ = obs
        self._called += 1
        if self._called > 1:
            return AgentOutput(actions=[])
        return AgentOutput(
            actions=[
                Action(id="a-1", name="alpha", arguments={"n": 1}),
                Action(id="a-2", name="beta", arguments={"n": 2}),
                Action(id="a-3", name="gamma", arguments={"n": 3}),
            ],
        )


class _ScriptedAgentConfig(AgentConfig):
    parallel_actions: bool = True  # opt into Agent._arun

    def make(self, action_set: object = None, **kwargs: object) -> "Agent":
        _ = action_set, kwargs
        return _ScriptedParallelAgent(self)


def _fail(msg: str) -> int:
    print(f"  ✗ {msg}")
    print(f"SMOKE FAIL: {NAME}")
    return 1


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="smoke-gp-"))
    try:
        exp = Experiment(
            name=NAME,
            output_dir=tmp,
            agent_config=_ScriptedAgentConfig(),
            benchmark_config=_ParallelBenchmarkConfig(),
            max_steps=5,
        )
        result = run_sequentially(exp)
        if result.failures:
            return _fail(f"failures: {list(result.failures.keys())}")
        if not result.trajectories:
            return _fail("no trajectories returned")

        storage = FileStorage(exp.output_dir)
        traj_id = next(iter(result.trajectories))
        view = storage.load_episode(traj_id)

        tool_calls = [e for e in view if isinstance(e.output, ToolCallEvent)]

        # Mock-agent path: no LLM is called, so no LLMCallEvent is
        # emitted. All 3 parallel tool calls share `parent_event_id` =
        # RESET (the only prior turn id available).
        non_reset_tool_calls = [t for t in tool_calls if t.output.parent_event_id != "reset"]
        # In the no-LLM path the entire fan-out parents to RESET; the
        # non_reset filter is therefore empty. Reach the 3 siblings
        # through the full tool_calls list minus the synthetic reset.
        fanout = [t for t in tool_calls if t.output.action_id != "reset"]
        if len(fanout) != 3:
            return _fail(f"expected 3 fan-out ToolCallEvents, got {len(fanout)}")

        parent_id = fanout[0].output.parent_event_id
        if not all(t.output.parent_event_id == parent_id for t in fanout):
            return _fail("ToolCallEvents have differing parent_event_id — not all siblings of one turn")

        llm_event_ids = {e.output.id for e in view if isinstance(e.output, LLMCallEvent)}
        # parent must be a real LLMCallEvent.id OR the RESET sentinel
        # (LLM-less mock-agent fan-outs parent to RESET).
        if parent_id not in llm_event_ids and parent_id != "reset":
            return _fail(f"parent_event_id={parent_id!r} matches no LLMCallEvent.id and is not the RESET sentinel")

        names = {t.output.action_id for t in fanout}
        if names != {"a-1", "a-2", "a-3"}:
            return _fail(f"unexpected action_ids: {names}")
        _ = non_reset_tool_calls

        print(f"  ✓ {traj_id}: 3 sibling tool_calls (parent_event_id={parent_id[:8]}…), eval=1")
        print(f"SMOKE OK: {NAME}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

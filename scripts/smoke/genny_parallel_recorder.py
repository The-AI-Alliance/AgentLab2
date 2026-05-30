#!/usr/bin/env python3
"""SMOKE: GennyParallel + Episode emit sibling ToolCallEvents in one turn.

Drives GennyParallel (RFC `agent-owns-loop`, Phase H) through the real
Episode body (Phase E), but bypasses the LLM by injecting a scripted
`step()` that returns multiple actions. Verifies:

  - The Episode finalizes cleanly.
  - The trajectory's event stream has one AgentEvent followed by N
    sibling ToolCallEvents sharing the same turn_id (the back-reference
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
from cube_harness.agents.genny_parallel import GennyParallel
from cube_harness.core import AgentEvent, AgentOutput, ToolCallEvent
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


class _ScriptedParallelAgent(GennyParallel):
    """GennyParallel without LLM — first step() returns 3 parallel
    actions; second returns empty actions to graceful-stop."""

    def __init__(self, _config: AgentConfig) -> None:
        # Skip Genny.__init__ — we don't need the LLM machinery.
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
            thoughts="three parallel calls",
        )


class _ScriptedAgentConfig(AgentConfig):
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
        loaded = storage.load_trajectory(traj_id)

        agent_events = [e for e in loaded.events if isinstance(e.output, AgentEvent)]
        tool_calls = [e for e in loaded.events if isinstance(e.output, ToolCallEvent)]

        # Expect ≥1 AgentEvent + 3 tool-call siblings + the reset
        # ToolCallEvent + possibly a graceful-stop AgentEvent.
        non_reset_tool_calls = [t for t in tool_calls if t.output.parent_event_id != "reset"]
        if len(non_reset_tool_calls) != 3:
            return _fail(f"expected 3 sibling ToolCallEvents, got {len(non_reset_tool_calls)}")

        parent_id = non_reset_tool_calls[0].output.parent_event_id
        if not all(t.output.parent_event_id == parent_id for t in non_reset_tool_calls):
            return _fail("ToolCallEvents have differing parent_event_id — not all siblings")
        if not all(t.output.turn_id == parent_id for t in non_reset_tool_calls):
            return _fail("ToolCallEvents have differing turn_id — not all in one turn")

        # The parent_event_id must reference a real AgentEvent.id.
        if parent_id not in {a.output.id for a in agent_events}:
            return _fail(f"parent_event_id={parent_id!r} does not match any AgentEvent.id")

        names = {t.output.action_id for t in non_reset_tool_calls}
        if names != {"a-1", "a-2", "a-3"}:
            return _fail(f"unexpected action_ids: {names}")

        print(f"  ✓ {traj_id}: 1 parent agent event, 3 sibling tool_calls (turn_id={parent_id[:8]}…), eval=1")
        print(f"SMOKE OK: {NAME}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

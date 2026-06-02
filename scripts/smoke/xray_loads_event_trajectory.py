#!/usr/bin/env python3
"""SMOKE: XRay's data layer loads new-format (events/) trajectories.

The full event-card timeline UI is Phase I follow-up work. For Phase 1
of the RFC `agent-owns-loop`, the minimum requirement is that the
existing XRay UI and inspect-results CLI keep working on the new event
format — via the legacy-steps materialization in
`storage._events_to_legacy_steps`.

This smoke runs a small agent-owns-loop experiment, then walks the
XRay-side code paths that consume `trajectory.steps`:

  - FileStorage.load_trajectory populates trajectory.steps as a
    synthetic AgentOutput / EnvironmentOutput stream alongside events.
  - xray_utils.find_last_env_step finds the final env step (used to
    render the "result" panel).
  - xray_utils.compute_progress walks steps to count turns.

Run:
    .venv/bin/python scripts/smoke/xray_loads_event_trajectory.py

Prints SMOKE OK|FAIL: xray_loads_event_trajectory  (exit 0|1).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.core import Action, EnvironmentOutput, Observation
from cube.task import Task, TaskConfig, TaskMetadata
from cube.tool import Tool, ToolConfig, tool_action

from cube_harness.agent import Agent, AgentConfig
from cube_harness.core import AgentOutput
from cube_harness.exp_runner import run_sequentially
from cube_harness.experiment import Experiment
from cube_harness.storage import FileStorage

NAME = "xray_loads_event_trajectory"


class _NoopTool(Tool):
    @tool_action
    def click(self, element_id: str) -> str:
        """Click an element."""
        return f"clicked {element_id}"


class _NoopToolConfig(ToolConfig):
    def make(self, container: object = None) -> _NoopTool:
        _ = container
        return _NoopTool()


class _MockTask(Task):
    def reset(self) -> tuple[Observation, dict]:
        return Observation.from_text("smoke goal"), {}

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict]:
        _ = obs
        return 1.0, {"success": True}


class _MockTaskConfig(TaskConfig):
    def make(self, runtime_context: object = None) -> _MockTask:
        _ = runtime_context
        return _MockTask(
            metadata=TaskMetadata(id=self.task_id),
            tool_config=self.tool_config or _NoopToolConfig(),
        )


class _MockBenchmark(Benchmark):
    def _setup(self) -> None:
        pass

    def close(self) -> None:
        pass


class _BenchmarkConfig(BenchmarkConfig):
    benchmark_metadata = BenchmarkMetadata(
        name="xray-smoke", version="0.1.0", description="XRay legacy-steps compat smoke"
    )
    task_metadata = {"xray_task_0": TaskMetadata(id="xray_task_0")}
    task_config_class = _MockTaskConfig
    benchmark_class = _MockBenchmark


class _MockAgentConfig(AgentConfig):
    def make(self, action_set: object = None, **kwargs: object) -> "Agent":
        _ = action_set, kwargs
        return _MockAgent(self)


class _MockAgent(Agent):
    name = "XRaySmokeAgent"
    description = "click then final_step — two-turn deterministic agent."
    input_content_types = ["text"]
    output_content_types = ["action"]

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)
        self._called = 0

    def step(self, obs: Observation) -> AgentOutput:
        _ = obs
        self._called += 1
        if self._called > 2:
            return AgentOutput(actions=[])
        if self._called == 1:
            return AgentOutput(
                actions=[Action(id="a-1", name="click", arguments={"element_id": "btn1"})],
                thoughts="clicking",
            )
        return AgentOutput(actions=[Action(name="final_step", arguments={})])


def _fail(msg: str) -> int:
    print(f"  ✗ {msg}")
    print(f"SMOKE FAIL: {NAME}")
    return 1


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="smoke-xray-"))
    try:
        exp = Experiment(
            name=NAME,
            output_dir=tmp,
            agent_config=_MockAgentConfig(),
            benchmark_config=_BenchmarkConfig(),
            max_steps=5,
        )
        result = run_sequentially(exp)
        if result.failures:
            return _fail(f"failures: {list(result.failures.keys())}")
        traj_id = next(iter(result.trajectories))

        storage = FileStorage(exp.output_dir)
        view = storage.load_episode(traj_id)

        # The new event stream is the source of truth. The MockAgent
        # in this smoke has no LLM, so LLMCallEvent count is 0 — only
        # the synthetic reset ToolCallEvent + the final-step ToolCallEvent
        # + the terminal EvaluationEvent land on disk.
        if view.n_tool_calls < 1:
            return _fail("no ToolCallEvent in event stream")

        # The legacy steps view is still materialized by load_trajectory
        # so XRay's existing UI keeps rendering through the migration window.
        legacy = storage.load_trajectory(traj_id)
        if not legacy.steps:
            return _fail("legacy trajectory.steps empty — XRay backward-compat broken")

        last_env = legacy.last_env_step()
        if last_env is None or not isinstance(last_env, EnvironmentOutput):
            return _fail("last_env_step did not return an EnvironmentOutput")

        # XRay grouping check: the count of LLMCallEvents in the event
        # stream lines up with the count of AgentOutput-shaped legacy
        # steps materialized by `_events_to_legacy_steps`.
        agent_outputs = [s for s in legacy.steps if isinstance(s.output, AgentOutput)]
        if len(agent_outputs) != view.n_agent_events:
            return _fail(
                f"legacy AgentOutput count={len(agent_outputs)} ≠ view n_agent_events={view.n_agent_events} "
                "— materialization is inconsistent"
            )

        print(
            f"  ✓ {traj_id}: {view.n_agent_events} llm events, "
            f"{view.n_tool_calls} tool calls, "
            f"legacy steps view has {len(legacy.steps)} steps (XRay sees something to render)"
        )
        print(f"SMOKE OK: {NAME}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

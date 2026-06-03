#!/usr/bin/env python3
"""SMOKE: agent-owns-loop end-to-end (events/, MonitoredTool, EventStreamer).

Runs a small deterministic mock-cube experiment through the rewritten
Episode (RFC `agent-owns-loop`, Phase E) and verifies:

  - the events/ dir is written alongside the legacy steps/ dir;
  - the per-event filenames carry the right `kind` suffix
    (NNN_agent / NNN_tool_call / NNN_eval);
  - the trajectory reloads via FileStorage and the event stream is
    complete (reset ToolCallEvent + per-turn LLMCallEvent / ToolCallEvent
    pairs + final EvaluationEvent);
  - ToolCallEvents reference their parent LLMCallEvent.id via
    parent_event_id (sibling parallel tool calls share that id);
  - summary_stats carries the agent-owns-loop counters (n_agent_events,
    n_tool_calls, n_evaluations folded into n_agent_steps / n_env_steps
    for backward compatibility).

Run:
    .venv/bin/python scripts/smoke/agent_owns_loop_events.py

Prints SMOKE OK|FAIL: agent_owns_loop_events  (exit 0|1).
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
from cube_harness.core import AgentOutput, EvaluationEvent, LLMCallEvent, ToolCallEvent
from cube_harness.exp_runner import run_sequentially
from cube_harness.experiment import Experiment
from cube_harness.storage import EVENTS_DIR, FileStorage

NAME = "agent_owns_loop_events"
N_TASKS = 2


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


class _SmokeBenchmarkConfig(BenchmarkConfig):
    benchmark_metadata = BenchmarkMetadata(name="aol-smoke", version="0.1.0", description="agent-owns-loop smoke")
    task_metadata = {f"aol_task_{i}": TaskMetadata(id=f"aol_task_{i}") for i in range(N_TASKS)}
    task_config_class = _MockTaskConfig
    benchmark_class = _MockBenchmark


class _MockAgentConfig(AgentConfig):
    def make(self, action_set: object = None, **kwargs: object) -> "Agent":
        _ = action_set, kwargs
        return _MockAgent(config=self)


class _MockAgent(Agent):
    """Submits final_step immediately — deterministic, no LLM."""

    name = "AolSmokeAgent"
    description = "Submits final_step immediately."
    input_content_types = ["text"]
    output_content_types = ["action"]

    def step(self, obs: Observation) -> AgentOutput:
        _ = obs
        return AgentOutput(actions=[Action(name="final_step", arguments={})])


def _fail(msg: str) -> int:
    print(f"  ✗ {msg}")
    print(f"SMOKE FAIL: {NAME}")
    return 1


def _check_trajectory(storage: FileStorage, traj_id: str) -> int:
    ep_dir = storage._episode_dir(traj_id)
    events_dir = ep_dir / EVENTS_DIR
    if not events_dir.exists():
        return _fail(f"events/ dir missing under {ep_dir}")
    files = sorted(events_dir.iterdir())
    kinds = [f.name.split("_", 1)[1] for f in files]
    if not any("tool_call" in k for k in kinds):
        return _fail(f"no tool_call event files: {kinds}")
    if not any("eval" in k for k in kinds):
        return _fail(f"no eval event files: {kinds}")
    # MockAgent has no LLM → no LLMCallEvent is expected. LLM agents
    # (Genny, React, ...) auto-emit LLMCallEvents via `LLM.attach_recorder`.

    view = storage.load_episode(traj_id)
    if view.n_tool_calls < 1:
        return _fail(f"expected ≥1 ToolCallEvent, got {view.n_tool_calls}")
    if view.n_evaluations != 1:
        return _fail(f"expected exactly 1 EvaluationEvent, got {view.n_evaluations}")

    # Back-reference invariant: each ToolCallEvent.parent_event_id must
    # be a preceding LLMCallEvent.id (or the RESET sentinel).
    agent_event_ids: set[str] = {"reset"}
    for e in view:
        if isinstance(e.output, LLMCallEvent):
            agent_event_ids.add(e.output.id)
        elif isinstance(e.output, ToolCallEvent):
            if e.output.parent_event_id not in agent_event_ids:
                return _fail(
                    f"ToolCallEvent.parent_event_id={e.output.parent_event_id!r} references no preceding LLMCallEvent.id"
                )
        elif isinstance(e.output, EvaluationEvent):
            pass

    # Summary stats fold the new event stream into the legacy counters
    # so XRay tables keep working.
    if view.summary_stats is None:
        return _fail("summary_stats missing on loaded view")
    if view.summary_stats.get("n_env_steps", 0) < 1:
        return _fail(f"summary_stats.n_env_steps={view.summary_stats.get('n_env_steps')} — expected ≥1")

    print(f"  ✓ {traj_id}: {view.n_agent_events} agent, {view.n_tool_calls} tool_call, {view.n_evaluations} eval")
    return 0


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="smoke-aol-"))
    try:
        exp = Experiment(
            name=NAME,
            output_dir=tmp,
            agent_config=_MockAgentConfig(),
            benchmark_config=_SmokeBenchmarkConfig(),
            max_steps=3,
        )
        result = run_sequentially(exp)
        if result.failures:
            return _fail(f"{len(result.failures)} episodes failed: {list(result.failures.keys())}")
        storage = FileStorage(exp.output_dir)
        for traj_id, _t in result.trajectories.items():
            if _check_trajectory(storage, traj_id) != 0:
                return 1
        print(f"SMOKE OK: {NAME}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

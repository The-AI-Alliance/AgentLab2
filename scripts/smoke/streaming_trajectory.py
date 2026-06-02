#!/usr/bin/env python3
"""SMOKE: the runner streams steps to disk and returns step-less trajectories.

Runs a small deterministic mock-cube experiment through the real sequential runner and
verifies the streaming contract end-to-end:

  - every trajectory in ExpResult.trajectories has steps == [] (nothing accumulated in the
    driver — this is the OSWorld ~20 GB fix);
  - summary_stats and reward_info are populated on those step-less trajectories;
  - the steps are fully persisted and reload from disk via FileStorage.load_trajectory;
  - print_stats and the eval-log export work off summary_stats (no len(steps)/last_env_step).

The Ray path returns whatever Episode.run() returns (the same step-less trajectory), so the
sequential path exercises the same contract without Ray-worker pickling of __main__ classes.

Run:
    .venv/bin/python scripts/smoke/streaming_trajectory.py

Prints SMOKE OK|FAIL: streaming_trajectory  (exit 0|1).
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
from cube_harness.core import AgentOutput
from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment, ExpResult
from cube_harness.storage import FileStorage

NAME = "streaming_trajectory"
N_TASKS = 4


class _NoopTool(Tool):
    @tool_action
    def click(self, element_id: str) -> str:
        """Click an element.

        Args:
            element_id: The element to click.
        """
        return f"clicked {element_id}"


class _NoopToolConfig(ToolConfig):
    def make(self, container: object = None) -> _NoopTool:
        _ = container
        return _NoopTool()


class _MockTask(Task):
    def reset(self) -> tuple[Observation, dict]:
        return Observation.from_text("smoke task goal"), {}

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict]:
        _ = obs
        return 1.0, {"success": True}


class _MockTaskConfig(TaskConfig):
    def make(self, runtime_context: object = None) -> _MockTask:
        _ = runtime_context
        return _MockTask(metadata=TaskMetadata(id=self.task_id), tool_config=self.tool_config or _NoopToolConfig())


class _MockBenchmark(Benchmark):
    def _setup(self) -> None:
        pass

    def close(self) -> None:
        pass


class _SmokeBenchmarkConfig(BenchmarkConfig):
    benchmark_metadata = BenchmarkMetadata(name="stream-smoke", version="0.1.0", description="streaming smoke")
    task_metadata = {f"smoke_task_{i}": TaskMetadata(id=f"smoke_task_{i}") for i in range(N_TASKS)}
    task_config_class = _MockTaskConfig
    benchmark_class = _MockBenchmark


class _MockAgentConfig(AgentConfig):
    def make(self, action_set: object = None, **kwargs: object) -> "Agent":
        _ = action_set, kwargs
        return _MockAgent(config=self)


class _MockAgent(Agent):
    name = "StreamSmokeAgent"
    description = "Submits final_step immediately — deterministic, no LLM."
    input_content_types = ["text"]
    output_content_types = ["action"]

    def step(self, obs: Observation) -> AgentOutput:
        _ = obs
        return AgentOutput(actions=[Action(name="final_step", arguments={})])


def _fail(msg: str) -> int:
    print(f"  ✗ {msg}")
    print(f"SMOKE FAIL: {NAME}")
    return 1


def _check(label: str, exp: Experiment, result: ExpResult) -> int:
    """Assert the streaming contract for one runner's output. Returns 0 on success."""
    storage = FileStorage(exp.output_dir)
    if len(result.trajectories) != N_TASKS:
        return _fail(f"[{label}] expected {N_TASKS} trajectories, got {len(result.trajectories)}")

    for traj_id, view in result.trajectories.items():
        # 1. Returned TrajectoryView holds no decoded events in RAM (cache empty).
        if view._cache:
            return _fail(f"[{label}] {traj_id}: returned view cache pre-populated with {len(view._cache)} events")
        if not view.summary_stats:
            return _fail(f"[{label}] {traj_id}: summary_stats missing on returned view")
        if (view.reward_info or {}).get("reward") != 1.0:
            return _fail(f"[{label}] {traj_id}: reward_info missing/wrong: {view.reward_info}")

        # 2. Events are fully persisted and reload from disk.
        reopened = storage.load_episode(traj_id)
        if len(reopened) < 2:
            return _fail(f"[{label}] {traj_id}: expected >=2 persisted events, got {len(reopened)}")
        if reopened.summary_stats != view.summary_stats:
            return _fail(f"[{label}] {traj_id}: summary_stats changed across disk round-trip")

    # 3. Stats + eval-log export work off summary_stats (would raise/return 0 if broken).
    exp.print_stats(result)
    eval_log = exp.export_eval_log()
    if len(eval_log.episodes) != N_TASKS:
        return _fail(f"[{label}] eval-log has {len(eval_log.episodes)} episodes, expected {N_TASKS}")
    # The MockAgent has no LLM, so `n_agent_steps` (= LLMCallEvent count) is 0.
    # Each task fires 1 env step via the synthetic ToolCallEvent — num_turns counts that.
    if not all(ep.num_turns >= 1 and ep.score == 1.0 for ep in eval_log.episodes):
        return _fail(f"[{label}] eval-log records have wrong num_turns/score (not derived from summary_stats)")

    print(f"  ✓ [{label}] {N_TASKS} step-less returns; steps + summary on disk; eval-log derived from summary")
    return 0


def _experiment(output_dir: Path) -> Experiment:
    return Experiment(
        name="stream_smoke",
        output_dir=output_dir,
        agent_config=_MockAgentConfig(),
        benchmark_config=_SmokeBenchmarkConfig(),
        max_steps=2,
    )


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="streaming_traj_")
    try:
        # Sequential: in-process runner (driver == worker).
        exp_seq = _experiment(Path(tmp) / "seq")
        rc = _check("sequential", exp_seq, run_sequentially(exp_seq))
        if rc:
            return rc

        # Ray: the parallel runner — the path where the ~20 GB driver accumulation
        # manifested (workers return trajectories to the driver via the object store).
        exp_ray = _experiment(Path(tmp) / "ray")
        rc = _check("ray", exp_ray, run_with_ray(exp_ray, n_cpus=2))
        if rc:
            return rc

        print(f"SMOKE OK: {NAME}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

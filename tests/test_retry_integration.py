"""End-to-end integration test for the episode-status retry mechanism.

A single Ray-based run over a 4-task benchmark covers ~80% of the retry machinery:
pre-claim, step-boundary heartbeat, driver-side stale-heartbeat kill, CANCELLED
status, auto-retry rounds, retry_count cap, and per-attempt archives.

Each task's "scenario" is a list of behaviours indexed by attempt number
(0 = original, 1.. = retries):

| Episode    | Scenarios                | Expected final | retry_count | archives |
|------------|--------------------------|----------------|-------------|----------|
| task_succeed | ["succeed"]            | COMPLETED      | 0           | 0        |
| task_flaky   | ["fail","fail","ok"]   | COMPLETED      | 2           | 2        |
| task_dead    | ["fail"]*4             | FAILED (cap)   | 3           | 3        |
| task_hang    | ["hang","succeed"]     | COMPLETED      | 1           | 1        |
"""

from __future__ import annotations

import fcntl
import time
from pathlib import Path

import litellm
import pytest
import ray
from cube.benchmark import Benchmark as CubeBenchmark
from cube.benchmark import BenchmarkConfig as CubeBenchmarkConfig
from cube.benchmark import BenchmarkMetadata
from cube.core import Action, Observation
from cube.task import Task as CubeTask
from cube.task import TaskConfig as CubeTaskConfig
from cube.task import TaskMetadata

from cube_harness.agent import Agent, AgentConfig
from cube_harness.core import AgentOutput
from cube_harness.episode_status import STATUS_FILENAME, EpisodeStatus
from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment
from cube_harness.storage import ARCHIVED_MARKER, EPISODES_DIR, FileStorage
from tests.conftest import MockToolConfig

# Per-test slow markers below — Ray tests are slow (~30s), sequential tests are fast (~1s).


# Scenarios for the main 4-scenario Ray integration test.
SCENARIOS = {
    "task_succeed": ["succeed"],
    "task_flaky": ["fail", "fail", "succeed"],
    "task_dead": ["fail", "fail", "fail", "fail"],
    "task_hang": ["hang", "succeed"],
}


# --- Debug task / benchmark ---


class DebugCubeTask(CubeTask):
    """Cube task that exposes its own task_id in the initial observation.

    The DebugAgent reads this to look up the per-task scripted scenario.
    """

    def reset(self) -> tuple[Observation, dict]:
        return Observation.from_text(f"task_id={self.metadata.id}"), {"task_id": self.metadata.id}

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict]:
        return 1.0, {"success": True}


class DebugCubeTaskConfig(CubeTaskConfig):
    def make(self, runtime_context=None) -> DebugCubeTask:
        return DebugCubeTask(
            metadata=self.metadata,
            tool_config=self.tool_config or MockToolConfig(),
        )


class _DebugBenchmark(CubeBenchmark):
    """Live runtime pair for the retry-integration debug benchmark."""

    def _setup(self) -> None:
        pass

    def close(self) -> None:
        pass


def make_debug_benchmark(scenarios: dict[str, list[str]]) -> CubeBenchmarkConfig:
    """Build a CubeBenchmarkConfig whose tasks match the keys of `scenarios`."""

    class _DebugBenchmarkConfig(CubeBenchmarkConfig):
        benchmark_metadata = BenchmarkMetadata(
            name="debug-retry",
            version="0.1.0",
            description="Scripted scenarios for retry-mechanism integration test",
        )
        task_metadata = {tid: TaskMetadata(id=tid) for tid in scenarios}
        task_config_class = DebugCubeTaskConfig
        benchmark_class = _DebugBenchmark

    return _DebugBenchmarkConfig()


# --- Debug agent ---


def _next_attempt(counter_dir: str, task_id: str) -> int:
    """Atomically read+increment a per-task attempt counter on disk.

    Uses fcntl to serialise concurrent writers (no contention expected within a
    single round, but cheap insurance against future parallelism bugs).
    """
    Path(counter_dir).mkdir(parents=True, exist_ok=True)
    path = Path(counter_dir) / f"{task_id}.txt"
    with open(path, "a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            raw = f.read().strip()
            current = int(raw) if raw else 0
            f.seek(0)
            f.truncate()
            f.write(str(current + 1))
            return current
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class DebugAgentConfig(AgentConfig):
    """Agent config carrying scenarios + counter dir.

    All fields are picklable plain data, so Ray can ship the config to workers.
    """

    counter_dir: str
    scenarios: dict[str, list[str]]
    hang_seconds: float = 60.0
    name: str = "debug_agent"

    def make(self, action_set=None, **kwargs) -> "DebugAgent":
        return DebugAgent(config=self)


class DebugAgent(Agent):
    name = "DebugAgent"
    description = "Scripted agent that reads scenarios per-task per-attempt"
    input_content_types = ["text"]
    output_content_types = ["action"]

    def __init__(self, config: DebugAgentConfig):
        super().__init__(config)
        self.config = config
        self._task_id: str | None = None

    def step(self, obs: Observation) -> AgentOutput:
        # Initial observation embeds task_id; cache it across subsequent steps.
        if self._task_id is None:
            text = obs.contents[0].data if obs.contents else ""
            self._task_id = text.replace("task_id=", "").strip()
        task_id = self._task_id
        scenarios = self.config.scenarios[task_id]
        attempt = _next_attempt(self.config.counter_dir, task_id)
        # Saturate at the last scenario if attempts exceed the script length.
        behavior = scenarios[min(attempt, len(scenarios) - 1)]

        if behavior == "succeed":
            return AgentOutput(actions=[Action(name="final_step", arguments={})])
        if behavior == "fail":
            raise RuntimeError(f"scripted failure: {task_id} attempt {attempt}")
        if behavior == "hang":
            time.sleep(self.config.hang_seconds)
            return AgentOutput(actions=[Action(name="final_step", arguments={})])
        if behavior == "loop":
            # Non-terminating action — drives the runner toward max_steps.
            return AgentOutput(actions=[Action(name="click", arguments={"element_id": "x"})])
        if behavior == "bad_model":
            raise litellm.NotFoundError(
                message="model 'gpt-5-mini-typo-this-model-does-not-exist' does not exist",
                model="gpt-5-mini-typo-this-model-does-not-exist",
                llm_provider="openai",
            )
        raise ValueError(f"Unknown behavior: {behavior}")


# --- The test ---


@pytest.fixture(autouse=True)
def _ray_shutdown_between_tests():
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.mark.slow
def test_retry_machinery_end_to_end(tmp_dir: Path) -> None:
    """4-task Ray run exercises the full retry mechanism."""
    counter_dir = str(tmp_dir / "_attempt_counters")
    agent_config = DebugAgentConfig(counter_dir=counter_dir, scenarios=SCENARIOS, hang_seconds=10.0)
    benchmark = make_debug_benchmark(SCENARIOS)

    exp = Experiment(
        name="retry_integration",
        output_dir=tmp_dir,
        agent_config=agent_config,
        benchmark_config=benchmark,
        max_retries=3,
    )

    result = run_with_ray(
        exp,
        n_cpus=2,
        ray_poll_timeout=0.5,
        step_timeout_s=1.5,
        cancel_grace_s=0.5,
        orphan_threshold_s=30.0,
    )

    storage = FileStorage(tmp_dir)
    statuses = storage.list_episode_statuses()
    # All four episode dirs exist.
    assert set(statuses.keys()) == {f"{tid}_ep{i}" for i, tid in enumerate(SCENARIOS)}

    # task_succeed: COMPLETED on first try, retry_count 0, no archives.
    s0 = statuses["task_succeed_ep0"]
    assert s0.status == "COMPLETED", s0
    assert s0.retry_count == 0
    assert _archive_count(tmp_dir, "task_succeed_ep0") == 0

    # task_flaky: 2 retries, finally COMPLETED. retry_count = 2 on the live attempt.
    # 2 archived dirs (the two failed attempts).
    s1 = statuses["task_flaky_ep1"]
    assert s1.status == "COMPLETED", s1
    assert s1.retry_count == 2
    assert _archive_count(tmp_dir, "task_flaky_ep1") == 2

    # task_dead: 4 attempts, all failed. retry_count = 3 (capped). 3 archives.
    s2 = statuses["task_dead_ep2"]
    assert s2.status == "FAILED", s2
    assert s2.retry_count == 3
    assert s2.error_type == "RuntimeError"
    assert _archive_count(tmp_dir, "task_dead_ep2") == 3

    # task_hang: first attempt CANCELLED via step-timeout, second succeeds.
    # 1 archive (the cancelled attempt).
    s3 = statuses["task_hang_ep3"]
    assert s3.status == "COMPLETED", s3
    assert s3.retry_count == 1
    assert _archive_count(tmp_dir, "task_hang_ep3") == 1

    # The CANCELLED attempt is preserved in the archive — verify error_type.
    archived_status = _read_archived_status(tmp_dir, "task_hang_ep3")
    assert archived_status is not None
    assert archived_status["status"] == "CANCELLED", archived_status
    assert archived_status["error_type"] == "StepTimeout"

    # Per-attempt forensics: walking task_dead's archives gives retry_counts 0..2 with FAILED + populated error_type.
    dead_archives = _read_all_archived_statuses(tmp_dir, "task_dead_ep2")
    assert len(dead_archives) == 3
    dead_retry_counts = sorted(a["retry_count"] for a in dead_archives)
    assert dead_retry_counts == [0, 1, 2]
    for archived in dead_archives:
        assert archived["status"] == "FAILED"
        assert archived["error_type"] == "RuntimeError"
        assert "scripted failure" in (archived["error_message"] or "")

    # Aggregated result: 3 successful trajectories, 1 failure (task_dead).
    assert "task_dead_ep2" in result.failures
    assert {"task_succeed_ep0", "task_flaky_ep1", "task_hang_ep3"}.issubset(result.trajectories.keys())


@pytest.mark.slow
def test_mixed_state_recovery_via_ray(tmp_dir: Path) -> None:
    """Restart with mixed pre-existing state: COMPLETED preserved, STALE swept+retried, missing retried.

    Simulates "the prior driver crashed mid-experiment" by hand-writing a heterogeneous
    `output_dir` and verifies the new driver makes the right decision per episode:

    - `task_already_done`: COMPLETED status → never re-run.
    - `task_was_stuck`: stale RUNNING → swept to STALE → retried → COMPLETED.
    - `task_never_started`: no status.json → retried (resume) → COMPLETED.
    """
    scenarios = {
        "task_already_done": ["succeed"],  # won't actually be re-run
        "task_was_stuck": ["succeed"],
        "task_never_started": ["succeed"],
    }
    benchmark = make_debug_benchmark(scenarios)
    agent_config = DebugAgentConfig(
        counter_dir=str(tmp_dir / "_counters"),
        scenarios=scenarios,
        hang_seconds=10.0,
    )
    exp = Experiment(
        name="mixed_state",
        output_dir=tmp_dir,
        agent_config=agent_config,
        benchmark_config=benchmark,
        max_retries=3,
    )

    # Materialise episode_config.json for all three (without running anything).
    with benchmark.make() as bm:
        episodes = {ep.config.task_config.task_id: ep for ep in exp.get_episodes_to_run(bm)}
    storage = FileStorage(tmp_dir)

    # Hand-craft pre-existing state to mimic a crashed prior driver:
    done_traj_id = f"task_already_done_ep{episodes['task_already_done'].config.id}"
    storage.write_episode_status(
        done_traj_id,
        EpisodeStatus(
            status="COMPLETED",
            task_id="task_already_done",
            episode_id=episodes["task_already_done"].config.id,
            started_at=time.time() - 100,
            ended_at=time.time() - 90,
            last_heartbeat_at=time.time() - 90,
            reward=1.0,
            retry_count=0,
        ),
    )

    stuck_traj_id = f"task_was_stuck_ep{episodes['task_was_stuck'].config.id}"
    storage.write_episode_status(
        stuck_traj_id,
        EpisodeStatus(
            status="RUNNING",
            task_id="task_was_stuck",
            episode_id=episodes["task_was_stuck"].config.id,
            started_at=time.time() - 7200,
            last_heartbeat_at=time.time() - 7200,  # stale (>>step_timeout+grace)
            current_step=4,
            retry_count=0,
        ),
    )

    missing_traj_id = f"task_never_started_ep{episodes['task_never_started'].config.id}"
    # No status.json written for this one — exists as an episode_config only.

    # New driver: resume picks up both the swept-STALE and the missing-status episode.
    exp.resume = True
    result = run_with_ray(
        exp,
        n_cpus=2,
        ray_poll_timeout=0.5,
        step_timeout_s=10.0,
        cancel_grace_s=1.0,
        orphan_threshold_s=10.0,
    )

    # task_already_done: untouched.
    done = storage.read_episode_status(done_traj_id)
    assert done is not None
    assert done.status == "COMPLETED"
    assert done.retry_count == 0
    assert _archive_count(tmp_dir, done_traj_id) == 0
    assert done_traj_id not in result.trajectories  # not re-run this round

    # task_was_stuck: STALE-swept, then retried to COMPLETED. Prior STALE preserved in archive.
    stuck = storage.read_episode_status(stuck_traj_id)
    assert stuck is not None
    assert stuck.status == "COMPLETED", stuck
    assert stuck.retry_count == 1
    stuck_archives = _read_all_archived_statuses(tmp_dir, stuck_traj_id)
    assert len(stuck_archives) == 1
    assert stuck_archives[0]["status"] == "STALE"
    assert stuck_archives[0]["retry_count"] == 0
    assert stuck_traj_id in result.trajectories

    # task_never_started: ran fresh, no archive (no prior attempt).
    missing = storage.read_episode_status(missing_traj_id)
    assert missing is not None
    assert missing.status == "COMPLETED"
    assert missing.retry_count == 0
    assert _archive_count(tmp_dir, missing_traj_id) == 0
    assert missing_traj_id in result.trajectories


def test_run_sequentially_auto_retries_flaky_episode(tmp_dir: Path) -> None:
    """run_sequentially auto-retries; exercises the worker-side archive (no pre-claim path)."""
    scenarios = {"task_seq_flaky": ["fail", "succeed"]}
    benchmark = make_debug_benchmark(scenarios)
    agent_config = DebugAgentConfig(
        counter_dir=str(tmp_dir / "_counters"),
        scenarios=scenarios,
        hang_seconds=0.0,
    )
    exp = Experiment(
        name="seq_flaky",
        output_dir=tmp_dir,
        agent_config=agent_config,
        benchmark_config=benchmark,
        max_retries=3,
    )

    result = run_sequentially(exp)

    storage = FileStorage(tmp_dir)
    statuses = storage.list_episode_statuses()
    assert len(statuses) == 1
    traj_id, final = next(iter(statuses.items()))
    assert final.status == "COMPLETED"
    assert final.retry_count == 1

    # The failed attempt is preserved in an archive (worker-side path, no pre-claim).
    archived = _read_all_archived_statuses(tmp_dir, traj_id)
    assert len(archived) == 1
    assert archived[0]["status"] == "FAILED"
    assert archived[0]["retry_count"] == 0
    assert archived[0]["error_type"] == "RuntimeError"

    assert traj_id in result.trajectories


def test_max_steps_terminates_with_forced_eval(tmp_dir: Path) -> None:
    """Hitting max_steps produces MAX_STEPS_REACHED + a real reward (forced evaluate).

    The agent issues non-terminating actions, so the loop runs until `max_steps`
    fires. Without our forced final-eval the trajectory would record reward=0;
    the fix calls evaluate() once at end-of-loop so the reward reflects the task's
    real assessment (DebugCubeTask returns 1.0). Status is MAX_STEPS_REACHED, which
    is terminal but NOT retriable — a second resume run leaves it alone.
    """
    scenarios = {"task_loops": ["loop"]}
    benchmark = make_debug_benchmark(scenarios)
    agent_config = DebugAgentConfig(
        counter_dir=str(tmp_dir / "_counters"),
        scenarios=scenarios,
        hang_seconds=0.0,
    )
    exp = Experiment(
        name="max_steps",
        output_dir=tmp_dir,
        agent_config=agent_config,
        benchmark_config=benchmark,
        max_steps=2,
        max_retries=3,
    )

    result = run_sequentially(exp)

    storage = FileStorage(tmp_dir)
    statuses = storage.list_episode_statuses()
    assert len(statuses) == 1
    traj_id, final = next(iter(statuses.items()))
    assert final.status == "MAX_STEPS_REACHED", final
    assert final.retry_count == 0
    assert final.reward == 1.0  # from the forced evaluate call (DebugCubeTask returns 1.0)
    assert _archive_count(tmp_dir, traj_id) == 0  # no retries

    # The aggregated trajectory got a real reward, not 0.0.
    assert traj_id in result.trajectories
    assert result.trajectories[traj_id].last_env_step().reward == 1.0

    # MAX_STEPS_REACHED is not retriable: a fresh resume returns nothing.
    exp.resume = True
    with benchmark.make() as bm:
        assert exp.get_episodes_to_run(bm) == []


def test_max_retries_zero_disables_auto_retry(tmp_dir: Path) -> None:
    """max_retries=0 → a failing episode stays FAILED with no retry attempts.

    The auto-retry loop only retries episodes with `retry_count < max_retries`.
    Setting `max_retries=0` makes the cap bind immediately so the loop never runs
    a second round.
    """
    scenarios = {"task_no_retry": ["fail"]}
    benchmark = make_debug_benchmark(scenarios)
    agent_config = DebugAgentConfig(
        counter_dir=str(tmp_dir / "_counters"),
        scenarios=scenarios,
        hang_seconds=0.0,
    )
    exp = Experiment(
        name="no_retry",
        output_dir=tmp_dir,
        agent_config=agent_config,
        benchmark_config=benchmark,
        max_retries=0,
    )

    result = run_sequentially(exp)

    storage = FileStorage(tmp_dir)
    statuses = storage.list_episode_statuses()
    assert len(statuses) == 1
    traj_id, final = next(iter(statuses.items()))
    assert final.status == "FAILED"
    assert final.retry_count == 0
    assert _archive_count(tmp_dir, traj_id) == 0  # only one attempt was ever made
    assert traj_id in result.failures


def test_bad_model_does_not_waste_retries(tmp_dir: Path) -> None:
    """A permanent LLM provider error (e.g. typo'd model name) must not be retried.

    A permanent error like `litellm.NotFoundError` ("model does not exist") would
    fail identically on every retry. `episode.py` classifies it via
    `is_permanent_llm_error` and tags the episode with the terminal, non-retriable
    `INVALID_CONFIG` status instead of `FAILED`, so the runner stops immediately
    (retry_count=0) instead of burning `max_retries` worker slots with zero chance
    of success.
    """
    scenarios = {"task_bad_model": ["bad_model"]}
    benchmark = make_debug_benchmark(scenarios)
    agent_config = DebugAgentConfig(
        counter_dir=str(tmp_dir / "_counters"),
        scenarios=scenarios,
        hang_seconds=0.0,
    )
    exp = Experiment(
        name="bad_model_retry_waste",
        output_dir=tmp_dir,
        agent_config=agent_config,
        benchmark_config=benchmark,
        max_retries=3,
    )

    result = run_sequentially(exp)

    storage = FileStorage(tmp_dir)
    statuses = storage.list_episode_statuses()
    traj_id, final = next(iter(statuses.items()))

    assert final.status == "INVALID_CONFIG"
    assert final.error_type == "NotFoundError"
    assert final.retry_count == 0, (
        f"permanent provider error was retried {final.retry_count} times; "
        f"the harness must classify NotFoundError as non-retriable"
    )
    assert _archive_count(tmp_dir, traj_id) == 0
    assert traj_id in result.failures


def _archive_count(output_dir: Path, traj_id: str) -> int:
    base = output_dir / EPISODES_DIR
    return sum(1 for p in base.iterdir() if p.name.startswith(f"{traj_id}{ARCHIVED_MARKER}"))


def _read_archived_status(output_dir: Path, traj_id: str) -> dict | None:
    """Read status.json from any archived dir for `traj_id` (returns the first found)."""
    import json

    base = output_dir / EPISODES_DIR
    for p in base.iterdir():
        if p.name.startswith(f"{traj_id}{ARCHIVED_MARKER}"):
            status_path = p / STATUS_FILENAME
            if status_path.exists():
                return json.loads(status_path.read_text())
    return None


def _read_all_archived_statuses(output_dir: Path, traj_id: str) -> list[dict]:
    """Return every archived `status.json` dict for `traj_id`, ordered by archive timestamp."""
    import json

    base = output_dir / EPISODES_DIR
    archived = []
    for p in sorted(base.iterdir()):
        if p.name.startswith(f"{traj_id}{ARCHIVED_MARKER}"):
            status_path = p / STATUS_FILENAME
            if status_path.exists():
                archived.append(json.loads(status_path.read_text()))
    return archived


# ---------------------------------------------------------------------------
# experiment_status.json lifecycle — end-to-end
# ---------------------------------------------------------------------------

from cube_harness.analyze.xray_utils import GHOST_TIMEOUT, _promote_ghost_episodes  # noqa: E402
from cube_harness.experiment_status import EXPERIMENT_STATUS_FILENAME, ExperimentStatus, is_driver_alive  # noqa: E402


def _make_simple_exp(tmp_dir: Path, scenarios: dict[str, list[str]], name: str = "exp_status") -> Experiment:
    """Build an Experiment with the debug benchmark for a given scenario dict."""
    return Experiment(
        name=name,
        output_dir=tmp_dir,
        agent_config=DebugAgentConfig(
            counter_dir=str(tmp_dir / "_counters"),
            scenarios=scenarios,
            hang_seconds=0.0,
        ),
        benchmark_config=make_debug_benchmark(scenarios),
        max_retries=0,
    )


def test_experiment_status_sequential_happy_path(tmp_dir: Path) -> None:
    """After a successful sequential run, experiment_status.json is COMPLETED with correct counters."""
    exp = _make_simple_exp(tmp_dir, {"task_a": ["succeed"], "task_b": ["succeed"]})

    run_sequentially(exp)

    es = ExperimentStatus.read(tmp_dir / EXPERIMENT_STATUS_FILENAME)
    assert es is not None, "experiment_status.json should exist after run"
    assert es.status == "COMPLETED"
    assert es.mode == "sequential"
    assert es.total_episodes == 2
    assert es.completed == 2
    assert es.failed == 0
    assert es.ended_at is not None
    assert es.ended_at >= es.started_at
    # Driver is done — XRay must see it as not alive.
    assert is_driver_alive(es, tmp_dir, timeout_s=GHOST_TIMEOUT) is False


def test_experiment_status_sequential_interrupted(tmp_dir: Path, monkeypatch) -> None:
    """A driver-level exception leaves experiment_status.json marked INTERRUPTED."""
    exp = _make_simple_exp(tmp_dir, {"task_a": ["succeed"]})

    # Inject a failure in the retry loop so the lifecycle context manager
    # exits via exception, exercising the INTERRUPTED branch.
    from cube_harness import exp_runner

    def _boom(*_args, **_kwargs) -> None:
        raise RuntimeError("simulated driver crash")

    monkeypatch.setattr(exp_runner, "_run_sequentially_with_retries", _boom)

    with pytest.raises(RuntimeError, match="simulated driver crash"):
        run_sequentially(exp)

    es = ExperimentStatus.read(tmp_dir / EXPERIMENT_STATUS_FILENAME)
    assert es is not None
    assert es.status == "INTERRUPTED"
    assert es.ended_at is not None
    assert is_driver_alive(es, tmp_dir, timeout_s=GHOST_TIMEOUT) is False


@pytest.mark.slow
def test_experiment_status_ray_happy_path(tmp_dir: Path) -> None:
    """Ray run heartbeats experiment_status.json, ends COMPLETED with counters set."""
    exp = _make_simple_exp(tmp_dir, {"task_a": ["succeed"], "task_b": ["succeed"]})
    started_before = time.time()

    run_with_ray(exp, n_cpus=2, ray_poll_timeout=0.2, step_timeout_s=10.0, cancel_grace_s=0.5)

    es = ExperimentStatus.read(tmp_dir / EXPERIMENT_STATUS_FILENAME)
    assert es is not None
    assert es.status == "COMPLETED"
    assert es.mode == "ray"
    assert es.total_episodes == 2
    assert es.completed == 2
    assert es.failed == 0
    assert es.started_at >= started_before
    assert es.ended_at is not None and es.ended_at >= es.started_at
    # Driver finished — not alive.
    assert is_driver_alive(es, tmp_dir, timeout_s=GHOST_TIMEOUT) is False


def test_experiment_status_interrupted_promotes_queued(tmp_dir: Path, monkeypatch) -> None:
    """When the driver dies mid-run, leftover QUEUED episodes are swept to STALE by XRay.

    This is the contract XRay relies on: a missing terminal status (or an
    INTERRUPTED one) plus a stale heartbeat is the signal to clean up orphans.
    """
    from cube_harness import exp_runner

    # Set up an experiment whose pre-claim runs but never reaches episode.run().
    exp = _make_simple_exp(tmp_dir, {"task_a": ["succeed"]})

    # Allow pre-claim to write QUEUED, then crash before any episode runs.
    real_impl = exp_runner._run_sequentially_impl

    def _crash_after_preclaim(exp_arg, **kw):
        # Borrow the pre-claim path: get_episodes_to_run + _pre_claim + then bail.
        storage = FileStorage(exp_arg.output_dir)
        with exp_arg.benchmark_config.make(exp_arg.infra) as benchmark:
            episodes = exp_arg.get_episodes_to_run(benchmark, step_timeout_s=10.0, cancel_grace_s=0.5)
            for ep in episodes:
                exp_runner._pre_claim(storage, ep)
        raise RuntimeError("crash after pre-claim")

    monkeypatch.setattr(exp_runner, "_run_sequentially_impl", _crash_after_preclaim)

    with pytest.raises(RuntimeError):
        run_sequentially(exp)

    # Reset patch so we can use the runner internals freely.
    monkeypatch.setattr(exp_runner, "_run_sequentially_impl", real_impl)

    # The episode is QUEUED on disk, experiment is INTERRUPTED.
    storage = FileStorage(tmp_dir)
    statuses = storage.list_episode_statuses()
    assert len(statuses) == 1
    traj_id, ep_status = next(iter(statuses.items()))
    assert ep_status.status == "QUEUED"

    es = ExperimentStatus.read(tmp_dir / EXPERIMENT_STATUS_FILENAME)
    assert es is not None and es.status == "INTERRUPTED"

    # XRay refresh should now promote the orphan QUEUED → STALE.
    _promote_ghost_episodes(tmp_dir)
    promoted = storage.list_episode_statuses()[traj_id]
    assert promoted.status == "STALE"

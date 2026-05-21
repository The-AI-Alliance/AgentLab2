import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Self
from uuid import uuid4

from cube.benchmark import Benchmark, BenchmarkConfig
from cube.core import TypedBaseModel
from cube.resource import InfraConfig
from pydantic import Field, SerializeAsAny, model_validator

from cube_harness.agent import AgentConfig
from cube_harness.core import Trajectory
from cube_harness.episode import MAX_STEPS, Episode
from cube_harness.episode_logs import trajectory_log_id
from cube_harness.episode_status import RETRIABLE_STATUSES, EpisodeStatus, should_sweep_running_to_stale
from cube_harness.eval_log import EvalLog, ExperimentRecord
from cube_harness.storage import FileStorage

logger = logging.getLogger(__name__)

EXP_DIR = Path(os.environ.get("CH_EXP_DIR", "~/cube_harness_results")).expanduser().resolve()
_UUID_SUFFIX_RE = re.compile(r"_[0-9a-f]{8}$")


def make_experiment_output_dir(
    agent_name: str,
    benchmark_name: str,
    llm_name: str | None = None,
    tag: str | None = None,
    base_dir: Path | None = None,
) -> Path:
    """Create and return a unique output directory for an experiment run.

    Directory name: {date}_{agent_name}_{benchmark_name}[_{llm_name}][_{tag}]_{uuid8}.
    The directory is created before returning.
    """
    now = time.strftime("%Y%m%d_%H%M%S")
    parts = [now, agent_name, benchmark_name]
    if llm_name:
        parts.append(llm_name)
    if tag:
        parts.append(tag)
    parts.append(uuid4().hex[:8])
    path = (base_dir or EXP_DIR) / "_".join(parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


class ExpResult(TypedBaseModel):
    exp_id: str
    tasks_num: int
    config: dict = Field(default_factory=dict)
    trajectories: dict[str, Trajectory] = Field(default_factory=dict)
    failures: dict[str, str] = Field(default_factory=dict)


class Experiment(TypedBaseModel):
    name: str
    output_dir: Path | None = None
    agent_config: AgentConfig
    benchmark_config: SerializeAsAny[BenchmarkConfig]
    infra: SerializeAsAny[InfraConfig] | None = None
    resume: bool = False
    max_steps: int = MAX_STEPS
    max_retries: int = 3
    git_cwd: str | None = None

    @model_validator(mode="after")
    def _ensure_unique_output_dir(self) -> "Experiment":
        if self.output_dir is None:
            self.output_dir = make_experiment_output_dir(
                agent_name=self.agent_config.agent_name,
                benchmark_name=self.benchmark_config.benchmark_metadata.name,
                tag=self.name,
            )
        elif not _UUID_SUFFIX_RE.search(self.output_dir.name):
            self.output_dir = self.output_dir.parent / f"{self.output_dir.name}_{uuid4().hex[:8]}"
        return self

    @property
    def evaluation_id(self) -> str:
        return self.output_dir.name

    @property
    def config(self) -> dict:
        return self.model_dump(serialize_as_any=True)

    def get_episodes_to_run(
        self,
        benchmark: Benchmark,
        *,
        step_timeout_s: float = 1800.0,
        cancel_grace_s: float = 120.0,
        orphan_threshold_s: float = 3600.0,
        process_start_s: float | None = None,
    ) -> list[Episode]:
        """Return episodes to run based on `resume`.

        ``benchmark`` is the live ``Benchmark`` returned by
        ``self.benchmark_config.make(self.infra)``; the runner is responsible
        for the make/close lifecycle and passes the live instance in so
        episodes can pick up its ``_runtime_context``.

        Decisions are driven by `status.json` per episode (no trajectory deserialisation).

        - `resume=False`: create all episodes from scratch (any prior data is archived
          per-attempt by pre-claim / save_trajectory).
        - `resume=True`: pick up everything that's not COMPLETED (or MAX_STEPS_REACHED) —
          unstarted (no status.json) plus retriable failures (FAILED / STALE /
          CANCELLED), gated by `retry_count < max_retries`. Stale `RUNNING` entries are
          swept to `STALE` first so they become eligible.
        """
        if not self.resume:
            return self._create_all_episodes(benchmark)

        storage = FileStorage(self.output_dir)
        config_files = storage.list_episode_configs()
        if not config_files:
            logger.warning(f"No episode configs found in {self.output_dir}, creating from scratch")
            return self._create_all_episodes(benchmark)

        # Sweep stale RUNNING entries first so they show up as STALE in retry selection.
        sweep_stale_statuses(
            storage,
            step_timeout_s=step_timeout_s,
            cancel_grace_s=cancel_grace_s,
            orphan_threshold_s=orphan_threshold_s,
            process_start_s=process_start_s,
        )

        statuses = storage.list_episode_statuses()
        episodes: list[Episode] = []

        for config_file in config_files:
            trajectory_id = self._trajectory_id_from_config(config_file)
            if trajectory_id is None:
                logger.warning(f"Could not parse task_id from config filename: {config_file.name}")
                continue
            status = statuses.get(trajectory_id)
            if not self._should_relaunch(status):
                continue
            try:
                episode = Episode.load_episode_from_config(config_file, benchmark)
            except Exception:
                logger.exception(f"Failed to load episode config {config_file}")
                continue
            # Existing trajectory (if any) will be archived on the next save_trajectory.
            episode.allow_overwrite = status is not None
            episodes.append(episode)

        logger.info(f"Selected {len(episodes)} episode(s) to run (out of {len(config_files)} total) with resume=True")
        return episodes

    def _should_relaunch(self, status: EpisodeStatus | None) -> bool:
        """True iff `resume=True` should re-run this episode (status-driven)."""
        if status is None:
            return True  # never started → run it
        if status.status not in RETRIABLE_STATUSES:
            # COMPLETED / MAX_STEPS_REACHED / in-flight (QUEUED, RUNNING) → leave alone
            return False
        return status.retry_count < self.max_retries

    def _create_all_episodes(self, benchmark: Benchmark) -> list[Episode]:
        """Create all episodes from scratch and save their configs to disk."""
        task_configs = list(self.benchmark_config.get_task_configs())
        runtime_context = benchmark._runtime_context
        episodes = [
            Episode(
                id=i,
                output_dir=self.output_dir,
                agent_config=self.agent_config,
                task_config=tc,
                exp_name=self.name,
                max_steps=self.max_steps,
                runtime_context=runtime_context,
                storage=None,
            )
            for i, tc in enumerate(task_configs)
        ]
        for episode in episodes:
            episode.storage.save_episode_config(episode.config)
        logger.info(f"Prepared {len(episodes)} episodes for experiment '{self.name}'")
        return episodes

    def save_config(self) -> None:
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        config_path = output_path / "experiment_config.json"
        with open(config_path, "w") as f:
            f.write(self.model_dump_json(indent=2, serialize_as_any=True))
        logger.info(f"Saved experiment config to {config_path}")
        exp_record = ExperimentRecord.from_experiment(
            exp_name=self.name,
            output_dir=self.output_dir,
            agent_config=self.agent_config,
            benchmark_config=self.benchmark_config,
            git_cwd=self.git_cwd,
        )
        exp_record.write(output_path)

    @classmethod
    def load_config(cls, path: str) -> Self:
        """Load experiment from a JSON config file."""
        with open(path) as f:
            data = json.load(f)
        return cls.model_validate(data)

    def _trajectory_id_from_config(self, config_file: Path) -> str | None:
        """Return trajectory_id for a config file, handling both V2 and V1 filename formats."""
        if config_file.name == "episode_config.json":
            return config_file.parent.name
        parsed = self._parse_episode_config_filename(config_file)
        if parsed:
            episode_id, task_id = parsed
            return trajectory_log_id(task_id, episode_id)
        return None

    def _parse_episode_config_filename(self, config_file: Path) -> tuple[int, str] | None:
        """
        Parse episode config filename to extract episode id and task_id.

        Args:
            config_file: Path to the episode config file (e.g., episode_0_task_my_task.json)

        Returns:
            Tuple of (episode_id, task_id) if parsing succeeds, None otherwise.
        """
        parts = config_file.stem.split("_task_", 1)
        if len(parts) == 2:
            try:
                episode_id_str = parts[0].replace("episode_", "")
                episode_id = int(episode_id_str)
                task_id = parts[1]
                return (episode_id, task_id)
            except (ValueError, AttributeError):
                return None
        return None

    def print_stats(self, results: ExpResult) -> None:
        if not results.trajectories:
            logger.info("No trajectories to compute stats")
            return

        total_steps = sum(len(trajectory.steps) for trajectory in results.trajectories.values())
        avg_steps = total_steps / len(results.trajectories)

        rewards = []
        for traj in results.trajectories.values():
            rewards.append(traj.last_env_step().reward)

        accuracy = sum(rewards) / len(rewards) if rewards else 0.0

        logger.info(f"Experiment '{self.name}' stats:")
        logger.info(f"  Total trajectories: {len(results.trajectories)}")
        logger.info(f"  Avg steps per trajectory: {avg_steps:.2f}")
        logger.info(f"  Accuracy (avg. final reward): {accuracy:.4f}")
        logger.info(f"  Failed tasks: {len(results.failures)}")
        logger.info(f"Saved to: {self.output_dir}")

    def export_eval_log(self, output_dir: Path | None = None) -> EvalLog:
        """Load the EvalLog from per-trajectory records written during the run.

        experiment_record.json is written at experiment start (save_config).
        episode_record.json files are written by the runner after each episode.
        This method simply reads those files and packages them.

        Args:
            output_dir: Directory to read from. Defaults to self.output_dir.

        Returns:
            EvalLog with ExperimentRecord and one EpisodeRecord per completed episode.
        """
        resolved_dir = output_dir if output_dir is not None else self.output_dir
        return EvalLog.load(resolved_dir)


def sweep_stale_statuses(
    storage: FileStorage,
    *,
    step_timeout_s: float,
    cancel_grace_s: float,
    orphan_threshold_s: float,
    process_start_s: float | None = None,
) -> list[str]:
    """Mark in-flight episodes whose worker is dead as `STALE`.

    Three cases:
    - `RUNNING` with `last_heartbeat_at` predating `process_start_s` (worker is from a
      prior, now-dead process — force-sweep regardless of heartbeat age). Fixes the
      kill-and-retry race where a freshly killed worker has a recent heartbeat that
      would otherwise pass the age threshold.
    - `RUNNING` with `last_heartbeat_at` older than `step_timeout_s + cancel_grace_s`
      (worker died mid-episode without writing a terminal status).
    - `QUEUED` with `started_at` older than `orphan_threshold_s` (driver pre-claimed
      but Ray never picked the episode up — typically because the prior driver
      crashed before the worker started).

    Returns the list of trajectory_ids that were swept.
    """
    now = time.time()
    swept: list[str] = []
    for trajectory_id, status in storage.list_episode_statuses().items():
        is_stale = False
        if status.status == "RUNNING":
            is_stale = should_sweep_running_to_stale(
                status,
                now=now,
                step_timeout_s=step_timeout_s,
                cancel_grace_s=cancel_grace_s,
                process_start_s=process_start_s,
            )
        elif status.status == "QUEUED":
            if now - status.started_at > orphan_threshold_s:
                is_stale = True
        if not is_stale:
            continue
        prior_state = status.status
        status.status = "STALE"
        status.ended_at = now
        status.error_type = status.error_type or "WorkerDied"
        status.error_message = status.error_message or (
            "Worker died without writing terminal status (no fresh heartbeat)"
        )
        storage.write_episode_status(trajectory_id, status)
        swept.append(trajectory_id)
        logger.warning(f"Swept stale {prior_state} -> STALE for {trajectory_id}")
    return swept

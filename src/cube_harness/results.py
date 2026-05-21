import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from cube_harness.core import Trajectory, TrajectoryStep
from cube_harness.episode_status import STATUS_FILENAME
from cube_harness.episode_status import EpisodeStatus as RawEpisodeStatus
from cube_harness.storage import (
    ARCHIVED_MARKER,
    EPISODE_METADATA,
    EPISODES_DIR,
    STEPS_DIR,
    FileStorage,
    _read_step_file,
)
from cube_harness.summary import EpisodeStatus, ExperimentSummary, StepSummary

if TYPE_CHECKING:
    from cube_harness.episode import EpisodeConfig


class EpisodeRecord(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")

    trajectory_id: str
    status: EpisodeStatus
    n_env_steps: int = 0
    n_agent_steps: int = 0
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reward: float = 0.0


class EpisodeResult:
    def __init__(self, episode_dir: Path, storage: FileStorage) -> None:
        self._dir = episode_dir
        self._storage = storage
        self._metadata: Trajectory | None = None
        self._steps: dict[int, TrajectoryStep] = {}
        self._traj_id: str | None = None
        self._summary: list[StepSummary] | None = None

    def trajectory_id(self) -> str:
        if self._traj_id is None:
            self._traj_id = self.metadata().id
        return self._traj_id

    def metadata(self) -> Trajectory:
        if self._metadata is None:
            with open(self._dir / EPISODE_METADATA) as f:
                data = json.load(f)
            data["steps"] = []
            self._metadata = Trajectory.model_validate(data)
        return self._metadata

    def config(self) -> "EpisodeConfig":
        from cube_harness.episode import EpisodeConfig

        config_path = self._dir / "episode_config.json"
        return EpisodeConfig.model_validate_json(config_path.read_text())

    def summary_stats(self) -> dict[str, Any] | None:
        return self.metadata().summary_stats

    def summary(self) -> list[StepSummary]:
        if self._summary is None:
            path = self._dir / "episode_summary.jsonl"
            if not path.exists():
                self._summary = []
            else:
                self._summary = [
                    StepSummary.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()
                ]
        return self._summary

    def status(self) -> EpisodeStatus:
        path = self._dir / "episode_summary.jsonl"
        if not path.exists():
            return EpisodeStatus.PENDING
        last_line = None
        for line in path.read_text().splitlines():
            if line.strip():
                last_line = line
        if last_line is None:
            return EpisodeStatus.PENDING
        return StepSummary.model_validate_json(last_line).status

    def n_turns(self) -> int:
        steps_dir = self._dir / STEPS_DIR
        if not steps_dir.exists():
            return 0
        return sum(1 for f in steps_dir.iterdir() if "_obs." in f.name)

    def __len__(self) -> int:
        steps_dir = self._dir / STEPS_DIR
        if not steps_dir.exists():
            return 0
        return sum(1 for _ in steps_dir.iterdir())

    def __getitem__(self, index: int) -> TrajectoryStep:
        if index not in self._steps:
            self._steps[index] = self._storage.load_step(self.trajectory_id(), index)
        return self._steps[index]

    def __iter__(self) -> Iterator[TrajectoryStep]:
        for i in range(len(self)):
            yield self[i]

    def get_obs(self, turn: int) -> TrajectoryStep:
        return self._load_step_by_suffix(turn, "obs")

    def get_act(self, turn: int) -> TrajectoryStep:
        return self._load_step_by_suffix(turn, "act")

    def _load_step_by_suffix(self, turn: int, suffix: str) -> TrajectoryStep:
        steps_dir = self._dir / STEPS_DIR
        prefix = f"{turn:03d}_{suffix}."
        for f in steps_dir.iterdir():
            if f.name.startswith(prefix):
                data = _read_step_file(f)
                assert data is not None
                return TrajectoryStep.model_validate(data)
        raise FileNotFoundError(f"Step {prefix}* not found in {steps_dir}")

    def load_full(self) -> Trajectory:
        return self._storage.load_trajectory(self.trajectory_id())

    def get_exp_record(self) -> EpisodeRecord:
        meta = self.metadata()
        stats = meta.summary_stats or {}
        known_fields = EpisodeRecord.model_fields
        return EpisodeRecord(
            trajectory_id=self.trajectory_id(),
            status=self.status(),
            **{k: v for k, v in stats.items() if k in known_fields},
            **meta.metadata,
        )


class ExperimentResult:
    def __init__(self, exp_dir: str | Path) -> None:
        self._dir = Path(exp_dir)
        self._storage = FileStorage(self._dir)
        self._episodes: dict[str, EpisodeResult] | None = None

    def __iter__(self) -> Iterator[EpisodeResult]:
        return iter(self.episodes().values())

    def episodes(self) -> dict[str, EpisodeResult]:
        if self._episodes is None:
            self._episodes = {}
            episodes_dir = self._dir / EPISODES_DIR
            if episodes_dir.exists():
                for ep_dir in sorted(episodes_dir.iterdir()):
                    if ep_dir.is_dir() and ARCHIVED_MARKER not in ep_dir.name:
                        if (ep_dir / EPISODE_METADATA).exists():
                            self._episodes[ep_dir.name] = EpisodeResult(ep_dir, self._storage)
        return self._episodes

    def iter_episode_statuses(self) -> Iterator[RawEpisodeStatus]:
        """Yield typed :class:`cube_harness.episode_status.EpisodeStatus` for every
        non-archived episode dir that has a ``status.json`` file.

        Unlike :meth:`episodes` (which requires the finalized ``episode.metadata.json``),
        this also surfaces in-flight (``QUEUED``/``RUNNING``) episodes — used by
        ``scripts/experiments_report.py`` and the XRay viewer's per-experiment row computation
        so both tools see the same episode set. Deduplicated by ``(task_id, episode_id)``.
        """
        episodes_dir = self._dir / EPISODES_DIR
        if not episodes_dir.exists():
            return
        seen: set[tuple[str, int]] = set()
        for ep_dir in sorted(episodes_dir.iterdir()):
            if ARCHIVED_MARKER in ep_dir.name or not ep_dir.is_dir():
                continue
            es = RawEpisodeStatus.read(ep_dir / STATUS_FILENAME)
            if es is None:
                continue
            key = (es.task_id, es.episode_id)
            if key in seen:
                continue
            seen.add(key)
            yield es

    def summary(self) -> ExperimentSummary | None:
        path = self._dir / "experiment_summary.json"
        if path.exists():
            return ExperimentSummary.model_validate_json(path.read_text())
        return None

    def iter_records(self) -> Iterator[EpisodeRecord]:
        for ep in self.episodes().values():
            yield ep.get_exp_record()

    def get_records(self) -> list[EpisodeRecord]:
        return list(self.iter_records())

    def to_df(self) -> Any:
        from cube_harness.analyze.inspect_results import trajectories_to_df

        trajs = self._storage.load_all_trajectory_metadata()
        return trajectories_to_df(trajs)

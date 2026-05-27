import fcntl
import json
import logging
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import msgpack
import zstandard
from cube.core import EnvironmentOutput
from pydantic import BaseModel

from cube_harness.core import AgentOutput, Trajectory, TrajectoryStep
from cube_harness.episode_logs import get_log_path as get_episode_log_path
from cube_harness.episode_logs import trajectory_log_id
from cube_harness.episode_status import STATUS_FILENAME, EpisodeStatus

if TYPE_CHECKING:
    from cube_harness.episode import EpisodeConfig

logger = logging.getLogger(__name__)

EPISODES_DIR = "episodes"
TRAJECTORIES_DIR = "trajectories"
EPISODE_METADATA = "episode.metadata.json"
STEPS_DIR = "steps"
ARCHIVED_MARKER = ".archived_"


class LLMCallRef(BaseModel):
    llm_call_id: str


class Storage(Protocol):
    def save_trajectory(self, trajectory: Trajectory, allow_overwrite: bool = False) -> None: ...

    def save_step(self, step: TrajectoryStep, trajectory_id: str, step_num: int) -> None: ...

    def save_episode_config(self, episode_config: "EpisodeConfig") -> None: ...

    def update_experiment_summary(self, trajectory: Trajectory) -> None: ...

    def write_episode_status(self, trajectory_id: str, status: EpisodeStatus) -> None: ...

    def read_episode_status(self, trajectory_id: str) -> EpisodeStatus | None: ...

    def archive_episode(self, trajectory_id: str) -> None: ...


_thread_local = threading.local()


def _get_compressor() -> zstandard.ZstdCompressor:
    if not hasattr(_thread_local, "compressor"):
        _thread_local.compressor = zstandard.ZstdCompressor(level=3)
    return _thread_local.compressor


def _get_decompressor() -> zstandard.ZstdDecompressor:
    if not hasattr(_thread_local, "decompressor"):
        _thread_local.decompressor = zstandard.ZstdDecompressor()
    return _thread_local.decompressor


def _serialize_step(step: TrajectoryStep) -> bytes:
    data = json.loads(step.model_dump_json())
    packed = msgpack.packb(data, use_bin_type=True)
    return _get_compressor().compress(packed)


def _deserialize_step(raw: bytes) -> dict:
    decompressed = _get_decompressor().decompress(raw)
    return msgpack.unpackb(decompressed, raw=False)


def _step_filename(step_num: int, step: TrajectoryStep) -> str:
    suffix = "obs" if isinstance(step.output, EnvironmentOutput) else "act"
    return f"{step_num:03d}_{suffix}.msgpack.zst"


def _read_step_file(path: Path) -> dict | None:
    if path.name.endswith(".msgpack.zst"):
        return _deserialize_step(path.read_bytes())
    if path.suffix == ".json":
        with open(path) as f:
            return json.loads(f.read())
    return None


def _resolve_llm_call_file(output_dir: Path, step_id: str, llm_call_id: str) -> Path:
    flat = output_dir / f"{step_id}_{llm_call_id}.json"
    if flat.exists():
        return flat
    return output_dir / "llm_calls" / f"{step_id}_{llm_call_id}.json"


class FileStorage:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self._saved_ids: set[str] = set()

    # --- V2 episode directory helpers ---

    def _episode_dir(self, trajectory_id: str) -> Path:
        return self.output_dir / EPISODES_DIR / trajectory_id

    def _episode_dirs(self) -> Iterator[Path]:
        episodes_dir = self.output_dir / EPISODES_DIR
        if not episodes_dir.exists():
            return
        for ep_dir in episodes_dir.iterdir():
            if ep_dir.is_dir() and ARCHIVED_MARKER not in ep_dir.name and (ep_dir / EPISODE_METADATA).exists():
                yield ep_dir

    # --- V1 trajectory file helpers (flat output_dir + legacy trajectories/) ---

    def _v1_metadata_files(self) -> Iterator[Path]:
        seen: set[str] = set()
        for search_dir in (self.output_dir, self.output_dir / TRAJECTORIES_DIR):
            if not search_dir.exists():
                continue
            for f in search_dir.glob("*.metadata.json"):
                if ARCHIVED_MARKER not in f.name:
                    tid = f.stem.replace(".metadata", "")
                    if tid not in seen:
                        seen.add(tid)
                        yield f

    @staticmethod
    def _v1_traj_id_from_file(metadata_file: Path) -> str:
        return metadata_file.stem.replace(".metadata", "")

    def _v1_resolve_trajectory_paths(self, trajectory_id: str) -> tuple[Path, Path]:
        meta = self.output_dir / f"{trajectory_id}.metadata.json"
        jsonl = self.output_dir / f"{trajectory_id}.jsonl"
        if not meta.exists():
            legacy_meta = self.output_dir / TRAJECTORIES_DIR / f"{trajectory_id}.metadata.json"
            if legacy_meta.exists():
                return legacy_meta, self.output_dir / TRAJECTORIES_DIR / f"{trajectory_id}.jsonl"
        return meta, jsonl

    # --- Write (always V2) ---

    def save_trajectory(self, trajectory: Trajectory, allow_overwrite: bool = False) -> None:
        ep_dir = self._episode_dir(trajectory.id)
        metadata_path = ep_dir / EPISODE_METADATA
        is_resave = trajectory.id in self._saved_ids

        if not is_resave and ep_dir.exists() and metadata_path.exists():
            if not allow_overwrite:
                raise FileExistsError(
                    f"Trajectory '{trajectory.id}' already exists at {ep_dir}. "
                    "Use allow_overwrite=True to archive the old trajectory and overwrite."
                )
            self._archive_episode(ep_dir)

        ep_dir.mkdir(parents=True, exist_ok=True)
        (ep_dir / STEPS_DIR).mkdir(exist_ok=True)
        self._saved_ids.add(trajectory.id)

        trajectory_data = trajectory.model_dump(exclude={"steps"}, mode="json")
        with open(metadata_path, "w") as f:
            f.write(json.dumps(trajectory_data, indent=2))

        for i, step in enumerate(trajectory.steps):
            self._write_step(ep_dir, i, step)

        logger.info(f"Saved trajectory to {ep_dir}")

    def _archive_episode(self, ep_dir: Path) -> None:
        archived = ep_dir.parent / f"{ep_dir.name}{ARCHIVED_MARKER}{time.time()}"
        # Preserve episode_config.json so the retry pipeline still discovers this
        # episode via _episode_config_dirs(). The config is written once at
        # experiment prep time (Experiment.prepare_episodes) and is not re-saved
        # on retry, so the archive must not destroy it.
        config_src = ep_dir / "episode_config.json"
        config_bytes = config_src.read_bytes() if config_src.exists() else None
        ep_dir.rename(archived)
        if config_bytes is not None:
            ep_dir.mkdir()
            (ep_dir / "episode_config.json").write_bytes(config_bytes)
        logger.info(f"Archived {ep_dir.name} -> {archived.name}")

    def archive_episode(self, trajectory_id: str) -> None:
        """Archive the episode directory for a terminal attempt, preserving its history."""
        ep_dir = self._episode_dir(trajectory_id)
        if ep_dir.exists():
            self._archive_episode(ep_dir)

    def save_step(self, step: TrajectoryStep, trajectory_id: str, step_num: int) -> None:
        ep_dir = self._episode_dir(trajectory_id)
        if not ep_dir.exists():
            raise ValueError(f"Episode directory does not exist: {ep_dir}. Call save_trajectory first.")
        try:
            self._write_step(ep_dir, step_num, step)
        except Exception as e:
            logger.exception(f"Error saving step to trajectory {trajectory_id}: {e}")
            raise e

    def _write_step(self, ep_dir: Path, step_num: int, step: TrajectoryStep) -> None:
        filename = _step_filename(step_num, step)
        step_path = ep_dir / STEPS_DIR / filename
        step_path.write_bytes(_serialize_step(step))

    # --- Load single trajectory ---

    def load_trajectory(self, trajectory_id: str) -> Trajectory:
        ep_dir = self._episode_dir(trajectory_id)
        if (ep_dir / EPISODE_METADATA).exists():
            return self._load_trajectory(ep_dir, trajectory_id)
        return self._v1_load_trajectory(trajectory_id)

    def _maybe_inject_failure_text(self, ep_dir: Path, trajectory_data: dict) -> None:
        """Inject _failure_text into metadata if failure.txt exists and trajectory has no end_time."""
        if trajectory_data.get("end_time") is not None:
            return
        failure_path = ep_dir / "failure.txt"
        if failure_path.exists():
            trajectory_data.setdefault("metadata", {})["_failure_text"] = failure_path.read_text()

    def _maybe_inject_episode_status(self, ep_dir: Path, trajectory_data: dict) -> None:
        """Inject episode status fields from status.json into trajectory metadata.

        Adds _episode_status, _retry_count, _error_type, _error_message so that
        xray_utils.trajectory_status() can use the authoritative status.json rather
        than falling back to the legacy heuristic.
        """
        status = EpisodeStatus.read(ep_dir / STATUS_FILENAME)
        if status is None:
            return
        trajectory_data.setdefault("metadata", {}).update(
            {
                "_episode_status": status.status,
                "_retry_count": status.retry_count,
                "_error_type": status.error_type,
                "_error_message": status.error_message,
            }
        )

    def _load_trajectory(self, ep_dir: Path, trajectory_id: str) -> Trajectory:
        with open(ep_dir / EPISODE_METADATA) as f:
            trajectory_data = json.load(f)

        self._maybe_inject_failure_text(ep_dir, trajectory_data)
        self._maybe_inject_episode_status(ep_dir, trajectory_data)

        steps: list[TrajectoryStep] = []
        steps_dir = ep_dir / STEPS_DIR
        if steps_dir.exists():
            for step_file in sorted(steps_dir.iterdir()):
                step_data = _read_step_file(step_file)
                if step_data is not None:
                    steps.append(TrajectoryStep.model_validate(step_data))

        trajectory_data["steps"] = steps
        return Trajectory.model_validate(trajectory_data)

    def load_step(self, trajectory_id: str, step_index: int) -> TrajectoryStep:
        ep_dir = self._episode_dir(trajectory_id)
        if not ep_dir.exists():
            raise FileNotFoundError(f"Episode directory not found for trajectory: {trajectory_id}")
        steps_dir = ep_dir / STEPS_DIR
        for suffix in ("obs", "act"):
            path = steps_dir / f"{step_index:03d}_{suffix}.msgpack.zst"
            if path.exists():
                return TrajectoryStep.model_validate(_deserialize_step(path.read_bytes()))
        raise IndexError(f"Step {step_index} not found in {steps_dir}")

    def _v1_load_trajectory(self, trajectory_id: str) -> Trajectory:
        metadata_path, steps_path = self._v1_resolve_trajectory_paths(trajectory_id)

        if not metadata_path.exists():
            raise FileNotFoundError(f"Trajectory metadata not found: {metadata_path}")

        with open(metadata_path) as f:
            trajectory_data = json.load(f)

        if "metadata" not in trajectory_data:
            trajectory_data = {"id": trajectory_id, "metadata": trajectory_data}

        steps: list[TrajectoryStep] = []
        if steps_path.exists():
            with open(steps_path) as f:
                for i, line in enumerate(f):
                    if line.strip():
                        step_data = json.loads(line)
                        step_data = self._v1_resolve_llm_call_refs(step_data, trajectory_id, i)
                        if "output" not in step_data and ("obs" in step_data or "actions" in step_data):
                            step_data = {"output": step_data}
                        steps.append(TrajectoryStep.model_validate(step_data))

        trajectory_data["steps"] = steps
        return Trajectory.model_validate(trajectory_data)

    def _v1_resolve_llm_call_refs(self, step_data: dict, trajectory_id: str, step_num: int) -> dict:
        output = step_data.get("output", {})
        llm_calls = output.get("llm_calls", [])
        if not llm_calls:
            return step_data

        step_id = f"{trajectory_id}_step{step_num:03d}"
        resolved_calls = []
        for ref in llm_calls:
            if llm_call_id := ref.get("llm_call_id", None):
                call_path = _resolve_llm_call_file(self.output_dir, step_id, llm_call_id)
                if not call_path.exists():
                    raise FileNotFoundError(f"LLM call file not found: {call_path}")
                with open(call_path) as f:
                    resolved_calls.append(json.load(f))
            else:
                raise ValueError(f"Invalid LLM call reference format {ref}")

        step_data["output"]["llm_calls"] = resolved_calls
        return step_data

    # --- Load metadata (no steps) ---

    def load_trajectory_metadata(self, trajectory_id: str) -> Trajectory:
        ep_dir = self._episode_dir(trajectory_id)
        metadata_path = ep_dir / EPISODE_METADATA
        if not metadata_path.exists():
            metadata_path, _ = self._v1_resolve_trajectory_paths(trajectory_id)

        if not metadata_path.exists():
            raise FileNotFoundError(f"Trajectory metadata not found: {metadata_path}")

        with open(metadata_path) as f:
            trajectory_data = json.load(f)

        if "metadata" not in trajectory_data:
            trajectory_data = {"id": trajectory_id, "metadata": trajectory_data}

        if (ep_dir / EPISODE_METADATA).exists():
            self._maybe_inject_failure_text(ep_dir, trajectory_data)
            self._maybe_inject_episode_status(ep_dir, trajectory_data)

        trajectory_data["steps"] = []
        return Trajectory.model_validate(trajectory_data)

    # --- Bulk listing ---

    def load_all_trajectory_metadata(self) -> list[Trajectory]:
        return self._load_all_metadata() + self._v1_load_all_metadata()

    def _load_all_metadata(self) -> list[Trajectory]:
        results: list[Trajectory] = []
        for ep_dir in self._episode_dirs():
            try:
                with open(ep_dir / EPISODE_METADATA) as f:
                    data = json.load(f)
                self._maybe_inject_failure_text(ep_dir, data)
                self._maybe_inject_episode_status(ep_dir, data)
                data["steps"] = []
                results.append(Trajectory.model_validate(data))
            except Exception as e:
                logger.error(f"Failed to load episode metadata {ep_dir.name}: {e}")
        return results

    def _v1_load_all_metadata(self) -> list[Trajectory]:
        results: list[Trajectory] = []
        for metadata_file in self._v1_metadata_files():
            trajectory_id = self._v1_traj_id_from_file(metadata_file)
            try:
                results.append(self.load_trajectory_metadata(trajectory_id))
            except Exception as e:
                logger.error(f"Failed to load trajectory metadata {trajectory_id}: {e}")
        return results

    def list_trajectory_ids(self) -> list[str]:
        return self._list_ids() + self._v1_list_ids()

    def _list_ids(self) -> list[str]:
        return [ep_dir.name for ep_dir in self._episode_dirs()]

    def _v1_list_ids(self) -> list[str]:
        return [self._v1_traj_id_from_file(f) for f in self._v1_metadata_files()]

    def list_trajectory_ids_with_mtime(self) -> dict[str, float]:
        result = self._list_ids_with_mtime()
        result.update(self._v1_list_ids_with_mtime())
        return result

    def _list_ids_with_mtime(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for ep_dir in self._episode_dirs():
            traj_id = ep_dir.name
            summary_path = ep_dir / "episode_summary.jsonl"
            mtime = (
                summary_path.stat().st_mtime if summary_path.exists() else (ep_dir / EPISODE_METADATA).stat().st_mtime
            )
            failure_path = ep_dir / "failure.txt"
            if failure_path.exists():
                mtime = max(mtime, failure_path.stat().st_mtime)
            result[traj_id] = mtime
        return result

    def _v1_list_ids_with_mtime(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for metadata_file in self._v1_metadata_files():
            traj_id = self._v1_traj_id_from_file(metadata_file)
            mtime = metadata_file.stat().st_mtime
            jsonl_path = metadata_file.parent / f"{traj_id}.jsonl"
            if jsonl_path.exists():
                mtime = max(mtime, jsonl_path.stat().st_mtime)
            result[traj_id] = mtime
        return result

    def load_all_trajectories(self, exp_dir: str | Path | None = None) -> list[Trajectory]:
        if exp_dir is not None:
            return FileStorage(exp_dir).load_all_trajectories()
        return self._load_all_trajectories() + self._v1_load_all_trajectories()

    def _load_all_trajectories(self) -> list[Trajectory]:
        results: list[Trajectory] = []
        for ep_dir in self._episode_dirs():
            try:
                results.append(self._load_trajectory(ep_dir, ep_dir.name))
            except Exception as e:
                logger.error(f"Failed to load episode {ep_dir.name}: {e}")
        return results

    def _v1_load_all_trajectories(self) -> list[Trajectory]:
        results: list[Trajectory] = []
        for metadata_file in self._v1_metadata_files():
            trajectory_id = self._v1_traj_id_from_file(metadata_file)
            try:
                results.append(self._v1_load_trajectory(trajectory_id))
            except Exception as e:
                logger.error(f"Failed to load trajectory {trajectory_id}: {e}")
        return results

    # --- Logs ---

    def get_log_path(self, trajectory_id: str) -> Path:
        return get_episode_log_path(self.output_dir, trajectory_id)

    def load_logs(self, trajectory_id: str) -> str:
        log_path = self.get_log_path(trajectory_id)
        if not log_path.exists():
            legacy_log_path = self.output_dir / "logs" / f"{trajectory_id}.log"
            if not legacy_log_path.exists():
                return ""
            log_path = legacy_log_path
        return log_path.read_text()

    def has_logs(self, trajectory_id: str) -> bool:
        log_path = self.get_log_path(trajectory_id)
        legacy_log_path = self.output_dir / "logs" / f"{trajectory_id}.log"
        return log_path.exists() or legacy_log_path.exists()

    # --- Experiment summary ---

    def update_experiment_summary(self, trajectory: Trajectory) -> None:
        from cube_harness.summary import ExperimentSummary

        self.output_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.output_dir / "experiment_summary.lock"
        summary_path = self.output_dir / "experiment_summary.json"

        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                if summary_path.exists():
                    summary = ExperimentSummary.model_validate_json(summary_path.read_text())
                else:
                    summary = ExperimentSummary()

                stats = trajectory.summary_stats or {}
                has_error = any(
                    hasattr(step.output, "error") and step.output.error is not None
                    for step in trajectory.steps
                    if isinstance(step.output, AgentOutput)
                )

                summary.n_episodes += 1
                if has_error:
                    summary.n_errored += 1
                else:
                    summary.n_completed += 1
                summary.total_reward += stats.get("final_reward", 0.0)
                summary.total_prompt_tokens += stats.get("prompt_tokens", 0)
                summary.total_completion_tokens += stats.get("completion_tokens", 0)
                summary.total_cost += stats.get("cost", 0.0)

                if summary.n_completed > 0:
                    summary.avg_reward = round(summary.total_reward / summary.n_completed, 4)
                summary.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

                tmp_path = summary_path.with_suffix(".tmp")
                tmp_path.write_text(summary.model_dump_json(indent=2))
                tmp_path.rename(summary_path)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    # --- Episode configs ---

    def save_failure(self, trajectory_id: str, stack_trace: str) -> None:
        """Persist a failure stack trace for an episode that could not produce a trajectory."""
        ep_dir = self._episode_dir(trajectory_id)
        ep_dir.mkdir(parents=True, exist_ok=True)
        (ep_dir / "failure.txt").write_text(stack_trace)
        logger.info(f"Saved failure for {trajectory_id} to {ep_dir / 'failure.txt'}")

    def load_missing_trajectory_stubs(self) -> list[Trajectory]:
        """Return stub Trajectories for episodes with a config but no trajectory data.

        These represent tasks that were planned (episode_config.json saved upfront) but
        never produced a trajectory — either because they crashed during setup or never ran.
        The stubs have ``_missing=True`` in metadata so xray can display them distinctly.
        A ``_failure_text`` key is also added when a failure.txt file exists.
        """
        existing_ids = set(self.list_trajectory_ids())
        stubs: list[Trajectory] = []
        for ep_dir in self._episode_config_dirs():
            traj_id = ep_dir.name
            if traj_id in existing_ids:
                continue
            config_path = ep_dir / "episode_config.json"
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                task_id = cfg.get("task_id", traj_id)
                metadata: dict = {"task_id": task_id, "_missing": True}
                failure_path = ep_dir / "failure.txt"
                if failure_path.exists():
                    metadata["_failure_text"] = failure_path.read_text()
                stub_data: dict = {"metadata": metadata}
                self._maybe_inject_episode_status(ep_dir, stub_data)
                stubs.append(Trajectory(id=traj_id, metadata=metadata))
            except Exception:
                logger.debug(f"Could not read episode config for missing stub: {ep_dir}")
        return stubs

    def save_episode_config(self, episode_config: "EpisodeConfig") -> None:
        traj_id = trajectory_log_id(episode_config.task_config.task_id, episode_config.id)
        ep_dir = self._episode_dir(traj_id)
        ep_dir.mkdir(parents=True, exist_ok=True)
        config_path = ep_dir / "episode_config.json"
        with open(config_path, "w") as f:
            f.write(episode_config.model_dump_json(indent=2, serialize_as_any=True))
        logger.info(f"Saved episode config to {config_path}")

    def load_episode_config(self, config_path: Path) -> "EpisodeConfig":
        from cube_harness.episode import EpisodeConfig

        with open(config_path) as f:
            data = json.load(f)

        return EpisodeConfig.model_validate(data)

    def _episode_config_dirs(self) -> Iterator[Path]:
        """Yield all non-archived episode dirs that have episode_config.json (planned or run)."""
        episodes_dir = self.output_dir / EPISODES_DIR
        if not episodes_dir.exists():
            return
        for ep_dir in episodes_dir.iterdir():
            if ep_dir.is_dir() and ARCHIVED_MARKER not in ep_dir.name and (ep_dir / "episode_config.json").exists():
                yield ep_dir

    def list_episode_configs(self) -> list[Path]:
        v2_configs = [ep_dir / "episode_config.json" for ep_dir in self._episode_config_dirs()]
        v1_config_dir = self.output_dir / "episode_configs"
        v1_configs = list(v1_config_dir.glob("episode_*_task_*.json")) if v1_config_dir.exists() else []
        return v2_configs + v1_configs

    # --- Episode status (control plane) ---

    def _episode_status_path(self, trajectory_id: str) -> Path:
        return self._episode_dir(trajectory_id) / STATUS_FILENAME

    def write_episode_status(self, trajectory_id: str, status: EpisodeStatus) -> None:
        status.write(self._episode_status_path(trajectory_id))

    def read_episode_status(self, trajectory_id: str) -> EpisodeStatus | None:
        return EpisodeStatus.read(self._episode_status_path(trajectory_id))

    def list_episode_statuses(self) -> dict[str, EpisodeStatus]:
        """Return {trajectory_id: status} for every non-archived episode dir with a status.json."""
        result: dict[str, EpisodeStatus] = {}
        for ep_dir in self._episode_config_dirs():
            status = EpisodeStatus.read(ep_dir / STATUS_FILENAME)
            if status is not None:
                result[ep_dir.name] = status
        return result

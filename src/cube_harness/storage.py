import fcntl
import itertools
import json
import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import msgpack
import zstandard
from cube.core import EnvironmentOutput
from pydantic import BaseModel

from cube_harness.core import (
    AgentEvent,
    AgentOutput,
    EvaluationEvent,
    ToolCallEvent,
    Trajectory,
    TrajectoryEvent,
    TrajectoryMetadata,
    TrajectoryStep,
)
from cube_harness.episode_logs import get_log_path as get_episode_log_path
from cube_harness.episode_logs import trajectory_log_id
from cube_harness.episode_status import STATUS_FILENAME, EpisodeStatus

if TYPE_CHECKING:
    from cube_harness.episode import EpisodeConfig

logger = logging.getLogger(__name__)

EPISODES_DIR = "episodes"
EVENTS_DIR = "events"
TRAJECTORIES_DIR = "trajectories"
EPISODE_METADATA = "episode.metadata.json"
STEPS_DIR = "steps"
ARCHIVED_MARKER = ".archived_"


class LLMCallRef(BaseModel):
    llm_call_id: str


class Storage(Protocol):
    def save_metadata(self, meta: TrajectoryMetadata, allow_overwrite: bool = False) -> None: ...

    def finalize_episode(self, meta: TrajectoryMetadata) -> None: ...

    def save_event(self, event: TrajectoryEvent, trajectory_id: str) -> int:
        """Persist one TrajectoryEvent and return its assigned event_num.

        Storage owns numbering — callers don't coordinate."""
        ...

    def load_episode(self, trajectory_id: str) -> "TrajectoryView": ...

    def list_episodes(self) -> list[TrajectoryMetadata]: ...

    def save_episode_config(self, episode_config: "EpisodeConfig") -> None: ...

    def update_experiment_summary(self, meta: TrajectoryMetadata) -> None: ...

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


def _event_kind(event: TrajectoryEvent) -> str:
    """Map a TrajectoryEvent to its on-disk filename suffix
    (`agent` / `tool_call` / `eval`)."""
    if isinstance(event.output, AgentEvent):
        return "agent"
    if isinstance(event.output, ToolCallEvent):
        return "tool_call"
    if isinstance(event.output, EvaluationEvent):
        return "eval"
    raise TypeError(f"Unknown event output type: {type(event.output).__name__}")


def _event_filename(event_num: int, event: TrajectoryEvent) -> str:
    """Build the on-disk filename for one event: `NNN_<kind>.msgpack.zst`."""
    return f"{event_num:03d}_{_event_kind(event)}.msgpack.zst"


def _serialize_event(event: TrajectoryEvent) -> bytes:
    """Compress a TrajectoryEvent to bytes (msgpack + zstd, level 3)."""
    data = json.loads(event.model_dump_json())
    packed = msgpack.packb(data, use_bin_type=True)
    return _get_compressor().compress(packed)


def _deserialize_event(raw: bytes) -> dict:
    """Decode bytes written by `_serialize_event` back to a dict ready
    for `TrajectoryEvent.model_validate`."""
    decompressed = _get_decompressor().decompress(raw)
    return msgpack.unpackb(decompressed, raw=False)


def _events_to_legacy_steps(events: list[TrajectoryEvent]) -> list[TrajectoryStep]:
    """Materialize a legacy `steps` view from an event stream.

    Used by FileStorage.load_trajectory to keep XRay and other
    `trajectory.steps`-walking consumers working transparently while
    Phase I (XRay's full event-card timeline) is in progress.

    Mapping:
      - AgentEvent     → TrajectoryStep(output=AgentOutput(...)) carrying
                         the same actions / llm_calls / thoughts /
                         profiling / error.
      - ToolCallEvent  → TrajectoryStep(output=EnvironmentOutput) — the
                         tool result's underlying env output.
      - EvaluationEvent → omitted; the terminal reward lives in
                         Trajectory.reward_info already.
    """
    out: list[TrajectoryStep] = []
    for ev in events:
        body = ev.output
        if isinstance(body, AgentEvent):
            out.append(
                TrajectoryStep(
                    output=AgentOutput(
                        actions=list(body.actions),
                        llm_calls=list(body.llm_calls),
                        error=body.error,
                        profiling=dict(body.profiling),
                        thoughts=body.thoughts,
                    ),
                    start_time=ev.start_time,
                    end_time=ev.end_time,
                )
            )
        elif isinstance(body, ToolCallEvent):
            # Synthesize an EnvironmentOutput from the slim ToolCallEvent
            # (obs + error). reward/done/info are not on ToolCallEvent
            # anymore — reward lives on step-wise EvaluationEvents,
            # done is signalled by TaskDone (no longer stored). The
            # legacy XRay view sees zeros for those, which is fine —
            # the new event-card UI (follow-up PR) renders the proper
            # event types directly.
            out.append(
                TrajectoryStep(
                    output=EnvironmentOutput(
                        obs=body.obs,
                        reward=0.0,
                        done=False,
                        info={},
                        error=body.error,
                    ),
                    start_time=ev.start_time,
                    end_time=ev.end_time,
                )
            )
        elif isinstance(body, EvaluationEvent):
            # EvaluationEvent terminal payload is reflected in
            # trajectory.reward_info; no legacy step equivalent.
            pass
    return out


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


# --- TrajectoryView lazy loader (RFC: agent-owns-loop scope expansion) -------


@dataclass
class _EventIndexEntry:
    """One slot in `TrajectoryView._index`.

    `kind` is the canonical event kind (`agent` / `tool_call` / `eval`)
    regardless of on-disk layout. For V1 / V2-steps legacy layouts the
    entry points at a `_act.msgpack.zst` or `_obs.msgpack.zst` file; the
    view synthesizes a `TrajectoryEvent` on decode. For V2-events the
    file is already an event."""

    num: int
    kind: str  # 'agent' | 'tool_call' | 'eval'
    path: Path
    legacy: bool = False
    legacy_parent_num: int | None = None  # only set for legacy obs entries


class TrajectoryView:
    """Lazy reader for one episode directory.

    Replaces in-memory `Trajectory` for every consumer that walks events.
    `metadata` is loaded eagerly (one JSON read); events are decoded
    from disk on demand and cached in an internal `dict[int, TrajectoryEvent]`
    scoped to the view's lifetime (AgentLab pattern: no LRU, no eviction —
    when the view is GC'd the cache goes with it).

    Use via:

        view = storage.load_episode(id)
        view.metadata.summary_stats           # cheap
        for event in view: ...                # lazy, one decode at a time
        view[i]                               # random access, cached
        len(view)                             # from index, no decode
    """

    def __init__(
        self,
        storage: "FileStorage",
        trajectory_id: str,
        meta: TrajectoryMetadata,
        index: list[_EventIndexEntry],
    ) -> None:
        self.storage = storage
        self.id = trajectory_id
        self._meta = meta
        self._index = index
        self._cache: dict[int, TrajectoryEvent] = {}

    # --- Metadata shortcuts. The accessors below mirror the legacy
    # `Trajectory.<field>` API so existing call sites that did
    # `trajectory.metadata["task_id"]` keep working as `view.metadata["task_id"]`.

    @property
    def metadata(self) -> dict:
        """Free-form episode metadata dict (task_id, agent_name, action_schemas, …).

        Same shape as the legacy `Trajectory.metadata`. Mutating the
        returned dict won't be re-persisted — `Episode.run` is the only
        writer, and it merges any side-channel updates from the recorder
        into this dict at finalize_episode time.
        """
        return self._meta.metadata

    @property
    def start_time(self) -> float | None:
        """Episode start time (Unix timestamp)."""
        return self._meta.start_time

    @property
    def end_time(self) -> float | None:
        """Episode end time (Unix timestamp), or None if the episode is
        still running / crashed before finalize."""
        return self._meta.end_time

    @property
    def episode_metadata(self) -> TrajectoryMetadata:
        """The full `TrajectoryMetadata` record. Most callers want the
        individual shortcuts above; this is for callers (eval_log,
        atlas-style aggregation) that pass the metadata around whole."""
        return self._meta

    @property
    def n_agent_events(self) -> int:
        """Number of AgentEvent entries — read from the index, no decode."""
        return sum(1 for e in self._index if e.kind == "agent")

    @property
    def n_tool_calls(self) -> int:
        """Number of ToolCallEvent entries — read from the index, no decode."""
        return sum(1 for e in self._index if e.kind == "tool_call")

    @property
    def n_evaluations(self) -> int:
        """Number of EvaluationEvent entries (≤1 per episode) — index-only."""
        return sum(1 for e in self._index if e.kind == "eval")

    @property
    def is_complete(self) -> bool:
        """True once the episode finalized (`end_time` is set)."""
        return self._meta.is_complete

    @property
    def summary_stats(self) -> dict | None:
        """Aggregate per-episode stats produced by SummaryProcessor."""
        return self._meta.summary_stats

    @property
    def reward_info(self) -> dict:
        """Terminal reward + info dict (mirrors the final EvaluationEvent)."""
        return self._meta.reward_info

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> TrajectoryEvent:
        """Decode event at index `i`, caching by index for repeat access."""
        if i < 0:
            i = len(self._index) + i
        if not 0 <= i < len(self._index):
            raise IndexError(i)
        if i not in self._cache:
            self._cache[i] = self._decode(i)
        return self._cache[i]

    def __iter__(self) -> Iterator[TrajectoryEvent]:
        for i in range(len(self._index)):
            yield self[i]

    def iter_events(self) -> Iterator[TrajectoryEvent]:
        """Alias for `iter(view)` — explicit-method form for readability."""
        return iter(self)

    def events_of_turn(self, turn_id: str) -> list[TrajectoryEvent]:
        """All `ToolCallEvent`s sharing a `turn_id`. Decodes one pass."""
        return [e for e in self if isinstance(e.output, ToolCallEvent) and e.output.turn_id == turn_id]

    def last_env_output(self) -> EnvironmentOutput | None:
        """Most recent `ToolCallEvent` as an `EnvironmentOutput`-shaped
        record, or None if no tool call ran.

        Walks the index in reverse (cheap — kind is in the entry) and
        decodes only the matching event. The slim post-refactor
        `ToolCallEvent` (`obs` + `error`) is wrapped back into an
        `EnvironmentOutput` for legacy callers that expect that shape;
        `reward` / `done` / `info` are zero/empty since those no longer
        live on `ToolCallEvent` (reward is on the sibling
        `EvaluationEvent`, done is a TaskDone signal)."""
        for i in range(len(self._index) - 1, -1, -1):
            if self._index[i].kind == "tool_call":
                ev = self[i]
                if isinstance(ev.output, ToolCallEvent):
                    return EnvironmentOutput(
                        obs=ev.output.obs,
                        reward=0.0,
                        done=False,
                        info={},
                        error=ev.output.error,
                    )
        return None

    def _decode(self, i: int) -> TrajectoryEvent:
        entry = self._index[i]
        if entry.legacy:
            step_data = _deserialize_step(entry.path.read_bytes())
            step = TrajectoryStep.model_validate(step_data)
            return self._step_to_event(step, entry)
        data = _deserialize_event(entry.path.read_bytes())
        return TrajectoryEvent.model_validate(data)

    @staticmethod
    def _step_to_event(step: TrajectoryStep, entry: _EventIndexEntry) -> TrajectoryEvent:
        """Synthesize a TrajectoryEvent from a legacy step file.

        Used by V2-steps and V1-jsonl layouts. Parent / turn ids are
        derived from the step number (deterministic, so siblings line
        up across decodes).
        """
        if isinstance(step.output, AgentOutput):
            agent_event = AgentEvent.from_agent_output(step.output)
            # Override the default UUID with a deterministic id so child
            # ToolCallEvents can reference it across decodes.
            agent_event = agent_event.model_copy(update={"id": _legacy_agent_id(entry.num)})
            return TrajectoryEvent(output=agent_event, start_time=step.start_time, end_time=step.end_time)
        if isinstance(step.output, EnvironmentOutput):
            parent_id = (
                _legacy_agent_id(entry.legacy_parent_num) if entry.legacy_parent_num is not None else "__reset__"
            )
            # ToolCallEvent's fields are (id, parent_event_id, action_id,
            # obs, error, turn_id) — extract obs + error from the legacy
            # EnvironmentOutput. The old `output=step.output` form was
            # dropped silently by pydantic's extra="ignore" (default on
            # TypedBaseModel), zeroing every legacy ToolCallEvent's
            # observation on read.
            tool_event = ToolCallEvent(
                parent_event_id=parent_id,
                obs=step.output.obs,
                error=step.output.error,
                turn_id=parent_id,
            )
            return TrajectoryEvent(output=tool_event, start_time=step.start_time, end_time=step.end_time)
        raise TypeError(f"Unexpected legacy step output type: {type(step.output).__name__}")


def _legacy_agent_id(num: int) -> str:
    """Deterministic AgentEvent.id for V1/V2-steps legacy episodes."""
    return f"legacy_agent_{num:03d}"


def _max_event_num_on_disk(events_dir: Path) -> int:
    """Return the highest event-num present in `events_dir`, or -1 if
    empty / missing. Used by `FileStorage._get_event_counter` to seed
    its lazy counter so a resumed episode continues numbering past
    whatever was already written."""
    if not events_dir.exists():
        return -1
    highest = -1
    for path in events_dir.iterdir():
        if not path.name.endswith(".msgpack.zst"):
            continue
        num_str = path.name.split("_", 1)[0]
        try:
            highest = max(highest, int(num_str))
        except ValueError:
            continue
    return highest


def _build_events_index(events_dir: Path) -> list[_EventIndexEntry]:
    """Scan events/ and build an index entry per `NNN_<kind>.msgpack.zst`."""
    entries: list[_EventIndexEntry] = []
    for path in sorted(events_dir.iterdir()):
        if not path.name.endswith(".msgpack.zst"):
            continue
        stem = path.name[: -len(".msgpack.zst")]
        num_str, _, kind = stem.partition("_")
        try:
            num = int(num_str)
        except ValueError:
            continue
        if kind not in ("agent", "tool_call", "eval"):
            continue
        entries.append(_EventIndexEntry(num=num, kind=kind, path=path))
    return entries


def _build_legacy_steps_index(steps_dir: Path) -> list[_EventIndexEntry]:
    """Scan a legacy `steps/` dir and build event-shaped index entries.

    `_act.msgpack.zst` files become `agent` entries; `_obs.msgpack.zst`
    files become `tool_call` entries whose `legacy_parent_num` references
    the most recent `_act` step.
    """
    entries: list[_EventIndexEntry] = []
    last_agent_num: int | None = None
    for path in sorted(steps_dir.iterdir()):
        if not path.name.endswith(".msgpack.zst"):
            continue
        stem = path.name[: -len(".msgpack.zst")]
        num_str, _, suffix = stem.partition("_")
        try:
            num = int(num_str)
        except ValueError:
            continue
        if suffix.startswith("act"):
            entries.append(_EventIndexEntry(num=num, kind="agent", path=path, legacy=True))
            last_agent_num = num
        elif suffix.startswith("obs"):
            entries.append(
                _EventIndexEntry(
                    num=num,
                    kind="tool_call",
                    path=path,
                    legacy=True,
                    legacy_parent_num=last_agent_num,
                )
            )
    return entries


def _meta_to_trajectory(view: "TrajectoryView") -> Trajectory:
    """Build a legacy-shape `Trajectory` from an TrajectoryView's metadata
    (`steps=[]`). Used by legacy metadata-only loaders that don't need
    event payloads — XRay's experiment table, retry detection, etc."""
    return Trajectory(
        id=view.id,
        metadata=dict(view.metadata),
        start_time=view.start_time,
        end_time=view.end_time,
        reward_info=dict(view.reward_info),
        summary_stats=dict(view.summary_stats) if view.summary_stats else None,
        steps=[],
    )


def _episode_meta_to_trajectory(meta: TrajectoryMetadata) -> Trajectory:
    """Like `_meta_to_trajectory` but works directly from an
    TrajectoryMetadata record (no view construction). Used by
    `load_all_trajectory_metadata` for the cheap study-scan path."""
    return Trajectory(
        id=meta.id,
        metadata=dict(meta.metadata),
        start_time=meta.start_time,
        end_time=meta.end_time,
        reward_info=dict(meta.reward_info),
        summary_stats=dict(meta.summary_stats) if meta.summary_stats else None,
        steps=[],
    )


def _episode_metadata_from_dict(data: dict, fallback_id: str) -> TrajectoryMetadata:
    """Coerce a raw dict (from disk JSON) into an `TrajectoryMetadata`.

    Tolerant of legacy `trajectory.json` files that may carry extra
    fields (`steps`, `events`, `streaming`, …): only fields declared on
    `TrajectoryMetadata` are read. Missing `id` falls back to `fallback_id`
    (used when loading a crashed-mid-run dir whose metadata wasn't yet
    written).
    """
    allowed = set(TrajectoryMetadata.model_fields)
    filtered = {k: v for k, v in data.items() if k in allowed}
    filtered.setdefault("id", fallback_id)
    return TrajectoryMetadata.model_validate(filtered)


class FileStorage:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self._saved_ids: set[str] = set()
        # Per-trajectory event-number counter. Lazily seeded from the
        # existing events/ dir on first save_event call (so resumes /
        # retries continue numbering past the prior run). `itertools.count`
        # is thread-safe in CPython — `next()` on it is one C call, no
        # read-modify-write window — so parallel `save_event` from
        # `asyncio.to_thread` workers never collide. The dict access
        # itself is guarded by `_event_counter_lock` for first-init only.
        self._event_counters: dict[str, itertools.count] = {}
        self._event_counter_lock = threading.Lock()

    def __getstate__(self) -> dict:
        # threading.Lock can't be pickled. Ray pickles FileStorage when
        # it ships Episodes to workers; this hook strips the lock for
        # transport. __setstate__ reseeds a fresh lock + empty counters
        # on the worker side — the worker will lazily re-init counters
        # from its own filesystem state on first save_event.
        state = self.__dict__.copy()
        state.pop("_event_counter_lock", None)
        state["_event_counters"] = {}
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._event_counter_lock = threading.Lock()
        # _event_counters was reset to {} in __getstate__; this is a
        # no-op safeguard against future pickle formats.
        if not hasattr(self, "_event_counters"):
            self._event_counters = {}

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
        """Legacy entry — persists a `Trajectory` as the V2 on-disk layout.

        Internally splits into:
          - `save_metadata(TrajectoryMetadata(...))` — `episode.metadata.json`.
          - one `_write_step` per `trajectory.steps[i]` — `steps/NNN_*.msgpack.zst`.

        New code should call `save_metadata(TrajectoryMetadata(...))` directly
        and stream events through `save_event(event, id, n)`. This wrapper
        exists for legacy test fixtures and the few in-tree callers that
        still build a `Trajectory` by hand. Will be removed in the
        follow-up XRay-rewrite PR alongside the legacy step API."""
        meta = TrajectoryMetadata(
            id=trajectory.id,
            metadata=dict(trajectory.metadata),
            start_time=trajectory.start_time,
            end_time=trajectory.end_time,
            reward_info=dict(trajectory.reward_info),
            summary_stats=dict(trajectory.summary_stats) if trajectory.summary_stats else None,
        )
        self.save_metadata(meta, allow_overwrite=allow_overwrite)
        ep_dir = self._episode_dir(trajectory.id)
        (ep_dir / STEPS_DIR).mkdir(exist_ok=True)
        for i, step in enumerate(trajectory.steps):
            self._write_step(ep_dir, i, step)
        logger.info(f"Saved legacy trajectory to {ep_dir}")

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

    # --- TrajectoryMetadata write-at-start API (RFC: agent-owns-loop scope expansion) ---

    def save_metadata(self, meta: TrajectoryMetadata, allow_overwrite: bool = False) -> None:
        """Write `episode.metadata.json` for this episode.

        Called twice per episode: at START with `end_time=None` and stub
        summary fields, then at END via `finalize_episode` with the
        final summary. Both calls go through this method.

        First call for a given id creates the directory + the metadata
        file. Second+ call for the same id (tracked via `_saved_ids`)
        is treated as a re-save and is always allowed — that's how the
        start → end pattern works.

        First call for a NEW id when the directory already exists on
        disk (a retry / overwrite scenario) requires `allow_overwrite=True`
        to archive the old episode before writing.
        """
        ep_dir = self._episode_dir(meta.id)
        metadata_path = ep_dir / EPISODE_METADATA
        is_resave = meta.id in self._saved_ids

        if not is_resave and ep_dir.exists() and metadata_path.exists():
            if not allow_overwrite:
                raise FileExistsError(
                    f"Episode '{meta.id}' already exists at {ep_dir}. "
                    "Use allow_overwrite=True to archive and overwrite."
                )
            self._archive_episode(ep_dir)

        ep_dir.mkdir(parents=True, exist_ok=True)
        self._saved_ids.add(meta.id)

        metadata_path.write_text(json.dumps(meta.model_dump(mode="json"), indent=2))
        logger.info(f"Saved episode metadata to {ep_dir}")

    def finalize_episode(self, meta: TrajectoryMetadata) -> None:
        """Write the final episode metadata at episode end.

        Idempotent re-save: the same `episode.metadata.json` file is
        overwritten with the complete `end_time` + `summary_stats` +
        `reward_info`. Always allowed because `_saved_ids` records the
        start-write.
        """
        self.save_metadata(meta)

    def save_step(self, step: TrajectoryStep, trajectory_id: str, step_num: int) -> None:
        """Legacy V2 step writer. Pair to `save_trajectory`. New code
        streams via `save_event(event, trajectory_id)` instead — same
        removal target as `save_trajectory`."""
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

    # --- Event-stream layout (RFC: agent-owns-loop, Phase F) ---

    def save_event(self, event: TrajectoryEvent, trajectory_id: str) -> int:
        """Persist one TrajectoryEvent; return the assigned event_num.

        Files land at episodes/<trajectory_id>/events/<NNN>_<kind>.msgpack.zst
        with kind ∈ {agent, tool_call, eval}. The episodes/ dir must
        exist (call save_metadata first); the events/ dir is created
        lazily on first save. Numbering is owned by the storage —
        callers don't pass an event_num and don't coordinate. Safe
        under concurrent writes from `asyncio.to_thread` workers
        (e.g. GennyParallel's parallel tool dispatch)."""
        ep_dir = self._episode_dir(trajectory_id)
        if not ep_dir.exists():
            raise ValueError(f"Episode directory does not exist: {ep_dir}. Call save_metadata first.")
        events_dir = ep_dir / EVENTS_DIR
        events_dir.mkdir(exist_ok=True)
        event_num = next(self._get_event_counter(trajectory_id, events_dir))
        try:
            (events_dir / _event_filename(event_num, event)).write_bytes(_serialize_event(event))
        except Exception as e:
            logger.exception(f"Error saving event to trajectory {trajectory_id}: {e}")
            raise e
        return event_num

    def _get_event_counter(self, trajectory_id: str, events_dir: Path) -> itertools.count:
        """Return the per-trajectory event-number counter.

        Lazily seeded from existing events/ contents the first time —
        resumes / retries continue past the prior run's last number.
        Locked only on first init; steady-state lookup is dict-read
        + `next(counter)`, both safe under CPython's GIL."""
        counter = self._event_counters.get(trajectory_id)
        if counter is not None:
            return counter
        with self._event_counter_lock:
            counter = self._event_counters.get(trajectory_id)
            if counter is not None:
                return counter
            start = _max_event_num_on_disk(events_dir) + 1
            counter = itertools.count(start)
            self._event_counters[trajectory_id] = counter
            return counter

    def load_event(self, trajectory_id: str, event_num: int) -> TrajectoryEvent:
        ep_dir = self._episode_dir(trajectory_id)
        events_dir = ep_dir / EVENTS_DIR
        if not events_dir.exists():
            raise FileNotFoundError(f"No events directory at {events_dir}")
        for candidate in sorted(events_dir.iterdir()):
            if candidate.name.startswith(f"{event_num:03d}_") and candidate.name.endswith(".msgpack.zst"):
                return TrajectoryEvent.model_validate(_deserialize_event(candidate.read_bytes()))
        raise FileNotFoundError(f"No event at {events_dir}/{event_num:03d}_*")

    # --- TrajectoryView lazy load (RFC: agent-owns-loop scope expansion) ---

    def load_episode(self, trajectory_id: str) -> TrajectoryView:
        """Cheap lazy view onto an episode directory.

        Reads only `episode.metadata.json` (or its V1 equivalent) and
        the directory listing for events/ (or steps/). No event
        payloads are decoded; iteration / `view[i]` pays per-event I/O
        on demand.

        Auto-detects layout in this order:

        - V2 with `episode.metadata.json` + `events/` → standard.
        - V2 with `episode.metadata.json` + `steps/` (no `events/`) →
          legacy-upgrade view; iterator synthesizes events from step files.
        - V2 with `episode.metadata.json` + neither dir → empty view
          (e.g., crashed before any tool call).
        - V1 jsonl (`<id>.metadata.json` + `<id>.jsonl`) → legacy V1
          upgrade view (eager-loads the jsonl into a synthetic index
          since V1 has no per-step files to lazy-decode).
        - Mid-run crash with `events/` but no metadata file → stub
          metadata view with `is_complete=False`.
        """
        ep_dir = self._episode_dir(trajectory_id)
        metadata_path = ep_dir / EPISODE_METADATA

        if metadata_path.exists():
            return self._load_v2_episode_view(ep_dir, trajectory_id)
        # Mid-run crash: events/ exists but metadata wasn't written yet.
        if (ep_dir / EVENTS_DIR).exists() or (ep_dir / STEPS_DIR).exists():
            return self._load_v2_episode_view(ep_dir, trajectory_id, stub_metadata=True)
        # V1 layout (top-level <id>.metadata.json + <id>.jsonl).
        return self._v1_load_episode_view(trajectory_id)

    def _load_v2_episode_view(
        self,
        ep_dir: Path,
        trajectory_id: str,
        stub_metadata: bool = False,
    ) -> TrajectoryView:
        """Build TrajectoryView for a V2-layout episode (events/ or steps/)."""
        metadata_path = ep_dir / EPISODE_METADATA
        if stub_metadata:
            data: dict = {"id": trajectory_id}
        else:
            with open(metadata_path) as f:
                data = json.load(f)
        # status.json + failure.txt land inside metadata.metadata for XRay.
        self._maybe_inject_failure_text(ep_dir, data)
        self._maybe_inject_episode_status(ep_dir, data)
        meta = _episode_metadata_from_dict(data, trajectory_id)
        index = self._build_v2_index(ep_dir)
        return TrajectoryView(self, trajectory_id, meta, index)

    def _build_v2_index(self, ep_dir: Path) -> list[_EventIndexEntry]:
        """Build the TrajectoryView index for a V2 layout.

        Prefer events/ when present; fall back to steps/ (legacy-upgrade)
        when only steps/ exists. Returns [] if neither dir is on disk —
        a crashed-before-any-event episode is a valid view.
        """
        events_dir = ep_dir / EVENTS_DIR
        if events_dir.exists():
            return _build_events_index(events_dir)
        steps_dir = ep_dir / STEPS_DIR
        if steps_dir.exists():
            return _build_legacy_steps_index(steps_dir)
        return []

    def _v1_load_episode_view(self, trajectory_id: str) -> TrajectoryView:
        """Build TrajectoryView for a V1 jsonl-layout episode.

        V1 has no per-step files to lazy-decode, so this eager-loads
        the jsonl into a synthetic index that points at a series of
        in-memory TrajectoryStep objects. Acceptable: V1 episodes are
        historical and bounded in size.
        """
        metadata_path, steps_path = self._v1_resolve_trajectory_paths(trajectory_id)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Trajectory metadata not found: {metadata_path}")
        with open(metadata_path) as f:
            data = json.load(f)
        if "metadata" not in data:
            data = {"id": trajectory_id, "metadata": data}
        meta = _episode_metadata_from_dict(data, trajectory_id)

        # V1 jsonl: parse it into in-memory steps, then map to a synthetic
        # index whose entries point at TrajectoryStep payloads via a side
        # dict on the storage (one-shot for this view).
        view = TrajectoryView(self, trajectory_id, meta, [])
        if steps_path.exists():
            v1_steps: list[TrajectoryStep] = []
            with open(steps_path) as f:
                for i, line in enumerate(f):
                    if line.strip():
                        step_data = json.loads(line)
                        step_data = self._v1_resolve_llm_call_refs(step_data, trajectory_id, i)
                        if "output" not in step_data and ("obs" in step_data or "actions" in step_data):
                            step_data = {"output": step_data}
                        v1_steps.append(TrajectoryStep.model_validate(step_data))
            last_agent_num: int | None = None
            for i, step in enumerate(v1_steps):
                if isinstance(step.output, AgentOutput):
                    entry = _EventIndexEntry(num=i, kind="agent", path=steps_path, legacy=True)
                    last_agent_num = i
                elif isinstance(step.output, EnvironmentOutput):
                    entry = _EventIndexEntry(
                        num=i,
                        kind="tool_call",
                        path=steps_path,
                        legacy=True,
                        legacy_parent_num=last_agent_num,
                    )
                else:
                    continue
                view._index.append(entry)
                # Pre-populate the cache since V1 has no per-event files.
                view._cache[len(view._index) - 1] = TrajectoryView._step_to_event(step, entry)
        return view

    def list_episodes(self) -> list[TrajectoryMetadata]:
        """Cheap study-scan: one JSON read per episode dir, no events.

        Used by study aggregation, EpisodeRecord generation, Atlas
        indexing — everything that needs a list of episodes but not
        their events.
        """
        results: list[TrajectoryMetadata] = []
        for ep_dir in self._episode_dirs():
            try:
                with open(ep_dir / EPISODE_METADATA) as f:
                    data = json.load(f)
                self._maybe_inject_failure_text(ep_dir, data)
                self._maybe_inject_episode_status(ep_dir, data)
                results.append(_episode_metadata_from_dict(data, ep_dir.name))
            except Exception as e:
                logger.error(f"Failed to load episode metadata {ep_dir.name}: {e}")
        for metadata_file in self._v1_metadata_files():
            trajectory_id = self._v1_traj_id_from_file(metadata_file)
            try:
                with open(metadata_file) as f:
                    data = json.load(f)
                if "metadata" not in data:
                    data = {"id": trajectory_id, "metadata": data}
                results.append(_episode_metadata_from_dict(data, trajectory_id))
            except Exception as e:
                logger.error(f"Failed to load V1 episode metadata {trajectory_id}: {e}")
        return results

    # --- Load single trajectory ---

    def load_trajectory(self, trajectory_id: str) -> Trajectory:
        """Legacy entry point — returns a `Trajectory` for XRay /
        investigator / inspect_results consumers.

        Uses `load_episode` for metadata and event-format steps. For
        episodes that ONLY have a legacy `steps/` dir on disk (no
        `events/`), reads `TrajectoryStep` files directly to preserve
        the full `EnvironmentOutput` (reward / done / info) — going
        through the event-synthesis round-trip would lose those fields.
        Same lossless treatment for V1 jsonl episodes via the existing
        `_v1_load_trajectory_steps` path.

        Kept until XRay / investigator / inspect_results migrate to
        `TrajectoryView` directly (planned follow-up PR
        `agent-owns-loop-xray`)."""
        view = self.load_episode(trajectory_id)
        ep_dir = self._episode_dir(trajectory_id)
        has_events = (ep_dir / EVENTS_DIR).exists()
        if not has_events and (ep_dir / STEPS_DIR).exists():
            # Legacy V2-steps episode: read step files directly so we
            # don't lose the full EnvironmentOutput in the event-synthesis
            # round-trip (TrajectoryView builds slim ToolCallEvent(obs,
            # error) and `_events_to_legacy_steps` then synthesizes back
            # with reward=0/done=False — data loss).
            steps: list[TrajectoryStep] = []
            for step_file in sorted((ep_dir / STEPS_DIR).iterdir()):
                step_data = _read_step_file(step_file)
                if step_data is not None:
                    steps.append(TrajectoryStep.model_validate(step_data))
        elif not has_events and not (ep_dir / STEPS_DIR).exists():
            # V1 jsonl episode at output_dir/<id>.jsonl. Read directly to
            # preserve full EnvironmentOutput fields.
            steps = self._v1_read_steps(trajectory_id)
        else:
            steps = _events_to_legacy_steps(list(view))
        return Trajectory(
            id=view.id,
            metadata=dict(view.metadata),
            start_time=view.start_time,
            end_time=view.end_time,
            reward_info=dict(view.reward_info),
            summary_stats=dict(view.summary_stats) if view.summary_stats else None,
            steps=steps,
        )

    def _v1_read_steps(self, trajectory_id: str) -> list[TrajectoryStep]:
        """Read the V1 `<id>.jsonl` file into a list of `TrajectoryStep`s.

        Used by `load_trajectory` for V1-backcompat to bypass the
        event-synthesis round-trip (which would lose reward / done /
        info from EnvironmentOutput)."""
        _, steps_path = self._v1_resolve_trajectory_paths(trajectory_id)
        steps: list[TrajectoryStep] = []
        if not steps_path.exists():
            return steps
        with open(steps_path) as f:
            for i, line in enumerate(f):
                if line.strip():
                    step_data = json.loads(line)
                    step_data = self._v1_resolve_llm_call_refs(step_data, trajectory_id, i)
                    if "output" not in step_data and ("obs" in step_data or "actions" in step_data):
                        step_data = {"output": step_data}
                    steps.append(TrajectoryStep.model_validate(step_data))
        return steps

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

    def load_step(self, trajectory_id: str, step_index: int) -> TrajectoryStep:
        """Load one legacy step file (V2-steps layout only).

        New event-stream episodes don't have step files on disk; this
        path is only used by callers that still hold legacy step indices.
        """
        ep_dir = self._episode_dir(trajectory_id)
        if not ep_dir.exists():
            raise FileNotFoundError(f"Episode directory not found for trajectory: {trajectory_id}")
        steps_dir = ep_dir / STEPS_DIR
        for suffix in ("obs", "act"):
            path = steps_dir / f"{step_index:03d}_{suffix}.msgpack.zst"
            if path.exists():
                return TrajectoryStep.model_validate(_deserialize_step(path.read_bytes()))
        raise IndexError(f"Step {step_index} not found in {steps_dir}")

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
        """Legacy entry — returns a `Trajectory` with `steps=[]`.

        Thin wrapper over `load_episode` followed by `_meta_to_trajectory`.
        Used by XRay's metadata-only loader path (fast scan, no event decode)."""
        view = self.load_episode(trajectory_id)
        return _meta_to_trajectory(view)

    # --- Bulk listing ---

    def load_all_trajectory_metadata(self) -> list[Trajectory]:
        """Legacy entry — returns one `Trajectory(steps=[])` per episode.

        Thin wrapper over `list_episodes`. Used by XRay's experiment-level
        list view (one row per episode, no event decode)."""
        return [_episode_meta_to_trajectory(meta) for meta in self.list_episodes()]

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
        """Return ``{trajectory_id: episode-dir mtime}`` for every non-archived V2 episode dir.

        Uses the directory's own mtime as the single change signal — one ``stat()`` per
        episode rather than statting each inner file (summary/metadata/failure). This is
        both cheaper (matters for 1000-episode runs polled once per second) and *more*
        complete: a directory's mtime advances on any entry add/remove inside it, which
        includes every atomic ``status.json`` write (tmp + ``os.replace``). Because the
        run loop heartbeats ``status.json`` once per turn (``episode.py``), an active
        episode's dir mtime advances each turn — so this one signal covers BOTH
        trajectory-content changes and status-only transitions (STALE via ghost-sweep,
        CANCELLED via the stall-killer), including QUEUED stubs that have no trajectory
        file yet. The per-file approach missed the latter. See
        ``XRayState.refresh_experiment``.
        """
        result: dict[str, float] = {}
        episodes_dir = self.output_dir / EPISODES_DIR
        if not episodes_dir.exists():
            return result
        for ep_dir in episodes_dir.iterdir():
            if ep_dir.is_dir() and ARCHIVED_MARKER not in ep_dir.name:
                result[ep_dir.name] = ep_dir.stat().st_mtime
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
        """Legacy bulk loader — returns one full `Trajectory` per episode.

        Each entry routes through `load_trajectory` (thin wrapper over
        `load_episode`), so steps are materialized from events on the
        fly. Used by XRay / inspect_results study-aggregation paths.
        New code should iterate `list_episodes()` for metadata only and
        call `load_episode(id)` on demand.
        """
        if exp_dir is not None:
            return FileStorage(exp_dir).load_all_trajectories()
        results: list[Trajectory] = []
        for ep_dir in self._episode_dirs():
            try:
                results.append(self.load_trajectory(ep_dir.name))
            except Exception as e:
                logger.error(f"Failed to load episode {ep_dir.name}: {e}")
        for metadata_file in self._v1_metadata_files():
            trajectory_id = self._v1_traj_id_from_file(metadata_file)
            try:
                results.append(self.load_trajectory(trajectory_id))
            except Exception as e:
                logger.error(f"Failed to load V1 trajectory {trajectory_id}: {e}")
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

    def update_experiment_summary(self, meta: TrajectoryMetadata) -> None:
        """Roll one episode's summary_stats into the experiment-level
        `experiment_summary.json`. Accepts an `TrajectoryMetadata` rather
        than a Trajectory — only `meta.summary_stats` is read."""
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

                stats = meta.summary_stats or {}
                # trajectory.steps is empty post-stream refactor; the first step-level
                # error_type is captured incrementally by SummaryProcessor and lives in
                # summary_stats. Walking the (empty) step list here would have silently
                # always reported no error.
                has_error = bool(stats.get("error_type"))

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

"""Storage event-file layout tests — save_event / load_event,
trajectory-level save/load with events/ alongside steps/, dual-format
loading (legacy steps/-only AND new events/-only AND both)."""

import json
from pathlib import Path

import pytest
from cube.core import Action, EnvironmentOutput, Observation

from cube_harness.core import (
    AgentEvent,
    AgentOutput,
    EvaluationEvent,
    ToolCallEvent,
    Trajectory,
    TrajectoryEvent,
    TrajectoryStep,
)
from cube_harness.storage import EVENTS_DIR, FileStorage


def _agent_event(thoughts: str = "t") -> AgentEvent:
    return AgentEvent(actions=[Action(id="a-1", name="foo", arguments={})], thoughts=thoughts)


def _tool_call_event(parent_id: str) -> ToolCallEvent:
    return ToolCallEvent(
        parent_event_id=parent_id,
        action_id="a-1",
        output=EnvironmentOutput(obs=Observation(), reward=0.5, done=False, info={}),
        turn_id=parent_id,
    )


def _eval_event(reward: float = 1.0) -> EvaluationEvent:
    return EvaluationEvent(reward=reward, info={"score": reward})


# ---------------------------------------------------------------------------
# save_event / load_event
# ---------------------------------------------------------------------------


def test_save_and_load_event_round_trip(tmp_path: Path) -> None:
    storage = FileStorage(tmp_path)
    traj = Trajectory(id="t")
    storage.save_trajectory(traj)

    parent = _agent_event()
    te = TrajectoryEvent(output=parent, start_time=0.0, end_time=0.1)
    storage.save_event(te, "t", 0)

    # Loaded event round-trips through msgpack+zstd serialization.
    loaded = storage.load_event("t", 0)
    assert isinstance(loaded.output, AgentEvent)
    assert loaded.output.thoughts == parent.thoughts


def test_save_event_creates_events_dir_lazily(tmp_path: Path) -> None:
    storage = FileStorage(tmp_path)
    traj = Trajectory(id="t")
    storage.save_trajectory(traj)
    ep_dir = tmp_path / "episodes" / "t"
    assert not (ep_dir / EVENTS_DIR).exists()
    storage.save_event(TrajectoryEvent(output=_eval_event(), start_time=0.0, end_time=0.0), "t", 0)
    assert (ep_dir / EVENTS_DIR).exists()


def test_save_event_filename_carries_kind(tmp_path: Path) -> None:
    storage = FileStorage(tmp_path)
    storage.save_trajectory(Trajectory(id="t"))
    storage.save_event(TrajectoryEvent(output=_agent_event()), "t", 0)
    storage.save_event(TrajectoryEvent(output=_tool_call_event("p")), "t", 1)
    storage.save_event(TrajectoryEvent(output=_eval_event()), "t", 2)
    files = sorted((tmp_path / "episodes" / "t" / EVENTS_DIR).iterdir())
    names = [f.name for f in files]
    assert "000_agent.msgpack.zst" in names
    assert "001_tool_call.msgpack.zst" in names
    assert "002_eval.msgpack.zst" in names


def test_save_event_requires_episode_dir(tmp_path: Path) -> None:
    storage = FileStorage(tmp_path)
    with pytest.raises(ValueError):
        storage.save_event(TrajectoryEvent(output=_eval_event()), "missing", 0)


def test_load_event_missing_raises(tmp_path: Path) -> None:
    storage = FileStorage(tmp_path)
    storage.save_trajectory(Trajectory(id="t"))
    with pytest.raises(FileNotFoundError):
        storage.load_event("t", 99)


# ---------------------------------------------------------------------------
# Trajectory-level save/load with events
# ---------------------------------------------------------------------------


def test_save_trajectory_writes_events_alongside_steps(tmp_path: Path) -> None:
    storage = FileStorage(tmp_path)
    parent = _agent_event()
    traj = Trajectory(
        id="t",
        events=[
            TrajectoryEvent(output=parent),
            TrajectoryEvent(output=_tool_call_event(parent.id)),
            TrajectoryEvent(output=_eval_event()),
        ],
    )
    storage.save_trajectory(traj)
    events_dir = tmp_path / "episodes" / "t" / EVENTS_DIR
    files = list(events_dir.iterdir())
    assert len(files) == 3


def test_load_trajectory_recovers_events(tmp_path: Path) -> None:
    storage = FileStorage(tmp_path)
    parent = _agent_event(thoughts="my reasoning")
    traj = Trajectory(
        id="t",
        events=[
            TrajectoryEvent(output=parent),
            TrajectoryEvent(output=_tool_call_event(parent.id)),
            TrajectoryEvent(output=_eval_event(reward=0.75)),
        ],
    )
    storage.save_trajectory(traj)
    loaded = storage.load_trajectory("t")
    assert loaded.n_agent_events == 1
    assert loaded.n_tool_calls == 1
    assert loaded.n_evaluations == 1
    # AgentEvent identity preserved
    agent_event = loaded.events[0].output
    assert isinstance(agent_event, AgentEvent)
    assert agent_event.id == parent.id
    assert agent_event.thoughts == "my reasoning"


def test_load_trajectory_handles_legacy_steps_only(tmp_path: Path) -> None:
    """Trajectories written before the migration (steps/ only) must still load."""
    storage = FileStorage(tmp_path)
    legacy = Trajectory(
        id="t-legacy",
        steps=[
            TrajectoryStep(output=EnvironmentOutput(obs=Observation(), reward=0.0, done=False, info={})),
            TrajectoryStep(output=AgentOutput()),
        ],
    )
    storage.save_trajectory(legacy)
    # Confirm no events/ dir was created (none of the events were populated).
    ep_dir = tmp_path / "episodes" / "t-legacy"
    assert not (ep_dir / EVENTS_DIR).exists()

    loaded = storage.load_trajectory("t-legacy")
    assert loaded.n_env_steps == 1  # counted from legacy steps
    assert loaded.n_agent_steps == 1
    assert loaded.n_agent_events == 0


def test_load_trajectory_handles_mixed_formats(tmp_path: Path) -> None:
    """A trajectory mid-migration may end up with both steps/ AND events/.
    Both must load and the counters fold them together."""
    storage = FileStorage(tmp_path)
    parent = _agent_event()
    traj = Trajectory(
        id="t-mixed",
        steps=[
            TrajectoryStep(output=EnvironmentOutput(obs=Observation(), reward=0.0, done=False, info={})),
        ],
        events=[
            TrajectoryEvent(output=parent),
            TrajectoryEvent(output=_tool_call_event(parent.id)),
        ],
    )
    storage.save_trajectory(traj)

    loaded = storage.load_trajectory("t-mixed")
    assert loaded.n_env_steps == 1 + 1  # legacy env step + new tool-call event
    assert loaded.n_agent_events == 1
    assert loaded.n_tool_calls == 1


# ---------------------------------------------------------------------------
# Sanity: metadata round-trip preserves trajectory state
# ---------------------------------------------------------------------------


def test_episode_metadata_excludes_steps_and_events(tmp_path: Path) -> None:
    """The metadata file must not duplicate per-step / per-event content
    that lives in the steps/ and events/ directories — those would
    explode the metadata size and undermine the streamed-to-disk
    invariant from stream-trajectory-steps."""
    storage = FileStorage(tmp_path)
    parent = _agent_event()
    traj = Trajectory(
        id="t",
        events=[TrajectoryEvent(output=parent), TrajectoryEvent(output=_eval_event())],
    )
    storage.save_trajectory(traj)
    meta = json.loads((tmp_path / "episodes" / "t" / "episode.metadata.json").read_text())
    assert meta.get("steps") in (None, [])
    assert meta.get("events") in (None, [])

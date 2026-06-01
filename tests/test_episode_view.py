"""Phase M+N: EpisodeMetadata + EpisodeView lazy loader unit tests.

Validates the AgentLab-style lazy loader pattern that replaces the
Trajectory class as the consumer-facing trajectory abstraction.
"""

from cube.core import Action, EnvironmentOutput, Observation

from cube_harness.core import (
    AgentEvent,
    AgentOutput,
    EpisodeMetadata,
    EvaluationEvent,
    ToolCallEvent,
    TrajectoryEvent,
    TrajectoryStep,
)
from cube_harness.storage import FileStorage


def _agent_event(turn_index: int = 0) -> TrajectoryEvent:
    return TrajectoryEvent(
        output=AgentEvent(
            id=f"agent_{turn_index}",
            actions=[Action(name="bash", arguments={"cmd": "ls"})],
            thoughts=f"thought {turn_index}",
        ),
        start_time=1.0 + turn_index,
        end_time=1.5 + turn_index,
    )


def _tool_call_event(parent_id: str, env_output: EnvironmentOutput | None = None) -> TrajectoryEvent:
    obs = env_output or EnvironmentOutput(obs=Observation.from_text("ok"), reward=0.0)
    return TrajectoryEvent(
        output=ToolCallEvent(parent_event_id=parent_id, output=obs, turn_id=parent_id),
        start_time=2.0,
        end_time=2.5,
    )


def _eval_event() -> TrajectoryEvent:
    return TrajectoryEvent(
        output=EvaluationEvent(reward=1.0, info={"ok": True}),
        start_time=3.0,
        end_time=3.1,
    )


class TestEpisodeMetadata:
    def test_stub_metadata_is_incomplete(self) -> None:
        meta = EpisodeMetadata(id="t1", start_time=1.0)
        assert meta.is_complete is False
        assert meta.end_time is None
        assert meta.reward_info == {}
        assert meta.summary_stats is None

    def test_finalized_metadata_is_complete(self) -> None:
        meta = EpisodeMetadata(
            id="t1",
            start_time=1.0,
            end_time=2.0,
            summary_stats={"n_agent_events": 3},
            reward_info={"reward": 1.0},
        )
        assert meta.is_complete is True
        assert meta.summary_stats == {"n_agent_events": 3}

    def test_json_roundtrip(self) -> None:
        meta = EpisodeMetadata(
            id="t1",
            metadata={"task_id": "arithmetic_0"},
            start_time=1.0,
            end_time=2.0,
            summary_stats={"k": 1},
            reward_info={"reward": 0.5},
        )
        restored = EpisodeMetadata.model_validate_json(meta.model_dump_json())
        assert restored == meta


class TestEpisodeViewWriteAtStart:
    def test_metadata_at_start_then_finalize(self, tmp_path) -> None:
        storage = FileStorage(tmp_path)
        meta = EpisodeMetadata(id="t1", metadata={"task_id": "arith"}, start_time=1.0)
        storage.save_metadata(meta)

        view = storage.load_episode("t1")
        assert view.is_complete is False
        assert len(view) == 0
        assert view.summary_stats is None

        finalized = meta.model_copy(update={"end_time": 2.0, "summary_stats": {"n": 0}, "reward_info": {"reward": 0.0}})
        storage.finalize_episode(finalized)

        view2 = storage.load_episode("t1")
        assert view2.is_complete is True
        assert view2.reward_info == {"reward": 0.0}
        assert view2.summary_stats == {"n": 0}


class TestEpisodeViewIteration:
    def test_iteration_decodes_events_lazily(self, tmp_path) -> None:
        storage = FileStorage(tmp_path)
        meta = EpisodeMetadata(id="t1")
        storage.save_metadata(meta)
        events = [_agent_event(0), _tool_call_event("agent_0"), _eval_event()]
        for i, ev in enumerate(events):
            storage.save_event(ev, "t1", i)
        storage.finalize_episode(meta.model_copy(update={"end_time": 2.0}))

        view = storage.load_episode("t1")
        assert len(view) == 3
        decoded = list(view)
        assert len(decoded) == 3
        assert isinstance(decoded[0].output, AgentEvent)
        assert isinstance(decoded[1].output, ToolCallEvent)
        assert isinstance(decoded[2].output, EvaluationEvent)

    def test_random_access_caches_decode(self, tmp_path) -> None:
        storage = FileStorage(tmp_path)
        meta = EpisodeMetadata(id="t1")
        storage.save_metadata(meta)
        events = [_agent_event(0), _tool_call_event("agent_0")]
        for i, ev in enumerate(events):
            storage.save_event(ev, "t1", i)

        view = storage.load_episode("t1")
        # First access decodes; second access hits the cache.
        first = view[1]
        second = view[1]
        assert first is second

    def test_kind_counts_no_decode(self, tmp_path) -> None:
        storage = FileStorage(tmp_path)
        meta = EpisodeMetadata(id="t1")
        storage.save_metadata(meta)
        events = [
            _agent_event(0),
            _tool_call_event("agent_0"),
            _tool_call_event("agent_0"),
            _agent_event(1),
            _eval_event(),
        ]
        for i, ev in enumerate(events):
            storage.save_event(ev, "t1", i)

        view = storage.load_episode("t1")
        assert view.n_agent_events == 2
        assert view.n_tool_calls == 2
        assert view.n_evaluations == 1
        # No decoding happened just to count.
        assert view._cache == {}

    def test_events_of_turn(self, tmp_path) -> None:
        storage = FileStorage(tmp_path)
        meta = EpisodeMetadata(id="t1")
        storage.save_metadata(meta)
        siblings = [
            _agent_event(0),
            _tool_call_event("agent_0"),
            _tool_call_event("agent_0"),
        ]
        for i, ev in enumerate(siblings):
            storage.save_event(ev, "t1", i)

        view = storage.load_episode("t1")
        turn = view.events_of_turn("agent_0")
        assert len(turn) == 2

    def test_last_env_output(self, tmp_path) -> None:
        storage = FileStorage(tmp_path)
        meta = EpisodeMetadata(id="t1")
        storage.save_metadata(meta)
        final_obs = EnvironmentOutput(obs=Observation.from_text("final"), reward=1.0)
        events = [
            _agent_event(0),
            _tool_call_event("agent_0"),
            _agent_event(1),
            _tool_call_event("agent_1", env_output=final_obs),
        ]
        for i, ev in enumerate(events):
            storage.save_event(ev, "t1", i)

        view = storage.load_episode("t1")
        last = view.last_env_output()
        assert last is not None
        assert last.reward == 1.0

    def test_last_env_output_none_when_no_tool_calls(self, tmp_path) -> None:
        storage = FileStorage(tmp_path)
        meta = EpisodeMetadata(id="t1")
        storage.save_metadata(meta)
        storage.save_event(_agent_event(0), "t1", 0)

        view = storage.load_episode("t1")
        assert view.last_env_output() is None


class TestEpisodeViewCrashedMidRun:
    def test_view_loads_without_metadata_file(self, tmp_path) -> None:
        """A crash before save_metadata wrote anything: events on disk,
        no `episode.metadata.json`. The view should still load with stub
        metadata."""
        storage = FileStorage(tmp_path)
        # Manually create the dir + events bypassing save_metadata to
        # simulate a pre-metadata crash. In real usage this can't happen
        # because save_metadata is the first thing Episode does.
        ep_dir = storage._episode_dir("crashed_id")
        ep_dir.mkdir(parents=True)
        (ep_dir / "events").mkdir()
        storage._saved_ids.discard("crashed_id")
        from cube_harness.storage import _event_filename, _serialize_event

        ev = _agent_event(0)
        (ep_dir / "events" / _event_filename(0, ev)).write_bytes(_serialize_event(ev))

        view = storage.load_episode("crashed_id")
        assert view.is_complete is False
        assert len(view) == 1
        assert view.end_time is None


class TestEpisodeViewLegacyStepsLayout:
    def test_v2_steps_only_layout_synthesizes_events(self, tmp_path) -> None:
        """Old V2 episodes that only have `steps/` (no `events/`) load
        through EpisodeView with events synthesized on the fly."""
        storage = FileStorage(tmp_path)
        meta = EpisodeMetadata(id="legacy_v2")
        storage.save_metadata(meta)

        ep_dir = storage._episode_dir("legacy_v2")
        steps_dir = ep_dir / "steps"
        steps_dir.mkdir()
        # Write a legacy `_act` step (AgentOutput).
        from cube_harness.storage import _serialize_step

        act_step = TrajectoryStep(
            output=AgentOutput(actions=[Action(name="bash", arguments={"cmd": "ls"})]),
            start_time=1.0,
            end_time=1.5,
        )
        (steps_dir / "000_act.msgpack.zst").write_bytes(_serialize_step(act_step))
        obs_step = TrajectoryStep(
            output=EnvironmentOutput(obs=Observation.from_text("out"), reward=0.5),
            start_time=2.0,
            end_time=2.5,
        )
        (steps_dir / "001_obs.msgpack.zst").write_bytes(_serialize_step(obs_step))

        view = storage.load_episode("legacy_v2")
        assert len(view) == 2
        assert view.n_agent_events == 1
        assert view.n_tool_calls == 1
        first = view[0]
        second = view[1]
        assert isinstance(first.output, AgentEvent)
        assert isinstance(second.output, ToolCallEvent)
        # The synthesized parent_event_id ties them together.
        assert second.output.parent_event_id == first.output.id

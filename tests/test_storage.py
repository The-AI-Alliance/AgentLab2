import json
import time
from pathlib import Path

import pytest
from cube.core import Action, Content, EnvironmentOutput, Observation
from PIL import Image

from cube_harness.core import (
    AgentOutput,
    Trajectory,
    TrajectoryStep,
)
from cube_harness.episode_status import EpisodeStatus
from cube_harness.llm import LLMCall, LLMConfig, Message, Prompt
from cube_harness.storage import FileStorage, _deserialize_step


class TestFileStorageBasic:
    def test_init_creates_path(self, tmp_dir):
        storage = FileStorage(tmp_dir)
        assert storage.output_dir == Path(tmp_dir)
        assert storage._saved_ids == set()

    def test_init_with_string_path(self, tmp_dir):
        storage = FileStorage(str(tmp_dir))
        assert storage.output_dir == Path(tmp_dir)

    def test_save_trajectory_creates_episode_directory(self, tmp_dir):
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "TestAgent"})
        storage.save_trajectory(traj)

        episodes_dir = Path(tmp_dir) / "episodes"
        assert episodes_dir.exists()
        ep_dirs = list(episodes_dir.iterdir())
        assert len(ep_dirs) == 1
        assert ep_dirs[0].name == "task_1_ep0"

    def test_save_trajectory_creates_metadata_file(self, tmp_dir):
        storage = FileStorage(tmp_dir)
        traj = Trajectory(
            id="task_1_ep0",
            metadata={"task_id": "task_1", "agent_name": "TestAgent"},
            start_time=0.0,
            end_time=1.0,
        )
        storage.save_trajectory(traj)

        metadata_path = Path(tmp_dir) / "episodes" / "task_1_ep0" / "episode.metadata.json"
        assert metadata_path.exists()

        with open(metadata_path) as f:
            data = json.load(f)
        assert data["id"] == "task_1_ep0"
        assert data["metadata"]["task_id"] == "task_1"
        assert data["start_time"] == 0.0
        assert data["end_time"] == 1.0
        assert data["summary_stats"] is None

    def test_save_trajectory_creates_steps_directory(self, tmp_dir):
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "TestAgent"})
        storage.save_trajectory(traj)

        steps_dir = Path(tmp_dir) / "episodes" / "task_1_ep0" / "steps"
        assert steps_dir.exists()


class TestFileStorageWithSteps:
    def test_save_trajectory_with_env_step(self, tmp_dir, sample_env_output):
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        traj.steps.append(TrajectoryStep(output=sample_env_output))
        storage.save_trajectory(traj)

        steps_dir = Path(tmp_dir) / "episodes" / "task_1_ep0" / "steps"
        step_files = sorted(steps_dir.iterdir())
        assert len(step_files) == 1
        assert step_files[0].name == "000_obs.msgpack.zst"

        step_data = _deserialize_step(step_files[0].read_bytes())
        assert "output" in step_data
        assert "obs" in step_data["output"]

    def test_save_trajectory_with_agent_step(self, tmp_dir, sample_agent_output):
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        traj.steps.append(TrajectoryStep(output=sample_agent_output))
        storage.save_trajectory(traj)

        steps_dir = Path(tmp_dir) / "episodes" / "task_1_ep0" / "steps"
        step_files = sorted(steps_dir.iterdir())
        assert len(step_files) == 1
        assert step_files[0].name == "000_act.msgpack.zst"

    def test_save_trajectory_with_multiple_steps(self, tmp_dir, sample_env_output, sample_agent_output):
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        traj.steps.append(TrajectoryStep(output=sample_env_output))
        traj.steps.append(TrajectoryStep(output=sample_agent_output))
        traj.steps.append(TrajectoryStep(output=sample_env_output))
        storage.save_trajectory(traj)

        steps_dir = Path(tmp_dir) / "episodes" / "task_1_ep0" / "steps"
        step_files = sorted(steps_dir.iterdir())
        assert len(step_files) == 3
        assert [f.name for f in step_files] == ["000_obs.msgpack.zst", "001_act.msgpack.zst", "002_obs.msgpack.zst"]

    def test_save_step_appends_to_trajectory(self, tmp_dir, sample_env_output, sample_agent_output):
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        traj.steps.append(TrajectoryStep(output=sample_env_output))
        storage.save_trajectory(traj)

        storage.save_step(TrajectoryStep(output=sample_agent_output), "task_1_ep0", 1)

        steps_dir = Path(tmp_dir) / "episodes" / "task_1_ep0" / "steps"
        step_files = sorted(steps_dir.iterdir())
        assert len(step_files) == 2

    def test_save_step_without_trajectory_raises_error(self, tmp_dir, sample_env_output):
        storage = FileStorage(tmp_dir)
        with pytest.raises(ValueError, match="Episode directory does not exist"):
            storage.save_step(TrajectoryStep(output=sample_env_output), "unknown_traj", 0)


class TestFileStorageLogs:
    def test_get_log_path(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        log_path = storage.get_log_path("task_a_ep3")
        assert log_path == Path(tmp_dir) / "episodes" / "task_a_ep3" / "episode.log"

    def test_load_logs_returns_full_file_contents(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        log_path = storage.get_log_path("task_b_ep1")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("line 1\nline 2\nline 3\n")

        loaded = storage.load_logs("task_b_ep1")

        assert loaded == "line 1\nline 2\nline 3\n"
        assert storage.has_logs("task_b_ep1") is True

    def test_load_logs_missing_file(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        loaded = storage.load_logs("missing_ep0")
        assert loaded == ""
        assert storage.has_logs("missing_ep0") is False


class TestFileStorageWithLLMCalls:
    @pytest.fixture
    def sample_llm_call(self):
        return LLMCall(
            id="llm_call_1",
            llm_config=LLMConfig(model_name="test-model"),
            prompt=Prompt(messages=[{"role": "user", "content": "Hello"}]),
            output=Message(role="assistant", content="Hi there!"),
        )

    def test_v2_keeps_llm_calls_inline(self, tmp_dir, sample_llm_call):
        storage = FileStorage(tmp_dir)
        agent_output = AgentOutput(
            actions=[Action(name="click", arguments={"element": "btn"})],
            llm_calls=[sample_llm_call],
        )
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        traj.steps.append(TrajectoryStep(output=agent_output))
        storage.save_trajectory(traj)

        assert not (Path(tmp_dir) / "llm_calls").exists()

        step_file = Path(tmp_dir) / "episodes" / "task_1_ep0" / "steps" / "000_act.msgpack.zst"
        step_data = _deserialize_step(step_file.read_bytes())
        llm_calls = step_data["output"]["llm_calls"]
        assert len(llm_calls) == 1
        assert llm_calls[0]["id"] == "llm_call_1"
        assert "prompt" in llm_calls[0]
        assert "output" in llm_calls[0]

    def test_v2_multiple_llm_calls_inline(self, tmp_dir):
        storage = FileStorage(tmp_dir)
        llm_calls = [
            LLMCall(
                id=f"call_{i}",
                llm_config=LLMConfig(model_name="test-model"),
                prompt=Prompt(messages=[{"role": "user", "content": f"Message {i}"}]),
                output=Message(role="assistant", content=f"Response {i}"),
            )
            for i in range(3)
        ]
        agent_output = AgentOutput(actions=[], llm_calls=llm_calls)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        traj.steps.append(TrajectoryStep(output=agent_output))
        storage.save_trajectory(traj)

        assert not (Path(tmp_dir) / "llm_calls").exists()


class TestFileStorageLoad:
    def test_load_trajectory_basic(self, tmp_dir):
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        obs = Observation.from_text("Test observation")
        traj.steps.append(TrajectoryStep(output=EnvironmentOutput(obs=obs, reward=0.5)))
        storage.save_trajectory(traj)

        storage2 = FileStorage(tmp_dir)
        loaded = storage2.load_trajectory("task_1_ep0")

        assert loaded.id == "task_1_ep0"
        assert loaded.metadata == {"task_id": "task_1", "agent_name": "A"}
        assert len(loaded.steps) == 1

    def test_load_trajectory_preserves_step_data(self, tmp_dir, sample_env_output):
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        traj.steps.append(TrajectoryStep(output=sample_env_output, start_time=1.0, end_time=2.0))
        storage.save_trajectory(traj)

        loaded = storage.load_trajectory("task_1_ep0")
        loaded_step = loaded.steps[0]

        assert loaded_step.start_time == 1.0
        assert loaded_step.end_time == 2.0
        assert isinstance(loaded_step.output, EnvironmentOutput)
        assert loaded_step.output.reward == sample_env_output.reward

    def test_load_trajectory_not_found(self, tmp_dir):
        storage = FileStorage(tmp_dir)
        with pytest.raises(FileNotFoundError, match="Trajectory metadata not found"):
            storage.load_trajectory("nonexistent")

    def test_load_trajectory_resolves_inline_llm_calls(self, tmp_dir):
        storage = FileStorage(tmp_dir)
        llm_call = LLMCall(
            id="test_call",
            llm_config=LLMConfig(model_name="test-model"),
            prompt=Prompt(messages=[{"role": "user", "content": "Hello"}]),
            output=Message(role="assistant", content="Hi!"),
        )
        agent_output = AgentOutput(
            actions=[Action(name="test", arguments={})],
            llm_calls=[llm_call],
        )
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        traj.steps.append(TrajectoryStep(output=agent_output))
        storage.save_trajectory(traj)

        loaded = storage.load_trajectory("task_1_ep0")
        loaded_output = loaded.steps[0].output
        assert isinstance(loaded_output, AgentOutput)
        assert len(loaded_output.llm_calls) == 1
        loaded_llm_call = loaded_output.llm_calls[0]
        assert loaded_llm_call.id == "test_call"
        assert loaded_llm_call.output.content == "Hi!"


class TestFileStorageLoadAll:
    def test_load_all_empty_directory(self, tmp_dir):
        storage = FileStorage(tmp_dir)
        result = storage.load_all_trajectories()
        assert result == []

    def test_load_all_no_episodes_dir(self, tmp_dir):
        storage = FileStorage(tmp_dir)
        result = storage.load_all_trajectories()
        assert result == []

    def test_load_all_single_trajectory(self, tmp_dir, sample_env_output):
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        traj.steps.append(TrajectoryStep(output=sample_env_output))
        storage.save_trajectory(traj)

        result = storage.load_all_trajectories()
        assert len(result) == 1
        assert result[0].id == "task_1_ep0"

    def test_load_all_multiple_trajectories(self, tmp_dir, sample_env_output):
        storage = FileStorage(tmp_dir)
        for i in range(3):
            traj = Trajectory(id=f"task_{i}_ep0", metadata={"task_id": f"task_{i}", "agent_name": "A"})
            traj.steps.append(TrajectoryStep(output=sample_env_output))
            storage.save_trajectory(traj)

        result = storage.load_all_trajectories()
        assert len(result) == 3
        ids = {t.id for t in result}
        assert ids == {"task_0_ep0", "task_1_ep0", "task_2_ep0"}

    def test_load_all_with_exp_dir_parameter(self, tmp_dir, sample_env_output):
        storage1 = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        traj.steps.append(TrajectoryStep(output=sample_env_output))
        storage1.save_trajectory(traj)

        storage2 = FileStorage("/some/other/path")
        result = storage2.load_all_trajectories(exp_dir=tmp_dir)
        assert len(result) == 1
        assert result[0].id == "task_1_ep0"


class TestFileStorageWithImages:
    def test_save_and_load_trajectory_with_image(self, tmp_dir):
        storage = FileStorage(tmp_dir)
        img = Image.new("RGB", (100, 100), color="blue")
        obs = Observation(contents=[Content.from_data(img, name="screenshot")])
        env_output = EnvironmentOutput(obs=obs, reward=0.0)

        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        traj.steps.append(TrajectoryStep(output=env_output))
        storage.save_trajectory(traj)

        loaded = storage.load_trajectory("task_1_ep0")
        assert len(loaded.steps) == 1
        assert isinstance(loaded.steps[0].output, EnvironmentOutput)
        loaded_content = loaded.steps[0].output.obs.contents[0]
        assert isinstance(loaded_content.data, Image.Image)
        assert loaded_content.data.size == (100, 100)
        assert loaded_content.name == "screenshot"


class TestFileStorageRoundtrip:
    def test_full_trajectory_roundtrip(self, tmp_dir):
        storage = FileStorage(tmp_dir)
        llm_call = LLMCall(
            id="call_1",
            llm_config=LLMConfig(model_name="gpt-4"),
            prompt=Prompt(messages=[{"role": "user", "content": "Click the button"}]),
            output=Message(role="assistant", content="I'll click the button."),
        )

        traj = Trajectory(
            id="test_task_ep0",
            metadata={"task_id": "test_task", "agent_name": "TestAgent"},
            start_time=100.0,
            end_time=200.0,
        )

        obs1 = Observation.from_text("Initial state")
        traj.steps.append(
            TrajectoryStep(output=EnvironmentOutput(obs=obs1, reward=0.0), start_time=100.0, end_time=101.0)
        )
        agent_output = AgentOutput(
            actions=[Action(id="act_1", name="click", arguments={"element": "btn"})],
            llm_calls=[llm_call],
        )
        traj.steps.append(TrajectoryStep(output=agent_output, start_time=101.0, end_time=102.0))
        obs2 = Observation.from_text("Task completed")
        traj.steps.append(
            TrajectoryStep(output=EnvironmentOutput(obs=obs2, reward=1.0, done=True), start_time=102.0, end_time=103.0)
        )

        storage.save_trajectory(traj)

        storage2 = FileStorage(tmp_dir)
        loaded = storage2.load_trajectory("test_task_ep0")

        assert loaded.id == "test_task_ep0"
        assert loaded.metadata["task_id"] == "test_task"
        assert loaded.metadata["agent_name"] == "TestAgent"
        assert len(loaded.steps) == 3

        step0 = loaded.steps[0]
        assert isinstance(step0.output, EnvironmentOutput)
        assert step0.start_time == 100.0

        step1 = loaded.steps[1]
        assert isinstance(step1.output, AgentOutput)
        assert len(step1.output.actions) == 1
        assert step1.output.actions[0].name == "click"
        assert len(step1.output.llm_calls) == 1
        assert step1.output.llm_calls[0].output.content == "I'll click the button."

        step2 = loaded.steps[2]
        assert isinstance(step2.output, EnvironmentOutput)
        assert step2.output.reward == 1.0
        assert step2.output.done is True


class TestFileStorageEpisodeConfig:
    def test_save_episode_config_creates_directory(self, tmp_dir, mock_agent_config, mock_cube_task_config):
        from cube_harness.episode import EpisodeConfig

        storage = FileStorage(tmp_dir)
        episode_config = EpisodeConfig(
            id=0,
            agent_config=mock_agent_config,
            task_config=mock_cube_task_config,
            exp_name="test_exp",
            output_dir=tmp_dir,
            max_steps=100,
        )
        storage.save_episode_config(episode_config)

        ep_dir = Path(tmp_dir) / "episodes" / f"{mock_cube_task_config.task_id}_ep0"
        assert ep_dir.exists()

    def test_save_episode_config_creates_file(self, tmp_dir, mock_agent_config, mock_cube_task_config):
        from cube_harness.episode import EpisodeConfig

        storage = FileStorage(tmp_dir)
        episode_config = EpisodeConfig(
            id=5,
            agent_config=mock_agent_config,
            task_config=mock_cube_task_config,
            exp_name="test_exp",
            output_dir=tmp_dir,
            max_steps=200,
        )
        storage.save_episode_config(episode_config)

        config_path = Path(tmp_dir) / "episodes" / f"{mock_cube_task_config.task_id}_ep5" / "episode_config.json"
        assert config_path.exists()

    def test_load_episode_config_roundtrip(self, tmp_dir, mock_agent_config, mock_cube_task_config):
        from cube_harness.episode import EpisodeConfig

        storage = FileStorage(tmp_dir)
        original_config = EpisodeConfig(
            id=42,
            agent_config=mock_agent_config,
            task_config=mock_cube_task_config,
            exp_name="roundtrip_exp",
            output_dir=tmp_dir,
            max_steps=500,
        )
        storage.save_episode_config(original_config)

        config_path = Path(tmp_dir) / "episodes" / f"{mock_cube_task_config.task_id}_ep42" / "episode_config.json"
        loaded_config = storage.load_episode_config(config_path)

        assert loaded_config.id == original_config.id
        assert loaded_config.task_config.task_id == original_config.task_config.task_id
        assert loaded_config.exp_name == original_config.exp_name
        assert loaded_config.max_steps == original_config.max_steps
        assert loaded_config.output_dir == original_config.output_dir
        assert loaded_config.agent_config == original_config.agent_config

    def test_load_episode_config_not_found(self, tmp_dir):
        storage = FileStorage(tmp_dir)
        config_path = Path(tmp_dir) / "episode_configs" / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            storage.load_episode_config(config_path)

    def test_list_episode_configs(self, tmp_dir, mock_agent_config):
        from cube.task import TaskMetadata

        from cube_harness.episode import EpisodeConfig
        from tests.conftest import MockCubeTaskConfig

        storage = FileStorage(tmp_dir)
        for i in range(3):
            config = EpisodeConfig(
                id=i,
                agent_config=mock_agent_config,
                task_config=MockCubeTaskConfig(metadata=TaskMetadata(id=f"task_{i}")),
                exp_name="test_exp",
                output_dir=tmp_dir,
                max_steps=100,
            )
            storage.save_episode_config(config)

        config_files = storage.list_episode_configs()
        assert len(config_files) == 3
        for config_file in config_files:
            assert config_file.exists()
            assert config_file.name == "episode_config.json"
            # Parent dir name is the trajectory_id: {task_id}_ep{id}
            assert config_file.parent.name.startswith("task_")
            assert "_ep" in config_file.parent.name

    def test_list_episode_configs_empty_directory(self, tmp_dir):
        storage = FileStorage(tmp_dir)
        config_files = storage.list_episode_configs()
        assert config_files == []

    def test_episode_config_filename_parsing(self, tmp_dir, mock_agent_config):
        from cube.task import TaskMetadata

        from cube_harness.episode import EpisodeConfig
        from tests.conftest import MockCubeTaskConfig

        storage = FileStorage(tmp_dir)
        config = EpisodeConfig(
            id=10,
            agent_config=mock_agent_config,
            task_config=MockCubeTaskConfig(metadata=TaskMetadata(id="task_with_underscores_123")),
            exp_name="test_exp",
            output_dir=tmp_dir,
            max_steps=100,
        )
        storage.save_episode_config(config)

        config_path = Path(tmp_dir) / "episodes" / "task_with_underscores_123_ep10" / "episode_config.json"
        assert config_path.exists()

        loaded = storage.load_episode_config(config_path)
        assert loaded.id == 10
        assert loaded.task_config.task_id == "task_with_underscores_123"


class TestFileStorageOverwrite:
    def test_save_trajectory_raises_on_duplicate(self, tmp_dir) -> None:
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        storage.save_trajectory(traj)

        storage2 = FileStorage(tmp_dir)
        with pytest.raises(FileExistsError, match="task_1_ep0"):
            storage2.save_trajectory(traj)

    def test_save_trajectory_allows_resave_same_session(self, tmp_dir) -> None:
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        storage.save_trajectory(traj)
        traj.end_time = 999.0
        storage.save_trajectory(traj)

    def test_save_trajectory_archives_on_overwrite(self, tmp_dir) -> None:
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        obs = Observation.from_text("old data")
        env_out = EnvironmentOutput(obs=obs, reward=0.5)
        traj.steps.append(TrajectoryStep(output=env_out))
        storage.save_trajectory(traj)

        episodes_dir = Path(tmp_dir) / "episodes"

        storage2 = FileStorage(tmp_dir)
        traj2 = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        storage2.save_trajectory(traj2, allow_overwrite=True)

        archived = [d for d in episodes_dir.iterdir() if ".archived_" in d.name]
        assert len(archived) == 1

        current = episodes_dir / "task_1_ep0"
        assert current.exists()
        assert (current / "episode.metadata.json").exists()

    def test_save_trajectory_overwrite_false_is_default(self, tmp_dir) -> None:
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        storage.save_trajectory(traj)

        storage2 = FileStorage(tmp_dir)
        with pytest.raises(FileExistsError):
            storage2.save_trajectory(traj)

    def test_archive_preserves_episode_config_for_retry(
        self, tmp_dir, mock_agent_config, mock_cube_task_config
    ) -> None:
        """Archiving an episode dir must leave episode_config.json behind so the
        retry pipeline (list_episode_configs / _episode_config_dirs) still finds
        the task. The config is written once at experiment prep and is not
        re-saved on retry.
        """
        from cube_harness.episode import EpisodeConfig

        storage = FileStorage(tmp_dir)
        episode_config = EpisodeConfig(
            id=0,
            agent_config=mock_agent_config,
            task_config=mock_cube_task_config,
            exp_name="test_exp",
            output_dir=tmp_dir,
            max_steps=100,
        )
        storage.save_episode_config(episode_config)
        traj_id = f"{mock_cube_task_config.task_id}_ep0"
        traj = Trajectory(id=traj_id, metadata={"task_id": mock_cube_task_config.task_id, "agent_name": "A"})
        storage.save_trajectory(traj)

        ep_dir = storage._episode_dir(traj_id)
        storage._archive_episode(ep_dir)

        episodes_dir = Path(tmp_dir) / "episodes"
        archived = [d for d in episodes_dir.iterdir() if ".archived_" in d.name]
        assert len(archived) == 1
        assert (archived[0] / "episode_config.json").exists()
        assert ep_dir.exists()
        assert (ep_dir / "episode_config.json").exists()
        assert ep_dir in list(storage._episode_config_dirs())

    def test_archive_no_config_leaves_no_dir(self, tmp_dir) -> None:
        """If the episode dir has no episode_config.json (e.g. legacy run), archive
        should still rename the dir without creating an empty replacement.
        """
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        storage.save_trajectory(traj)

        ep_dir = storage._episode_dir("task_1_ep0")
        storage._archive_episode(ep_dir)

        assert not ep_dir.exists()
        episodes_dir = Path(tmp_dir) / "episodes"
        archived = [d for d in episodes_dir.iterdir() if ".archived_" in d.name]
        assert len(archived) == 1


class TestLoadTrajectoryMetadata:
    def test_load_metadata_returns_trajectory_with_no_steps(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"agent_name": "A", "task_id": "task_1"})
        traj.steps.append(TrajectoryStep(output=EnvironmentOutput(obs=Observation.from_text("obs"))))
        storage.save_trajectory(traj)

        loaded = storage.load_trajectory_metadata("task_1_ep0")
        assert loaded.id == "task_1_ep0"
        assert loaded.metadata == {"agent_name": "A", "task_id": "task_1"}
        assert loaded.steps == []

    def test_load_metadata_preserves_timing_and_reward_info(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        traj = Trajectory(
            id="task_1_ep0",
            metadata={"task_id": "task_1", "agent_name": "A"},
            start_time=0.0,
            end_time=5.0,
            reward_info={"reward": 1.0},
        )
        storage.save_trajectory(traj)

        loaded = storage.load_trajectory_metadata("task_1_ep0")
        assert loaded.start_time == 0.0
        assert loaded.end_time == 5.0
        assert loaded.reward_info == {"reward": 1.0}

    def test_load_metadata_raises_for_missing_trajectory(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        with pytest.raises(FileNotFoundError, match="Trajectory metadata not found"):
            storage.load_trajectory_metadata("nonexistent")

    def test_load_all_metadata_returns_stubs_with_empty_steps(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        for i in range(3):
            traj = Trajectory(id=f"task_{i}_ep0", metadata={"task_id": f"task_{i}", "agent_name": "A"})
            traj.steps.append(TrajectoryStep(output=EnvironmentOutput(obs=Observation.from_text("obs"))))
            storage.save_trajectory(traj)

        stubs = storage.load_all_trajectory_metadata()
        assert len(stubs) == 3
        assert all(t.steps == [] for t in stubs)
        ids = {t.id for t in stubs}
        assert ids == {"task_0_ep0", "task_1_ep0", "task_2_ep0"}

    def test_load_all_metadata_empty_directory(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        assert storage.load_all_trajectory_metadata() == []

    def test_load_all_metadata_skips_archived(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        storage.save_trajectory(traj)

        storage2 = FileStorage(tmp_dir)
        storage2.save_trajectory(traj, allow_overwrite=True)

        stubs = storage.load_all_trajectory_metadata()
        assert len(stubs) == 1
        assert stubs[0].id == "task_1_ep0"

    def test_metadata_stub_can_be_upgraded_to_full_trajectory(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        obs = Observation.from_text("Goal: click the button")
        traj.steps.append(TrajectoryStep(output=EnvironmentOutput(obs=obs, reward=1.0, done=True)))
        storage.save_trajectory(traj)

        stub = storage.load_trajectory_metadata("task_1_ep0")
        assert stub.steps == []

        full = storage.load_trajectory("task_1_ep0")
        assert len(full.steps) == 1
        assert isinstance(full.steps[0].output, EnvironmentOutput)
        assert full.steps[0].output.reward == 1.0


class TestSummaryStats:
    def _make_trajectory_with_stats(self, traj_id: str = "task_1_ep0") -> Trajectory:
        obs = Observation.from_text("obs")
        traj = Trajectory(
            id=traj_id,
            metadata={"task_id": "task_1", "agent_name": "A"},
            start_time=0.0,
            end_time=10.0,
            reward_info={"reward": 1.0, "done": True},
            summary_stats={
                "n_env_steps": 5,
                "n_agent_steps": 4,
                "total_actions": 8,
                "total_llm_calls": 4,
                "duration": 10.0,
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "cached_tokens": 50,
                "cache_creation_tokens": 0,
                "cost": 0.05,
                "final_reward": 1.0,
            },
        )
        traj.steps.append(TrajectoryStep(output=EnvironmentOutput(obs=obs, reward=1.0, done=True)))
        return traj

    def test_summary_stats_persisted_in_metadata(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        traj = self._make_trajectory_with_stats()
        storage.save_trajectory(traj)

        metadata_path = tmp_dir / "episodes" / "task_1_ep0" / "episode.metadata.json"
        with open(metadata_path) as f:
            data = json.load(f)

        assert data["summary_stats"] is not None
        assert data["summary_stats"]["n_env_steps"] == 5
        assert data["summary_stats"]["cost"] == 0.05
        assert data["summary_stats"]["final_reward"] == 1.0

    def test_summary_stats_loaded_in_metadata_stub(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        traj = self._make_trajectory_with_stats()
        storage.save_trajectory(traj)

        stub = storage.load_trajectory_metadata("task_1_ep0")
        assert stub.steps == []
        assert stub.summary_stats is not None
        assert stub.summary_stats["prompt_tokens"] == 1000
        assert stub.summary_stats["duration"] == 10.0

    def test_backward_compat_no_summary_stats(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        storage.save_trajectory(traj)

        metadata_path = tmp_dir / "episodes" / "task_1_ep0" / "episode.metadata.json"
        with open(metadata_path) as f:
            data = json.load(f)
        del data["summary_stats"]
        with open(metadata_path, "w") as f:
            json.dump(data, f)

        loaded = storage.load_trajectory_metadata("task_1_ep0")
        assert loaded.summary_stats is None

    def test_experiment_summary_created(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        traj = self._make_trajectory_with_stats()
        storage.save_trajectory(traj)
        storage.update_experiment_summary(traj)

        summary_path = tmp_dir / "experiment_summary.json"
        assert summary_path.exists()
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary["n_episodes"] == 1
        assert summary["n_completed"] == 1
        assert summary["n_errored"] == 0
        assert summary["total_prompt_tokens"] == 1000
        assert summary["total_cost"] == 0.05

    def test_experiment_summary_accumulates(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        for i in range(3):
            traj = self._make_trajectory_with_stats(traj_id=f"task_1_ep{i}")
            storage.save_trajectory(traj)
            storage.update_experiment_summary(traj)

        with open(tmp_dir / "experiment_summary.json") as f:
            summary = json.load(f)
        assert summary["n_episodes"] == 3
        assert summary["n_completed"] == 3
        assert summary["total_prompt_tokens"] == 3000
        assert summary["total_cost"] == pytest.approx(0.15)


class TestEpisodeSummary:
    def test_summary_appended_per_step(self, tmp_dir, sample_env_output, sample_agent_output):
        from cube_harness.summary import SummaryProcessor

        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        traj.steps.append(TrajectoryStep(output=sample_env_output))
        storage.save_trajectory(traj)

        ep_dir = storage._episode_dir("task_1_ep0")
        proc = SummaryProcessor(ep_dir)
        proc.on_step(0, traj.steps[0])

        step1 = TrajectoryStep(output=sample_agent_output)
        storage.save_step(step1, "task_1_ep0", 1)
        proc.on_step(1, step1)

        step2 = TrajectoryStep(output=sample_env_output)
        storage.save_step(step2, "task_1_ep0", 2)
        proc.on_step(2, step2)

        summary_path = ep_dir / "episode_summary.jsonl"
        assert summary_path.exists()
        with open(summary_path) as f:
            lines = f.readlines()
        assert len(lines) == 3

        last = json.loads(lines[-1])
        assert last["n_env_steps"] == 2
        assert last["n_agent_steps"] == 1
        assert last["turn"] == 2
        assert last["status"] == "running"
        assert "tokens" in last
        assert "cost_usd" in last

    def test_summary_tracks_running_totals(self, tmp_dir):
        from cube_harness.summary import SummaryProcessor

        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        obs = Observation.from_text("obs")
        traj.steps.append(TrajectoryStep(output=EnvironmentOutput(obs=obs, reward=0.0)))
        storage.save_trajectory(traj)

        ep_dir = storage._episode_dir("task_1_ep0")
        proc = SummaryProcessor(ep_dir)
        proc.on_step(0, traj.steps[0])

        llm_call = LLMCall(
            id="c1",
            llm_config=LLMConfig(model_name="test"),
            prompt=Prompt(messages=[{"role": "user", "content": "hi"}]),
            output=Message(role="assistant", content="hello"),
        )
        llm_call.usage.prompt_tokens = 100
        llm_call.usage.completion_tokens = 50
        llm_call.usage.cost = 0.01
        agent_step = TrajectoryStep(
            output=AgentOutput(
                actions=[Action(name="click", arguments={})],
                llm_calls=[llm_call],
            )
        )
        storage.save_step(agent_step, "task_1_ep0", 1)
        proc.on_step(1, agent_step)

        summary_path = ep_dir / "episode_summary.jsonl"
        with open(summary_path) as f:
            lines = f.readlines()
        last = json.loads(lines[-1])
        assert last["prompt_tokens"] == 100
        assert last["completion_tokens"] == 50
        assert last["tokens"] == 150
        assert last["cost_usd"] == pytest.approx(0.01)
        assert last["total_actions"] == 1
        assert last["total_llm_calls"] == 1


class TestMsgpackZstFormat:
    def test_step_files_are_binary(self, tmp_dir, sample_env_output):
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        traj.steps.append(TrajectoryStep(output=sample_env_output))
        storage.save_trajectory(traj)

        steps_dir = Path(tmp_dir) / "episodes" / "task_1_ep0" / "steps"
        step_files = list(steps_dir.iterdir())
        assert len(step_files) == 1
        assert step_files[0].name.endswith(".msgpack.zst")
        raw = step_files[0].read_bytes()
        assert len(raw) > 0
        assert raw[:4] != b'{"_t'

    def test_compression_reduces_size(self, tmp_dir):
        storage = FileStorage(tmp_dir)
        llm_call = LLMCall(
            id="call_1",
            llm_config=LLMConfig(model_name="gpt-4"),
            prompt=Prompt(
                messages=[{"role": "system", "content": "You are helpful. " * 200}],
                tools=[{"type": "function", "function": {"name": f"tool_{i}", "parameters": {}}} for i in range(20)],
            ),
            output=Message(role="assistant", content="I will help you. " * 100),
        )
        agent_output = AgentOutput(
            actions=[Action(name="click", arguments={"element": "btn"})],
            llm_calls=[llm_call],
        )
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        traj.steps.append(TrajectoryStep(output=agent_output))
        storage.save_trajectory(traj)

        step_file = Path(tmp_dir) / "episodes" / "task_1_ep0" / "steps" / "000_act.msgpack.zst"
        compressed_size = step_file.stat().st_size
        json_size = len(traj.steps[0].model_dump_json(serialize_as_any=True).encode())
        assert compressed_size < json_size * 0.5

    def test_random_step_access(self, tmp_dir, sample_env_output, sample_agent_output):
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        for i in range(5):
            output = sample_env_output if i % 2 == 0 else sample_agent_output
            traj.steps.append(TrajectoryStep(output=output, start_time=float(i), end_time=float(i + 1)))
        storage.save_trajectory(traj)

        storage2 = FileStorage(tmp_dir)
        step3 = storage2.load_step("task_1_ep0", 3)
        assert isinstance(step3.output, AgentOutput)
        assert step3.start_time == 3.0

        step0 = storage2.load_step("task_1_ep0", 0)
        assert isinstance(step0.output, EnvironmentOutput)
        assert step0.start_time == 0.0

    def test_random_step_access_out_of_range(self, tmp_dir, sample_env_output):
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        traj.steps.append(TrajectoryStep(output=sample_env_output))
        storage.save_trajectory(traj)

        with pytest.raises(IndexError):
            storage.load_step("task_1_ep0", 5)

    def test_random_step_access_not_found(self, tmp_dir):
        storage = FileStorage(tmp_dir)
        with pytest.raises(FileNotFoundError):
            storage.load_step("nonexistent", 0)


class TestLazyLoaders:
    def test_episode_result_lazy_metadata(self, tmp_dir, sample_env_output):
        from cube_harness.results import EpisodeResult

        storage = FileStorage(tmp_dir)
        traj = Trajectory(
            id="task_1_ep0",
            metadata={"task_id": "task_1", "agent_name": "A"},
            summary_stats={"final_reward": 1.0, "cost": 0.05},
        )
        traj.steps.append(TrajectoryStep(output=sample_env_output))
        storage.save_trajectory(traj)

        ep_dir = storage._episode_dir("task_1_ep0")
        result = EpisodeResult(ep_dir, storage)

        assert result.metadata().id == "task_1_ep0"
        assert result.summary_stats()["final_reward"] == 1.0
        assert result.metadata().steps == []

    def test_episode_result_random_access(self, tmp_dir, sample_env_output, sample_agent_output):
        from cube_harness.results import EpisodeResult

        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        traj.steps.append(TrajectoryStep(output=sample_env_output, start_time=0.0, end_time=0.1))
        traj.steps.append(TrajectoryStep(output=sample_agent_output, start_time=0.1, end_time=0.2))
        traj.steps.append(TrajectoryStep(output=sample_env_output, start_time=0.2, end_time=0.3))
        storage.save_trajectory(traj)

        ep_dir = storage._episode_dir("task_1_ep0")
        result = EpisodeResult(ep_dir, storage)

        assert len(result) == 3
        step1 = result[1]
        assert isinstance(step1.output, AgentOutput)

    def test_episode_result_iteration(self, tmp_dir, sample_env_output, sample_agent_output):
        from cube_harness.results import EpisodeResult

        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        traj.steps.append(TrajectoryStep(output=sample_env_output))
        traj.steps.append(TrajectoryStep(output=sample_agent_output))
        storage.save_trajectory(traj)

        ep_dir = storage._episode_dir("task_1_ep0")
        result = EpisodeResult(ep_dir, storage)

        steps = list(result)
        assert len(steps) == 2
        assert isinstance(steps[0].output, EnvironmentOutput)
        assert isinstance(steps[1].output, AgentOutput)

    def test_experiment_result_listing(self, tmp_dir, sample_env_output):
        from cube_harness.results import ExperimentResult

        storage = FileStorage(tmp_dir)
        for i in range(3):
            traj = Trajectory(id=f"task_1_ep{i}", metadata={"task_id": "task_1", "agent_name": "A"})
            traj.steps.append(TrajectoryStep(output=sample_env_output))
            storage.save_trajectory(traj)

        result = ExperimentResult(tmp_dir)
        assert len(result.episodes()) == 3
        for traj_id, episode in result.episodes().items():
            assert episode.metadata().steps == []

    def test_experiment_result_summary(self, tmp_dir, sample_env_output):
        from cube_harness.results import ExperimentResult

        storage = FileStorage(tmp_dir)
        traj = Trajectory(
            id="task_1_ep0",
            metadata={"task_id": "task_1", "agent_name": "A"},
            summary_stats={"final_reward": 1.0, "prompt_tokens": 500, "completion_tokens": 100, "cost": 0.02},
        )
        traj.steps.append(TrajectoryStep(output=sample_env_output))
        storage.save_trajectory(traj)
        storage.update_experiment_summary(traj)

        result = ExperimentResult(tmp_dir)
        assert result.summary() is not None
        assert result.summary().n_episodes == 1

    def test_episode_result_load_full(self, tmp_dir, sample_env_output, sample_agent_output):
        from cube_harness.results import EpisodeResult

        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        traj.steps.append(TrajectoryStep(output=sample_env_output))
        traj.steps.append(TrajectoryStep(output=sample_agent_output))
        storage.save_trajectory(traj)

        ep_dir = storage._episode_dir("task_1_ep0")
        result = EpisodeResult(ep_dir, storage)
        full = result.load_full()
        assert len(full.steps) == 2


class TestV1BackwardCompat:
    def _create_v1_trajectory(self, tmp_dir: Path, traj_id: str = "test_traj") -> None:
        traj_dir = tmp_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "_type": "cube_harness.core.Trajectory",
            "id": traj_id,
            "metadata": {"task_id": "task_1", "agent_name": "A"},
            "start_time": 0.0,
            "end_time": 1.0,
            "reward_info": {},
            "summary_stats": None,
        }
        with open(traj_dir / f"{traj_id}.metadata.json", "w") as f:
            json.dump(metadata, f)

        obs = Observation.from_text("hello")
        env_step = TrajectoryStep(output=EnvironmentOutput(obs=obs, reward=0.5))
        with open(traj_dir / f"{traj_id}.jsonl", "w") as f:
            f.write(env_step.model_dump_json(serialize_as_any=True) + "\n")

    def test_v1_load_trajectory(self, tmp_dir: Path) -> None:
        self._create_v1_trajectory(tmp_dir, "test_traj")

        storage = FileStorage(tmp_dir)
        loaded = storage.load_trajectory("test_traj")

        assert loaded.id == "test_traj"
        assert loaded.metadata["task_id"] == "task_1"
        assert len(loaded.steps) == 1
        assert isinstance(loaded.steps[0].output, EnvironmentOutput)
        assert loaded.steps[0].output.reward == 0.5

    def test_v1_load_metadata(self, tmp_dir: Path) -> None:
        self._create_v1_trajectory(tmp_dir, "test_traj")

        storage = FileStorage(tmp_dir)
        loaded = storage.load_trajectory_metadata("test_traj")

        assert loaded.id == "test_traj"
        assert loaded.steps == []

    def test_v1_load_all_trajectories(self, tmp_dir: Path) -> None:
        for i in range(2):
            self._create_v1_trajectory(tmp_dir, f"traj_{i}")

        storage = FileStorage(tmp_dir)
        result = storage.load_all_trajectories()
        assert len(result) == 2
        ids = {t.id for t in result}
        assert ids == {"traj_0", "traj_1"}

    def test_v1_load_all_metadata(self, tmp_dir: Path) -> None:
        for i in range(2):
            self._create_v1_trajectory(tmp_dir, f"traj_{i}")

        storage = FileStorage(tmp_dir)
        stubs = storage.load_all_trajectory_metadata()
        assert len(stubs) == 2
        assert all(t.steps == [] for t in stubs)

    def test_v1_list_trajectory_ids(self, tmp_dir: Path) -> None:
        for i in range(2):
            self._create_v1_trajectory(tmp_dir, f"traj_{i}")

        storage = FileStorage(tmp_dir)
        ids = storage.list_trajectory_ids()
        assert set(ids) == {"traj_0", "traj_1"}

    def test_v1_with_llm_call_refs(self, tmp_dir: Path) -> None:
        traj_dir = tmp_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        llm_calls_dir = tmp_dir / "llm_calls"
        llm_calls_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "_type": "cube_harness.core.Trajectory",
            "id": "t1",
            "metadata": {"task_id": "task_1"},
            "start_time": None,
            "end_time": None,
            "reward_info": {},
            "summary_stats": None,
        }
        with open(traj_dir / "t1.metadata.json", "w") as f:
            json.dump(metadata, f)

        llm_call = LLMCall(
            id="call_abc",
            llm_config=LLMConfig(model_name="test-model"),
            prompt=Prompt(messages=[{"role": "user", "content": "hi"}]),
            output=Message(role="assistant", content="hello"),
        )
        call_path = llm_calls_dir / "t1_step000_call_abc.json"
        with open(call_path, "w") as f:
            f.write(llm_call.model_dump_json(indent=2))

        step_data = {
            "_type": "cube_harness.core.TrajectoryStep",
            "output": {
                "_type": "cube_harness.core.AgentOutput",
                "actions": [],
                "llm_calls": [{"llm_call_id": "call_abc"}],
                "error": None,
                "profiling": {},
                "thoughts": None,
            },
            "start_time": None,
            "end_time": None,
        }
        with open(traj_dir / "t1.jsonl", "w") as f:
            f.write(json.dumps(step_data) + "\n")

        storage = FileStorage(tmp_dir)
        loaded = storage.load_trajectory("t1")
        assert len(loaded.steps) == 1
        assert isinstance(loaded.steps[0].output, AgentOutput)
        assert len(loaded.steps[0].output.llm_calls) == 1
        assert loaded.steps[0].output.llm_calls[0].id == "call_abc"
        assert loaded.steps[0].output.llm_calls[0].output.content == "hello"

    def test_mixed_v1_and_v2(self, tmp_dir: Path, sample_env_output) -> None:
        self._create_v1_trajectory(tmp_dir, "old_traj")

        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        traj.steps.append(TrajectoryStep(output=sample_env_output))
        storage.save_trajectory(traj)

        all_trajs = storage.load_all_trajectories()
        assert len(all_trajs) == 2
        ids = {t.id for t in all_trajs}
        assert ids == {"old_traj", "task_1_ep0"}


class TestEpisodeSummaryStatus:
    def test_final_line_written_on_complete(self, tmp_dir, sample_env_output):
        from cube_harness.summary import EpisodeStatus, StepSummary, SummaryProcessor

        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        traj.steps.append(TrajectoryStep(output=sample_env_output))
        storage.save_trajectory(traj)

        ep_dir = storage._episode_dir("task_1_ep0")
        proc = SummaryProcessor(ep_dir)
        proc.on_step(0, traj.steps[0])
        proc.on_episode_complete(traj, storage)

        summary_path = ep_dir / "episode_summary.jsonl"
        lines = summary_path.read_text().splitlines()
        assert len(lines) == 2

        running = StepSummary.model_validate_json(lines[0])
        assert running.status == EpisodeStatus.RUNNING
        assert running.turn == 0

        final = StepSummary.model_validate_json(lines[1])
        assert final.status == EpisodeStatus.DONE
        assert final.turn == -1

    def test_failed_status_on_error(self, tmp_dir, sample_env_output):
        from cube_harness.summary import EpisodeStatus, StepSummary, SummaryProcessor

        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        traj.steps.append(TrajectoryStep(output=sample_env_output))

        from cube.core import StepError

        error_output = AgentOutput(error=StepError(error_type="RuntimeError", exception_str="boom", stack_trace=""))
        traj.steps.append(TrajectoryStep(output=error_output))
        storage.save_trajectory(traj)

        ep_dir = storage._episode_dir("task_1_ep0")
        proc = SummaryProcessor(ep_dir)
        proc.on_step(0, traj.steps[0])
        proc.on_step(1, traj.steps[1])
        proc.on_episode_complete(traj, storage)

        lines = (ep_dir / "episode_summary.jsonl").read_text().splitlines()
        final = StepSummary.model_validate_json(lines[-1])
        assert final.status == EpisodeStatus.FAILED


class TestFailureTextInjection:
    def test_load_all_metadata_injects_failure_text(self, tmp_dir: Path) -> None:
        """load_all_trajectory_metadata injects _failure_text when failure.txt exists and no end_time."""
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1"})
        storage.save_trajectory(traj)
        (storage._episode_dir("task_1_ep0") / "failure.txt").write_text("Ray actor died")

        trajs = storage.load_all_trajectory_metadata()
        t = next(t for t in trajs if t.id == "task_1_ep0")
        assert t.metadata.get("_failure_text") == "Ray actor died"

    def test_load_all_metadata_no_injection_when_complete(self, tmp_dir: Path) -> None:
        """_failure_text is NOT injected when end_time is set (trajectory completed normally)."""
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1"}, end_time=1234567890.0)
        storage.save_trajectory(traj)
        (storage._episode_dir("task_1_ep0") / "failure.txt").write_text("stale error")

        trajs = storage.load_all_trajectory_metadata()
        t = next(t for t in trajs if t.id == "task_1_ep0")
        assert "_failure_text" not in t.metadata

    def test_load_trajectory_injects_failure_text(self, tmp_dir: Path) -> None:
        """load_trajectory (full load) also injects _failure_text."""
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1"})
        storage.save_trajectory(traj)
        (storage._episode_dir("task_1_ep0") / "failure.txt").write_text("crash trace")

        loaded = storage.load_trajectory("task_1_ep0")
        assert loaded.metadata.get("_failure_text") == "crash trace"

    def test_list_ids_with_mtime_uses_failure_txt_mtime(self, tmp_dir: Path) -> None:
        """list_trajectory_ids_with_mtime returns failure.txt mtime when it's newer."""
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1"})
        storage.save_trajectory(traj)
        time.sleep(0.01)  # ensure different mtime
        failure_path = storage._episode_dir("task_1_ep0") / "failure.txt"
        failure_path.write_text("crash")

        mtimes = storage.list_trajectory_ids_with_mtime()
        assert mtimes["task_1_ep0"] >= failure_path.stat().st_mtime


class TestEpisodeResultAPI:
    def _make_episode(self, tmp_dir, sample_env_output, sample_agent_output):
        from cube_harness.summary import SummaryProcessor

        storage = FileStorage(tmp_dir)
        traj = Trajectory(
            id="task_1_ep0",
            metadata={"task_id": "task_1", "agent_name": "A"},
            summary_stats={"final_reward": 1.0, "cost": 0.05, "n_env_steps": 2, "n_agent_steps": 1},
        )
        traj.steps.append(TrajectoryStep(output=sample_env_output, start_time=0.0, end_time=0.1))
        traj.steps.append(TrajectoryStep(output=sample_agent_output, start_time=0.1, end_time=0.2))
        traj.steps.append(TrajectoryStep(output=sample_env_output, start_time=0.2, end_time=0.3))
        storage.save_trajectory(traj)

        ep_dir = storage._episode_dir("task_1_ep0")
        proc = SummaryProcessor(ep_dir)
        for i, step in enumerate(traj.steps):
            proc.on_step(i, step)
        proc.on_episode_complete(traj, storage)
        return storage, ep_dir

    def test_status_done(self, tmp_dir, sample_env_output, sample_agent_output):
        from cube_harness.results import EpisodeResult
        from cube_harness.summary import EpisodeStatus

        storage, ep_dir = self._make_episode(tmp_dir, sample_env_output, sample_agent_output)
        result = EpisodeResult(ep_dir, storage)
        assert result.status() == EpisodeStatus.DONE

    def test_status_pending_no_summary(self, tmp_dir, sample_env_output):
        from cube_harness.results import EpisodeResult
        from cube_harness.summary import EpisodeStatus

        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1", "agent_name": "A"})
        traj.steps.append(TrajectoryStep(output=sample_env_output))
        storage.save_trajectory(traj)

        ep_dir = storage._episode_dir("task_1_ep0")
        result = EpisodeResult(ep_dir, storage)
        assert result.status() == EpisodeStatus.PENDING

    def test_summary_returns_step_summary_models(self, tmp_dir, sample_env_output, sample_agent_output):
        from cube_harness.results import EpisodeResult
        from cube_harness.summary import StepSummary

        storage, ep_dir = self._make_episode(tmp_dir, sample_env_output, sample_agent_output)
        result = EpisodeResult(ep_dir, storage)
        summary = result.summary()
        assert len(summary) == 4
        assert all(isinstance(s, StepSummary) for s in summary)

    def test_n_turns(self, tmp_dir, sample_env_output, sample_agent_output):
        from cube_harness.results import EpisodeResult

        storage, ep_dir = self._make_episode(tmp_dir, sample_env_output, sample_agent_output)
        result = EpisodeResult(ep_dir, storage)
        assert result.n_turns() == 2

    def test_get_obs_and_get_act(self, tmp_dir, sample_env_output, sample_agent_output):
        from cube_harness.results import EpisodeResult

        storage, ep_dir = self._make_episode(tmp_dir, sample_env_output, sample_agent_output)
        result = EpisodeResult(ep_dir, storage)

        obs_step = result.get_obs(0)
        assert isinstance(obs_step.output, EnvironmentOutput)

        act_step = result.get_act(1)
        assert isinstance(act_step.output, AgentOutput)

    def test_get_obs_not_found(self, tmp_dir, sample_env_output, sample_agent_output):
        from cube_harness.results import EpisodeResult

        storage, ep_dir = self._make_episode(tmp_dir, sample_env_output, sample_agent_output)
        result = EpisodeResult(ep_dir, storage)
        with pytest.raises(FileNotFoundError):
            result.get_obs(99)

    def test_get_exp_record(self, tmp_dir, sample_env_output, sample_agent_output):
        from cube_harness.results import EpisodeRecord, EpisodeResult
        from cube_harness.summary import EpisodeStatus

        storage, ep_dir = self._make_episode(tmp_dir, sample_env_output, sample_agent_output)
        result = EpisodeResult(ep_dir, storage)
        record = result.get_exp_record()
        assert isinstance(record, EpisodeRecord)
        assert record.trajectory_id == "task_1_ep0"
        assert record.status == EpisodeStatus.DONE


class TestExperimentResultGetRecords:
    def test_get_records(self, tmp_dir, sample_env_output):
        from cube_harness.results import EpisodeRecord, ExperimentResult
        from cube_harness.summary import SummaryProcessor

        storage = FileStorage(tmp_dir)
        for i in range(3):
            traj = Trajectory(id=f"task_1_ep{i}", metadata={"task_id": "task_1", "agent_name": "A"})
            traj.steps.append(TrajectoryStep(output=sample_env_output))
            storage.save_trajectory(traj)
            ep_dir = storage._episode_dir(f"task_1_ep{i}")
            proc = SummaryProcessor(ep_dir)
            proc.on_step(0, traj.steps[0])
            proc.on_episode_complete(traj, storage)

        result = ExperimentResult(tmp_dir)
        records = result.get_records()
        assert len(records) == 3
        assert all(isinstance(r, EpisodeRecord) for r in records)

    def test_iter_records(self, tmp_dir, sample_env_output) -> None:
        from cube_harness.results import EpisodeRecord, ExperimentResult
        from cube_harness.summary import SummaryProcessor

        storage = FileStorage(tmp_dir)
        for i in range(2):
            traj = Trajectory(id=f"task_1_ep{i}", metadata={"task_id": "task_1", "agent_name": "A"})
            traj.steps.append(TrajectoryStep(output=sample_env_output))
            storage.save_trajectory(traj)
            ep_dir = storage._episode_dir(f"task_1_ep{i}")
            proc = SummaryProcessor(ep_dir)
            proc.on_step(0, traj.steps[0])
            proc.on_episode_complete(traj, storage)

        result = ExperimentResult(tmp_dir)
        records = list(result.iter_records())
        assert len(records) == 2
        assert all(isinstance(r, EpisodeRecord) for r in records)

    def test_experiment_result_iter(self, tmp_dir, sample_env_output) -> None:
        from cube_harness.results import EpisodeResult, ExperimentResult
        from cube_harness.summary import SummaryProcessor

        storage = FileStorage(tmp_dir)
        for i in range(2):
            traj = Trajectory(id=f"task_1_ep{i}", metadata={"task_id": "task_1", "agent_name": "A"})
            traj.steps.append(TrajectoryStep(output=sample_env_output))
            storage.save_trajectory(traj)
            ep_dir = storage._episode_dir(f"task_1_ep{i}")
            proc = SummaryProcessor(ep_dir)
            proc.on_step(0, traj.steps[0])
            proc.on_episode_complete(traj, storage)

        result = ExperimentResult(tmp_dir)
        episodes = list(result)
        assert len(episodes) == 2
        assert all(isinstance(ep, EpisodeResult) for ep in episodes)


class TestEpisodeStatusIO:
    """Tests for status.json atomic write/read on FileStorage."""

    def test_write_then_read_roundtrip(self, tmp_dir) -> None:
        from cube_harness.episode_status import EpisodeStatus

        storage = FileStorage(tmp_dir)
        status = EpisodeStatus(
            status="COMPLETED",
            task_id="t1",
            episode_id=0,
            started_at=1.0,
            ended_at=2.0,
            last_heartbeat_at=2.0,
            current_step=3,
            reward=1.0,
        )
        storage.write_episode_status("t1_ep0", status)

        loaded = storage.read_episode_status("t1_ep0")
        assert loaded is not None
        assert loaded.status == "COMPLETED"
        assert loaded.task_id == "t1"
        assert loaded.reward == 1.0
        assert loaded.current_step == 3

    def test_read_missing_returns_none(self, tmp_dir) -> None:
        storage = FileStorage(tmp_dir)
        assert storage.read_episode_status("does_not_exist") is None

    def test_atomic_write_no_partial_file(self, tmp_dir) -> None:
        """Writing always goes via .tmp + os.replace — a partial status.json is never observed."""
        from cube_harness.episode_status import STATUS_FILENAME, EpisodeStatus

        storage = FileStorage(tmp_dir)
        status_path = storage._episode_status_path("t1_ep0")

        status = EpisodeStatus(status="RUNNING", task_id="t1", episode_id=0, started_at=1.0)
        storage.write_episode_status("t1_ep0", status)

        siblings = list(status_path.parent.iterdir())
        assert STATUS_FILENAME in [s.name for s in siblings]
        assert not any(s.name.endswith(".tmp") for s in siblings)

        status.status = "COMPLETED"
        storage.write_episode_status("t1_ep0", status)
        assert not any(s.name.endswith(".tmp") for s in status_path.parent.iterdir())
        loaded = storage.read_episode_status("t1_ep0")
        assert loaded is not None and loaded.status == "COMPLETED"

    def test_list_episode_statuses(self, tmp_dir) -> None:
        """list_episode_statuses returns statuses keyed by trajectory_id (skipping dirs without configs)."""
        from cube_harness.episode_status import EpisodeStatus

        storage = FileStorage(tmp_dir)
        for tid, st in [("t1_ep0", "COMPLETED"), ("t2_ep0", "FAILED")]:
            ep_dir = storage._episode_dir(tid)
            ep_dir.mkdir(parents=True, exist_ok=True)
            (ep_dir / "episode_config.json").write_text("{}")
            storage.write_episode_status(
                tid, EpisodeStatus(status=st, task_id=tid.split("_ep")[0], episode_id=0, started_at=0.0)
            )
        statuses = storage.list_episode_statuses()
        assert set(statuses.keys()) == {"t1_ep0", "t2_ep0"}
        assert statuses["t1_ep0"].status == "COMPLETED"
        assert statuses["t2_ep0"].status == "FAILED"

    def test_corrupt_status_returns_none(self, tmp_dir) -> None:
        """A malformed status.json is treated as missing rather than raising."""
        from cube_harness.episode_status import STATUS_FILENAME

        storage = FileStorage(tmp_dir)
        ep_dir = storage._episode_dir("t1_ep0")
        ep_dir.mkdir(parents=True, exist_ok=True)
        (ep_dir / STATUS_FILENAME).write_text("not valid json")
        assert storage.read_episode_status("t1_ep0") is None

    def test_unknown_fields_ignored_for_forward_compat(self, tmp_dir) -> None:
        """A status.json from a future version with extra fields still loads cleanly."""
        from cube_harness.episode_status import STATUS_FILENAME

        storage = FileStorage(tmp_dir)
        ep_dir = storage._episode_dir("t1_ep0")
        ep_dir.mkdir(parents=True, exist_ok=True)
        raw = {
            "status": "COMPLETED",
            "task_id": "t1",
            "episode_id": 0,
            "started_at": 1.0,
            "ended_at": 2.0,
            "last_heartbeat_at": 2.0,
            "current_step": 0,
            "reward": 1.0,
            "had_step_errors": False,
            "error_type": None,
            "error_message": None,
            "retry_count": 0,
            "extra": {},
            "future_v2_field": "should be ignored",
            "another_future_field": 42,
        }
        (ep_dir / STATUS_FILENAME).write_text(json.dumps(raw))
        loaded = storage.read_episode_status("t1_ep0")
        assert loaded is not None
        assert loaded.status == "COMPLETED"
        assert loaded.reward == 1.0

    def test_archive_episode_renames_directory(self, tmp_dir: Path) -> None:
        """archive_episode moves the episode dir to <id>.archived_<ts>/ and makes it invisible to readers."""
        from cube_harness.episode_status import EpisodeStatus

        storage = FileStorage(tmp_dir)
        status = EpisodeStatus(status="FAILED", task_id="t1", episode_id=0, started_at=1.0)
        storage.write_episode_status("t1_ep0", status)

        episodes_dir = tmp_dir / "episodes"
        assert (episodes_dir / "t1_ep0").exists()

        storage.archive_episode("t1_ep0")

        # Original directory is gone.
        assert not (episodes_dir / "t1_ep0").exists()

        # An archived copy exists.
        archived = [d for d in episodes_dir.iterdir() if ".archived_" in d.name]
        assert len(archived) == 1

        # read_episode_status sees nothing (archived dir is excluded from _episode_dirs).
        assert storage.read_episode_status("t1_ep0") is None

    def test_archive_episode_noop_when_dir_missing(self, tmp_dir: Path) -> None:
        """archive_episode on a non-existent trajectory_id does not raise."""
        storage = FileStorage(tmp_dir)
        storage.archive_episode("nonexistent_ep0")  # should not raise


# ---------------------------------------------------------------------------
# TestInjectEpisodeStatus
# ---------------------------------------------------------------------------


class TestInjectEpisodeStatus:
    """FileStorage injects _episode_status (and related fields) from status.json into metadata."""

    def _write_episode(self, storage: FileStorage, traj_id: str, status: str, **kwargs: object) -> None:
        traj = Trajectory(id=traj_id, metadata={"task_id": "task_1", "agent_name": "test_agent"})
        storage.save_trajectory(traj)
        ep_status = EpisodeStatus(
            status=status,  # type: ignore[arg-type]
            task_id="task_1",
            episode_id=0,
            started_at=1.0,
            retry_count=kwargs.get("retry_count", 0),
            error_type=kwargs.get("error_type"),
            error_message=kwargs.get("error_message"),
        )
        storage.write_episode_status(traj_id, ep_status)

    def test_load_all_metadata_injects_episode_status(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        self._write_episode(storage, "task_1_ep0", "COMPLETED")
        trajs = storage.load_all_trajectory_metadata()
        t = next(t for t in trajs if t.id == "task_1_ep0")
        assert t.metadata.get("_episode_status") == "COMPLETED"

    def test_load_trajectory_injects_episode_status(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        self._write_episode(storage, "task_1_ep0", "RUNNING")
        traj = storage.load_trajectory("task_1_ep0")
        assert traj.metadata.get("_episode_status") == "RUNNING"

    def test_retry_count_injected(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        self._write_episode(storage, "task_1_ep0", "COMPLETED", retry_count=2)
        trajs = storage.load_all_trajectory_metadata()
        t = next(t for t in trajs if t.id == "task_1_ep0")
        assert t.metadata.get("_retry_count") == 2

    def test_error_fields_injected(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        self._write_episode(
            storage,
            "task_1_ep0",
            "FAILED",
            error_type="RuntimeError",
            error_message="OOM on GPU",
        )
        trajs = storage.load_all_trajectory_metadata()
        t = next(t for t in trajs if t.id == "task_1_ep0")
        assert t.metadata.get("_error_type") == "RuntimeError"
        assert t.metadata.get("_error_message") == "OOM on GPU"

    def test_no_injection_when_status_json_absent(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="task_1_ep0", metadata={"task_id": "task_1"})
        storage.save_trajectory(traj)
        trajs = storage.load_all_trajectory_metadata()
        t = next(t for t in trajs if t.id == "task_1_ep0")
        assert "_episode_status" not in t.metadata

    def test_episode_status_injected_into_missing_stubs(self, tmp_dir: Path) -> None:
        """load_missing_trajectory_stubs also injects _episode_status from status.json."""
        storage = FileStorage(tmp_dir)
        # Write an episode_config.json but no trajectory (simulates a queued-but-unstarted episode).
        ep_dir = tmp_dir / "episodes" / "task_1_ep0"
        ep_dir.mkdir(parents=True)
        (ep_dir / "episode_config.json").write_text('{"task_id": "task_1"}')
        ep_status = EpisodeStatus(status="QUEUED", task_id="task_1", episode_id=0, started_at=1.0)
        storage.write_episode_status("task_1_ep0", ep_status)

        stubs = storage.load_missing_trajectory_stubs()
        stub = next(s for s in stubs if s.id == "task_1_ep0")
        assert stub.metadata.get("_episode_status") == "QUEUED"

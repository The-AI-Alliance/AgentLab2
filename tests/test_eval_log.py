"""Tests for cube_harness.eval_log — Atlas EvalLog system."""

import json
import re
import tempfile
from pathlib import Path

import pytest
from cube.core import Content, EnvironmentOutput, Observation, StepError

from cube_harness.core import AgentOutput, Trajectory, TrajectoryStep
from cube_harness.eval_log import (
    AgentInfo,
    BenchmarkSubset,
    EpisodeRecord,
    EvalLibrary,
    EvalLog,
    ExperimentRecord,
    Findings,
    InvestigatorLLMConfig,
    UsageSummary,
    Verifier,
    _extract_error_type,
    _extract_llm_model,
    _extract_tool_names,
    _to_github_url,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env_output(reward: float = 0.0, done: bool = True, text: str = "Task: do it") -> EnvironmentOutput:
    obs = Observation(contents=[Content.from_data(text)])
    return EnvironmentOutput(obs=obs, reward=reward, done=done, info={})


def _trajectory(reward: float = 1.0, task_id: str = "t1", n_agent_steps: int = 1) -> Trajectory:
    """Build a minimal completed trajectory for testing."""
    traj = Trajectory(
        id=f"{task_id}_ep0",
        metadata={"task_id": task_id},
        start_time=100.0,
        end_time=110.0,
        reward_info={"reward": reward, "done": reward > 0},
        summary_stats={
            "n_agent_steps": n_agent_steps,
            "n_env_steps": n_agent_steps + 1,
            "total_llm_calls": n_agent_steps,
            "prompt_tokens": 100 * n_agent_steps,
            "completion_tokens": 50 * n_agent_steps,
            "cached_tokens": 0,
            "cache_creation_tokens": 0,
            "cost": 0.01 * n_agent_steps,
        },
    )
    traj.steps.append(TrajectoryStep(output=_env_output(reward=0.0, done=False), start_time=100.0, end_time=101.0))
    for _ in range(n_agent_steps):
        traj.steps.append(TrajectoryStep(output=AgentOutput(actions=[]), start_time=101.0, end_time=102.0))
    traj.steps.append(
        TrajectoryStep(output=_env_output(reward=reward, done=reward > 0), start_time=102.0, end_time=103.0)
    )
    return traj


# ---------------------------------------------------------------------------
# _extract_llm_model
# ---------------------------------------------------------------------------


def test_extract_llm_model_top_level_model_name() -> None:
    assert _extract_llm_model({"model_name": "gpt-4o"}) == "gpt-4o"


def test_extract_llm_model_top_level_model() -> None:
    assert _extract_llm_model({"model": "claude-3-5-sonnet"}) == "claude-3-5-sonnet"


def test_extract_llm_model_nested_llm_config() -> None:
    assert _extract_llm_model({"llm_config": {"model_name": "gpt-4o-mini"}}) == "gpt-4o-mini"


def test_extract_llm_model_nested_llm() -> None:
    assert _extract_llm_model({"llm": {"model": "o1"}}) == "o1"


def test_extract_llm_model_returns_none_when_absent() -> None:
    assert _extract_llm_model({"temperature": 0.7}) is None


def test_extract_llm_model_ignores_non_string_values() -> None:
    assert _extract_llm_model({"model_name": 42}) is None


# ---------------------------------------------------------------------------
# _extract_tool_names
# ---------------------------------------------------------------------------


def test_extract_tool_names_litellm_format() -> None:
    tools = [{"type": "function", "function": {"name": "click", "description": "Click"}}]
    assert _extract_tool_names(tools) == ["click"]


def test_extract_tool_names_flat_format() -> None:
    tools = [{"name": "type_text", "description": "Type"}]
    assert _extract_tool_names(tools) == ["type_text"]


def test_extract_tool_names_mixed_formats() -> None:
    tools = [
        {"type": "function", "function": {"name": "click"}},
        {"name": "scroll"},
    ]
    assert _extract_tool_names(tools) == ["click", "scroll"]


def test_extract_tool_names_empty_list() -> None:
    assert _extract_tool_names([]) == []


def test_extract_tool_names_skips_tools_without_name() -> None:
    tools = [{"type": "function", "function": {"description": "No name here"}}]
    assert _extract_tool_names(tools) == []


# ---------------------------------------------------------------------------
# _extract_error_type
# ---------------------------------------------------------------------------


def test_extract_error_type_clean_trajectory() -> None:
    traj = _trajectory(reward=1.0)
    assert _extract_error_type(traj) is None


def test_extract_error_type_from_agent_output() -> None:
    traj = _trajectory(reward=0.0)
    err = StepError(error_type="ValueError", exception_str="bad value", stack_trace="")
    traj.steps.insert(1, TrajectoryStep(output=AgentOutput(error=err), start_time=101.0, end_time=101.5))
    assert _extract_error_type(traj) == "ValueError"


def test_extract_error_type_returns_first_error() -> None:
    traj = _trajectory(reward=0.0)
    err1 = StepError(error_type="TimeoutError", exception_str="timeout", stack_trace="")
    err2 = StepError(error_type="ValueError", exception_str="bad", stack_trace="")
    traj.steps.insert(1, TrajectoryStep(output=AgentOutput(error=err1), start_time=101.0, end_time=101.5))
    traj.steps.insert(2, TrajectoryStep(output=AgentOutput(error=err2), start_time=101.5, end_time=102.0))
    assert _extract_error_type(traj) == "TimeoutError"


# ---------------------------------------------------------------------------
# _to_github_url
# ---------------------------------------------------------------------------


def test_to_github_url_ssh() -> None:
    url = _to_github_url("git@github.com:org/repo.git", "abc123")
    assert url == "https://github.com/org/repo/tree/abc123"


def test_to_github_url_https() -> None:
    url = _to_github_url("https://github.com/org/repo", "abc123")
    assert url == "https://github.com/org/repo/tree/abc123"


def test_to_github_url_non_github_returns_none() -> None:
    url = _to_github_url("https://gitlab.com/org/repo.git", "abc123")
    assert url is None


def test_to_github_url_strips_git_suffix() -> None:
    url = _to_github_url("https://github.com/org/repo.git", "sha1")
    assert url == "https://github.com/org/repo/tree/sha1"


# ---------------------------------------------------------------------------
# EvalLibrary
# ---------------------------------------------------------------------------


def test_eval_library_defaults() -> None:
    lib = EvalLibrary(version="1.2.3")
    assert lib.name == "cube-harness"
    assert lib.version == "1.2.3"


def test_eval_library_roundtrip() -> None:
    lib = EvalLibrary(version="0.5.0")
    restored = EvalLibrary.model_validate_json(lib.model_dump_json())
    assert restored.name == "cube-harness"
    assert restored.version == "0.5.0"


# ---------------------------------------------------------------------------
# UsageSummary
# ---------------------------------------------------------------------------


def test_usage_summary_from_stats() -> None:
    stats = {
        "prompt_tokens": 500,
        "completion_tokens": 200,
        "cached_tokens": 50,
        "cache_creation_tokens": 10,
        "cost": 0.05,
        "total_llm_calls": 3,
    }
    usage = UsageSummary.from_summary_stats(stats)
    assert usage.input_tokens == 500
    assert usage.output_tokens == 200
    assert usage.total_tokens == 700
    assert usage.input_tokens_cache_read == 50
    assert usage.input_tokens_cache_write == 10
    assert usage.total_cost_usd == 0.05
    assert usage.n_llm_calls == 3


def test_usage_summary_from_none() -> None:
    usage = UsageSummary.from_summary_stats(None)
    assert usage.input_tokens == 0
    assert usage.total_cost_usd == 0.0


def test_usage_summary_total_tokens_is_sum() -> None:
    stats = {"prompt_tokens": 100, "completion_tokens": 40}
    usage = UsageSummary.from_summary_stats(stats)
    assert usage.total_tokens == 140


# ---------------------------------------------------------------------------
# AgentInfo
# ---------------------------------------------------------------------------


def test_agent_info_from_agent_config_basic(mock_agent_config) -> None:
    info = AgentInfo.from_agent_config(mock_agent_config)
    assert len(info.agent_id) == 64  # SHA-256 hex
    assert "MockAgentConfig" in info.config_type
    assert isinstance(info.config, dict)
    assert isinstance(info.dependency_versions, dict)
    assert isinstance(info.framework_version, str)


def test_agent_info_captures_cube_standard_git(mock_agent_config) -> None:
    info = AgentInfo.from_agent_config(mock_agent_config)
    # Populated only for an editable/source cube-standard checkout; None for a
    # released wheel. Either way the field exists and the (commit, dirty) pair
    # is internally consistent with the _get_git_info contract.
    assert hasattr(info, "cube_standard_git_commit")
    if info.cube_standard_git_commit is None:
        assert info.cube_standard_git_is_dirty is None
    else:
        assert re.fullmatch(r"[0-9a-f]{40}", info.cube_standard_git_commit)
        assert isinstance(info.cube_standard_git_is_dirty, bool)


def test_agent_info_agent_id_is_stable(mock_agent_config) -> None:
    info1 = AgentInfo.from_agent_config(mock_agent_config)
    info2 = AgentInfo.from_agent_config(mock_agent_config)
    assert info1.agent_id == info2.agent_id


def test_agent_info_has_no_tools_field(mock_agent_config) -> None:
    info = AgentInfo.from_agent_config(mock_agent_config)
    assert not hasattr(info, "tools")
    assert not hasattr(info, "tool_names")


def test_agent_info_llm_model_extracted() -> None:
    from cube_harness.agent import AgentConfig

    class LLMAgentConfig(AgentConfig):
        llm_config: dict = {"model_name": "gpt-4o"}

        def make(self, action_set=None, **kwargs):  # type: ignore[override]
            raise NotImplementedError

    cfg = LLMAgentConfig()
    info = AgentInfo.from_agent_config(cfg)
    assert info.llm_model == "gpt-4o"


# ---------------------------------------------------------------------------
# BenchmarkSubset
# ---------------------------------------------------------------------------


def test_benchmark_subset_from_benchmark(mock_cube_benchmark_config) -> None:
    subset = BenchmarkSubset.from_benchmark_config(mock_cube_benchmark_config)
    assert subset.name == "mock-cube"
    assert subset.n_tasks == 2
    assert subset.filter is None


def test_benchmark_subset_unknown_benchmark() -> None:
    with pytest.raises(AttributeError):
        BenchmarkSubset.from_benchmark_config(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ExperimentRecord
# ---------------------------------------------------------------------------


def test_experiment_record_evaluation_id_is_dir_name(mock_agent_config, mock_cube_benchmark_config, tmp_dir) -> None:
    rec = ExperimentRecord.from_experiment("my_exp", tmp_dir, mock_agent_config, mock_cube_benchmark_config)
    assert rec.evaluation_id == tmp_dir.name


def test_experiment_record_fields(mock_agent_config, mock_cube_benchmark_config, tmp_dir) -> None:
    rec = ExperimentRecord.from_experiment("test_exp", tmp_dir, mock_agent_config, mock_cube_benchmark_config)
    assert rec.experiment_name == "test_exp"
    assert rec.benchmark_name == "mock-cube"
    assert rec.benchmark_version == "0.1.0"
    assert rec.benchmark_subset.n_tasks == 2
    assert rec.investigator_llm_config is None
    assert rec.eval_library.name == "cube-harness"


def test_experiment_record_roundtrip(mock_agent_config, mock_cube_benchmark_config, tmp_dir) -> None:
    rec = ExperimentRecord.from_experiment("roundtrip_exp", tmp_dir, mock_agent_config, mock_cube_benchmark_config)
    serialized = rec.model_dump_json()
    restored = ExperimentRecord.model_validate_json(serialized)
    assert restored.evaluation_id == rec.evaluation_id
    assert restored.benchmark_name == rec.benchmark_name
    assert restored.eval_library.version == rec.eval_library.version


# ---------------------------------------------------------------------------
# EpisodeRecord
# ---------------------------------------------------------------------------


def test_episode_record_success() -> None:
    traj = _trajectory(reward=1.0)
    record = EpisodeRecord.from_trajectory(traj, evaluation_id="abc123")
    assert record.is_correct is True
    assert record.score == 1.0
    assert record.trajectory_id == "t1_ep0"


def test_episode_record_failure() -> None:
    traj = _trajectory(reward=0.0)
    record = EpisodeRecord.from_trajectory(traj, evaluation_id="abc123")
    assert record.is_correct is False
    assert record.score == 0.0


def test_episode_record_wall_time() -> None:
    traj = _trajectory(reward=1.0)
    record = EpisodeRecord.from_trajectory(traj, evaluation_id="abc123")
    assert record.wall_time_s == pytest.approx(10.0)


def test_episode_record_num_turns() -> None:
    traj = _trajectory(reward=1.0, n_agent_steps=3)
    record = EpisodeRecord.from_trajectory(traj, evaluation_id="abc123")
    assert record.num_turns == len(traj.steps)
    assert record.n_agent_steps == 3


def test_episode_record_tool_names_from_metadata() -> None:
    traj = _trajectory(reward=1.0)
    traj.metadata["action_schemas"] = [{"type": "function", "function": {"name": "click"}}]
    record = EpisodeRecord.from_trajectory(traj, evaluation_id="abc123")
    assert record.tool_names == ["click"]


def test_episode_record_tool_names_empty_without_metadata() -> None:
    traj = _trajectory(reward=1.0)
    record = EpisodeRecord.from_trajectory(traj, evaluation_id="abc123")
    assert record.tool_names == []


def test_episode_record_with_task_metadata() -> None:
    from cube.task import TaskMetadata

    traj = _trajectory(task_id="click-dialog")
    tm = TaskMetadata(id="click-dialog", split="test", abstract_description="Click a dialog button")
    record = EpisodeRecord.from_trajectory(traj, evaluation_id="abc123", task_metadata=tm)
    assert record.split == "test"
    assert record.task_description == "Click a dialog button"


def test_episode_record_with_task_config(mock_tool_config) -> None:
    from cube.task import TaskConfig

    class MockTaskConfig(TaskConfig):
        def make(self, runtime_context=None):  # type: ignore[override]
            raise NotImplementedError

    traj = _trajectory(task_id="t1")
    from cube.task import TaskMetadata

    tc = MockTaskConfig(metadata=TaskMetadata(id="t1"), seed=42, tool_config=mock_tool_config)
    record = EpisodeRecord.from_trajectory(traj, evaluation_id="abc123", task_config=tc)
    assert record.seed == 42
    assert record.sample_hash is not None
    assert len(record.sample_hash) == 64


def test_episode_record_findings_optional() -> None:
    traj = _trajectory(reward=1.0)
    record = EpisodeRecord.from_trajectory(traj, evaluation_id="abc123")
    assert record.findings is None
    assert record.verifier is None


def test_episode_record_with_findings() -> None:
    traj = _trajectory(reward=0.0)
    record = EpisodeRecord.from_trajectory(traj, evaluation_id="abc123")
    record = record.model_copy(
        update={
            "findings": Findings(
                analysis="Agent located the submit button at step 6 but never clicked it.",
                outcome="failure",
                summary="Agent identified the target but failed to submit.",
                primary_blame="agent_scaffolding",
                primary_blame_confidence=4,
                other_blames=[],
                evidence=[{"step": 6, "quote": "submit button visible at coords (412, 80)"}],
                hypothesis="Adding an explicit 'submit when ready' clause to the system prompt would close this gap.",
                hypothesis_confidence=3,
            )
        }
    )
    assert record.findings.outcome.value == "failure"
    assert record.findings.primary_blame.value == "agent_scaffolding"
    assert record.findings.primary_blame_confidence == 4


# ---------------------------------------------------------------------------
# EvalLog: two-level round-trip
# ---------------------------------------------------------------------------


def test_eval_log_save_and_load(mock_agent_config, mock_cube_benchmark_config, tmp_dir) -> None:
    traj = _trajectory(reward=1.0, task_id="task-a")
    exp_rec = ExperimentRecord.from_experiment("test_exp", tmp_dir, mock_agent_config, mock_cube_benchmark_config)
    ep_rec = EpisodeRecord.from_trajectory(traj, evaluation_id=exp_rec.evaluation_id)
    log = EvalLog(experiment=exp_rec, episodes=[ep_rec])

    with tempfile.TemporaryDirectory() as out:
        out_dir = Path(out)
        log.save(out_dir)
        assert (out_dir / "experiment_record.json").exists()
        assert (out_dir / "episodes" / "task-a_ep0" / "episode_record.json").exists()
        loaded = EvalLog.load(out_dir)

    assert loaded.experiment.evaluation_id == exp_rec.evaluation_id
    assert len(loaded.episodes) == 1
    assert loaded.episodes[0].trajectory_id == "task-a_ep0"


def test_eval_log_episode_record_is_valid_json(mock_agent_config, mock_cube_benchmark_config, tmp_dir) -> None:
    traj = _trajectory(reward=0.5)
    exp_rec = ExperimentRecord.from_experiment("test_exp", tmp_dir, mock_agent_config, mock_cube_benchmark_config)
    ep_rec = EpisodeRecord.from_trajectory(traj, evaluation_id=exp_rec.evaluation_id)
    log = EvalLog(experiment=exp_rec, episodes=[ep_rec])

    with tempfile.TemporaryDirectory() as out:
        out_dir = Path(out)
        log.save(out_dir)
        record_path = out_dir / "episodes" / "t1_ep0" / "episode_record.json"
        parsed = json.loads(record_path.read_text())

    assert "evaluation_id" in parsed
    assert "sample_id" in parsed
    assert "score" in parsed


def test_eval_log_experiment_record_is_valid_json(mock_agent_config, mock_cube_benchmark_config, tmp_dir) -> None:
    exp_rec = ExperimentRecord.from_experiment("test_exp", tmp_dir, mock_agent_config, mock_cube_benchmark_config)
    log = EvalLog(experiment=exp_rec, episodes=[])

    with tempfile.TemporaryDirectory() as out:
        log.save(Path(out))
        parsed = json.loads((Path(out) / "experiment_record.json").read_text())

    assert "evaluation_id" in parsed
    assert "agent" in parsed
    assert "benchmark_subset" in parsed
    assert "eval_library" in parsed


def test_eval_log_to_jsonl(mock_agent_config, mock_cube_benchmark_config, tmp_dir) -> None:
    traj1 = _trajectory(reward=1.0, task_id="t1")
    traj2 = _trajectory(reward=0.0, task_id="t2")
    exp_rec = ExperimentRecord.from_experiment("test_exp", tmp_dir, mock_agent_config, mock_cube_benchmark_config)
    rec1 = EpisodeRecord.from_trajectory(traj1, evaluation_id=exp_rec.evaluation_id)
    rec2 = EpisodeRecord.from_trajectory(traj2, evaluation_id=exp_rec.evaluation_id)
    log = EvalLog(experiment=exp_rec, episodes=[rec1, rec2])

    with tempfile.TemporaryDirectory() as out:
        jsonl_path = Path(out) / "submission.jsonl"
        log.to_jsonl(jsonl_path)
        lines = jsonl_path.read_text().strip().splitlines()

    assert len(lines) == 2
    sample_ids = {json.loads(line)["sample_id"] for line in lines}
    assert sample_ids == {"t1", "t2"}


def test_eval_log_evaluation_id_fk_consistent(mock_agent_config, mock_cube_benchmark_config, tmp_dir) -> None:
    """EpisodeRecords carry the same evaluation_id as ExperimentRecord."""
    exp_rec = ExperimentRecord.from_experiment("fk_test", tmp_dir, mock_agent_config, mock_cube_benchmark_config)
    traj = _trajectory(reward=1.0)
    ep_rec = EpisodeRecord.from_trajectory(traj, evaluation_id=exp_rec.evaluation_id)
    assert ep_rec.evaluation_id == exp_rec.evaluation_id


# ---------------------------------------------------------------------------
# Optional models
# ---------------------------------------------------------------------------


def test_investigator_llm_config_roundtrip() -> None:
    cfg = InvestigatorLLMConfig(model="claude-opus-4-7", prompt_version="v1.2", investigated_at="2026-04-28T12:00:00Z")
    restored = InvestigatorLLMConfig.model_validate_json(cfg.model_dump_json())
    assert restored.model == "claude-opus-4-7"
    assert restored.investigated_at == "2026-04-28T12:00:00Z"


def test_verifier_roundtrip() -> None:
    v = Verifier(ref="https://github.com/org/repo/blob/abc123/eval.py", source="def evaluate(): return 1.0")
    restored = Verifier.model_validate_json(v.model_dump_json())
    assert "abc123" in restored.ref


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


def test_export_eval_log_integration(tmp_dir, mock_agent_config, mock_cube_benchmark_config) -> None:
    """experiment_record.json is written at start; episode_record.json per episode; export_eval_log loads them."""
    from cube_harness.exp_runner import run_sequentially
    from cube_harness.experiment import Experiment

    exp = Experiment(
        name="integration_test",
        output_dir=tmp_dir,
        agent_config=mock_agent_config,
        benchmark_config=mock_cube_benchmark_config,
    )
    run_sequentially(exp)

    # experiment_record.json written at experiment start (save_config), not post-hoc
    exp_record_path = tmp_dir / "experiment_record.json"
    assert exp_record_path.exists(), "experiment_record.json was not created"
    exp_data = json.loads(exp_record_path.read_text())
    assert exp_data["experiment_name"] == "integration_test"
    assert "agent" in exp_data
    assert exp_data["benchmark_subset"]["n_tasks"] == 2
    assert exp_data["eval_library"]["name"] == "cube-harness"

    # episode_record.json written per trajectory directory during the run
    episode_records = list((tmp_dir / "episodes").glob("*/episode_record.json"))
    assert len(episode_records) == 2, f"Expected 2 episode records, got {len(episode_records)}"

    evaluation_id = exp_data["evaluation_id"]
    for record_path in episode_records:
        episode = json.loads(record_path.read_text())
        assert episode["evaluation_id"] == evaluation_id
        assert episode["score"] == pytest.approx(1.0)
        assert episode["is_correct"] is True

    # export_eval_log is now a thin reader — no trajectory loading
    eval_log = exp.export_eval_log(tmp_dir)
    assert eval_log.experiment.evaluation_id == evaluation_id
    assert len(eval_log.episodes) == 2

    # to_jsonl assembles flat submission file
    jsonl_path = tmp_dir / "submission.jsonl"
    eval_log.to_jsonl(jsonl_path)
    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) == 2

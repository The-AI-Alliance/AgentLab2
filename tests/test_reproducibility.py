"""Tests for cube_harness.reproducibility — journal + EEE converters."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from cube_harness.episode_status import STATUS_FILENAME, EpisodeStatus, Status
from cube_harness.eval_log import (
    AgentInfo,
    BenchmarkSubset,
    EpisodeRecord,
    EvalLibrary,
    EvalLog,
    ExperimentRecord,
    UsageSummary,
)
from cube_harness.reproducibility import (
    JOURNAL_SCHEMA_VERSION,
    Outcomes,
    build_journal_record,
    sanitize_filename,
)
from cube_harness.reproducibility.eee import EEE_SCHEMA_VERSION, build_eee_record
from cube_harness.reproducibility.journal import classify
from cube_harness.storage import EPISODES_DIR


def _agent_info() -> AgentInfo:
    return AgentInfo(
        agent_id="a" * 64,
        config_type="GennyConfig",
        config={"_type": "GennyConfig", "model": "azure/gpt-5.4-mini"},
        llm_model="azure/gpt-5.4-mini",
        framework_version="0.5.2",
        dependency_versions={"cube-harness": "0.5.2", "cube": "0.3.1"},
        git_commit="f" * 40,
        git_remote_url="https://github.com/The-AI-Alliance/cube-harness/tree/" + "f" * 40,
        git_is_dirty=False,
        cube_standard_git_commit="8" * 40,
        cube_standard_git_is_dirty=False,
    )


def _exp_record(n_tasks: int = 5) -> ExperimentRecord:
    return ExperimentRecord(
        evaluation_id="20260404_195953_genny_miniwob",
        experiment_name="genny_miniwob",
        evaluation_timestamp=1_748_560_000.0,
        eval_library=EvalLibrary(version="0.5.2"),
        agent=_agent_info(),
        benchmark_name="miniwob",
        benchmark_version="1.0.0",
        # filter is the registered named_subsets key (post named-subset PR), not a glob.
        benchmark_subset=BenchmarkSubset(name="miniwob[all]", n_tasks=n_tasks, filter="all"),
    )


def _ep_record(task_id: str, score: float, error: str | None = None) -> EpisodeRecord:
    return EpisodeRecord(
        evaluation_id="20260404_195953_genny_miniwob",
        sample_id=task_id,
        is_correct=score > 0,
        score=score,
        error=error,
        num_turns=3,
        n_agent_steps=1,
        n_env_steps=2,
        wall_time_s=10.0,
        usage=UsageSummary(total_cost_usd=0.05),
        trajectory_id=f"{task_id}_ep0",
        timestamp=1_748_560_000.0,
    )


def _write_status(exp_dir: Path, task_id: str, status: Status, reward: float | None) -> None:
    """Write the per-episode status.json that ExperimentResult.iter_episode_statuses() reads."""
    trajectory_id = f"{task_id}_ep0"
    ep_dir = exp_dir / EPISODES_DIR / trajectory_id
    ep_dir.mkdir(parents=True, exist_ok=True)
    EpisodeStatus(
        status=status,
        task_id=task_id,
        episode_id=0,
        started_at=1.0,
        ended_at=2.0,
        reward=reward,
    ).write(ep_dir / STATUS_FILENAME)


@pytest.fixture
def populated_exp_dir(tmp_path: Path) -> Path:
    """An experiment dir with one episode per outcome bucket (success/failure/max_steps/system_error/missing-CANCELLED)."""
    exp_dir = tmp_path / "20260404_195953_genny_miniwob"
    exp_dir.mkdir()

    exp = _exp_record(n_tasks=5)
    episodes = [
        _ep_record("t_success", score=1.0),
        _ep_record("t_failure", score=0.0),
        _ep_record("t_max_steps", score=0.0),
        _ep_record("t_system_error", score=0.0, error="ValueError"),
        # t_missing has no EpisodeRecord — exercises the "no record" missing path.
    ]
    EvalLog(experiment=exp, episodes=episodes).save(exp_dir)

    _write_status(exp_dir, "t_success", "COMPLETED", reward=1.0)
    _write_status(exp_dir, "t_failure", "COMPLETED", reward=0.0)
    _write_status(exp_dir, "t_max_steps", "MAX_STEPS_REACHED", reward=0.0)
    _write_status(exp_dir, "t_system_error", "FAILED", reward=None)
    _write_status(exp_dir, "t_missing", "CANCELLED", reward=None)
    return exp_dir


class TestClassify:
    @pytest.mark.parametrize(
        "status,reward,expected",
        [
            ("COMPLETED", 1.0, "n_success"),
            ("COMPLETED", 0.5, "n_success"),
            ("COMPLETED", 0.0, "n_failure"),
            ("COMPLETED", None, "n_failure"),
            ("MAX_STEPS_REACHED", 0.0, "n_max_steps"),
            ("FAILED", None, "n_system_error"),
            ("STALE", None, "n_system_error"),
            ("INVALID_CONFIG", None, "n_system_error"),
            ("CANCELLED", None, "n_missing"),
            ("QUEUED", None, "n_missing"),
            ("RUNNING", None, "n_missing"),
        ],
    )
    def test_classify(self, status: Status, reward: float | None, expected: str) -> None:
        es = EpisodeStatus(status=status, task_id="t", episode_id=0, started_at=1.0, reward=reward)
        assert classify(es) == expected


class TestSanitizeFilename:
    def test_slashes_become_double_underscore(self) -> None:
        assert sanitize_filename("alacoste/run1") == "alacoste__run1"

    def test_no_op_when_no_slash(self) -> None:
        assert sanitize_filename("plain-id") == "plain-id"


class TestOutcomes:
    def test_total_sums_all_buckets(self) -> None:
        o = Outcomes(n_success=3, n_failure=2, n_max_steps=1, n_system_error=4, n_missing=0)
        assert o.total() == 10

    def test_to_dict_round_trip(self) -> None:
        o = Outcomes(n_success=1, n_failure=1, n_max_steps=1, n_system_error=1, n_missing=1)
        d = o.to_dict()
        assert set(d) == {"n_success", "n_failure", "n_max_steps", "n_system_error", "n_missing"}
        assert sum(d.values()) == 5


class TestBuildJournalRecord:
    def test_outcomes_breakdown(self, populated_exp_dir: Path) -> None:
        record = build_journal_record(populated_exp_dir, submitter="alacoste")
        outcomes = record["results"]["outcomes"]
        assert outcomes == {
            "n_success": 1,
            "n_failure": 1,
            "n_max_steps": 1,
            "n_system_error": 1,
            "n_missing": 1,  # 1 CANCELLED — see status.json setup
        }

    def test_outcomes_sum_equals_n_tasks(self, populated_exp_dir: Path) -> None:
        record = build_journal_record(populated_exp_dir, submitter="alacoste")
        outcomes = record["results"]["outcomes"]
        assert sum(outcomes.values()) == record["benchmark_subset"]["n_tasks"]

    def test_avg_score_from_episode_records(self, populated_exp_dir: Path) -> None:
        # 4 episodes recorded (one of them score=1.0, three score=0.0): mean = 0.25.
        record = build_journal_record(populated_exp_dir, submitter="alacoste")
        assert record["results"]["avg_score"] == pytest.approx(0.25)

    def test_total_cost_summed(self, populated_exp_dir: Path) -> None:
        # 4 episodes × 0.05 USD = 0.20.
        record = build_journal_record(populated_exp_dir, submitter="alacoste")
        assert record["results"]["total_cost_usd"] == pytest.approx(0.20)

    def test_evaluation_id_namespaced_by_submitter(self, populated_exp_dir: Path) -> None:
        record = build_journal_record(populated_exp_dir, submitter="alacoste")
        assert record["evaluation_id"] == "alacoste/20260404_195953_genny_miniwob"

    def test_submitter_with_spaces_rejected_locally(self, populated_exp_dir: Path) -> None:
        # A git user.name like "Ada Lovelace" fails the registry's evaluation_id
        # pattern — must be caught before a PR is opened, not by registry CI.
        with pytest.raises(ValueError, match="--submitter"):
            build_journal_record(populated_exp_dir, submitter="Ada Lovelace")

    def test_schema_version_pinned(self, populated_exp_dir: Path) -> None:
        record = build_journal_record(populated_exp_dir, submitter="alacoste")
        assert record["schema_version"] == JOURNAL_SCHEMA_VERSION

    def test_cube_id_defaults_to_stripped_benchmark_name(self, populated_exp_dir: Path) -> None:
        # ExperimentRecord.benchmark_subset.name is "miniwob[level=all]" — must
        # become "miniwob" as benchmark_name (cube-registry's <cube-id>).
        record = build_journal_record(populated_exp_dir, submitter="alacoste")
        assert record["benchmark_name"] == "miniwob"

    def test_cube_id_override_respected(self, populated_exp_dir: Path) -> None:
        record = build_journal_record(populated_exp_dir, submitter="alacoste", cube_id="forced")
        assert record["benchmark_name"] == "forced"

    def test_agent_provenance_included(self, populated_exp_dir: Path) -> None:
        record = build_journal_record(populated_exp_dir, submitter="alacoste")
        agent = record["agent"]
        assert agent["git_commit"] == "f" * 40
        assert agent["cube_standard_git_commit"] == "8" * 40
        assert agent["git_is_dirty"] is False
        assert agent["agent_id"] == "a" * 64


class TestBuildEEERecord:
    def test_schema_version_and_required_top_level(self, populated_exp_dir: Path) -> None:
        record = build_eee_record(populated_exp_dir)
        assert record["schema_version"] == EEE_SCHEMA_VERSION
        for key in (
            "evaluation_id",
            "retrieved_timestamp",
            "source_metadata",
            "model_info",
            "eval_library",
            "evaluation_results",
        ):
            assert key in record, f"missing required EEE field: {key}"

    def test_provenance_stashed_in_source_metadata_additional_details(self, populated_exp_dir: Path) -> None:
        record = build_eee_record(populated_exp_dir)
        extras = record["source_metadata"]["additional_details"]
        # All values must be strings (EEE schema rule).
        assert all(isinstance(v, str) for v in extras.values())
        assert extras["cube_harness_git_commit"] == "f" * 40
        assert extras["cube_standard_git_commit"] == "8" * 40

    def test_developer_and_platform_split(self, populated_exp_dir: Path) -> None:
        # azure/gpt-* is an OpenAI model served by Azure — EEE's schema keeps
        # those in separate fields (developer vs inference_platform).
        record = build_eee_record(populated_exp_dir)
        assert record["model_info"]["developer"] == "openai"
        assert record["model_info"]["inference_platform"] == "azure"

    def test_evaluation_id_follows_eee_convention(self, populated_exp_dir: Path) -> None:
        # eval_name/model_id/timestamp, model slashes flattened to "_" — the
        # shape used by records in the live EEE datastore. The harness run
        # name moves to source_metadata.additional_details.
        record = build_eee_record(populated_exp_dir)
        assert record["evaluation_id"] == "miniwob/azure_gpt-5.4-mini/1748560000.0"
        extras = record["source_metadata"]["additional_details"]
        assert extras["cube_harness_run_id"] == "20260404_195953_genny_miniwob"


class TestProviderPrefixMapping:
    """Coverage for the (developer, inference_platform) slug extraction (W4)."""

    @pytest.mark.parametrize(
        "llm_model,expected",
        [
            ("openai/gpt-4o", ("openai", "openai")),
            ("azure/gpt-5.4-mini", ("openai", "azure")),  # host implies developer
            ("anthropic/claude-opus-4-7", ("anthropic", "anthropic")),
            ("vertex_ai/gemini-2.0", ("google", "vertex_ai")),  # alias
            ("gemini/gemini-2.5-pro", ("google", "gemini")),  # alias
            ("openrouter/anthropic/claude-3-5-sonnet", ("anthropic", "openrouter")),  # routed
            ("together_ai/meta-llama/Llama-3-70b", ("meta-llama", "together_ai")),  # routed
            ("bedrock/anthropic.claude-3-5", ("bedrock", "bedrock")),  # best-effort passthrough
            ("nonexistent_provider/model", ("nonexistent_provider", "nonexistent_provider")),
            ("", ("", "")),
            ("no-slash-model", ("", "")),
        ],
    )
    def test_developer_and_platform_mapping(self, llm_model: str, expected: tuple[str, str]) -> None:
        from cube_harness.reproducibility.eee import _developer_and_platform

        assert _developer_and_platform(llm_model) == expected


class TestEEEEvaluationIdNoneSafety:
    """Regression for W3: f"{None}" formatting in evaluation_id."""

    def test_none_llm_model_does_not_appear_as_literal_None(self, populated_exp_dir: Path) -> None:
        # Reload + null out llm_model on disk to simulate the bad path.
        import json as _json

        rec_path = populated_exp_dir / "experiment_record.json"
        data = _json.loads(rec_path.read_text())
        data["agent"]["llm_model"] = None
        rec_path.write_text(_json.dumps(data))

        record = build_eee_record(populated_exp_dir)
        eid = record["evaluation_id"]
        assert "/None/" not in eid, f"literal 'None' leaked into EEE id: {eid!r}"
        assert "/unknown/" in eid, f"expected 'unknown' fallback, got: {eid!r}"

    def test_evaluation_results_has_one_entry_with_score_and_uncertainty(self, populated_exp_dir: Path) -> None:
        record = build_eee_record(populated_exp_dir)
        results = record["evaluation_results"]
        assert len(results) == 1
        result = results[0]
        assert result["score_details"]["score"] == pytest.approx(0.25)
        assert "uncertainty" in result["score_details"]
        assert result["score_details"]["uncertainty"]["num_samples"] == 4


class TestArchivedAttemptsExcluded:
    """A retry archives the prior attempt's episode dir (``storage.archive_episode``)
    with its episode_record.json still inside. EvalLog.load must skip those dirs,
    or every consumer (journal/EEE avg_score, samples bundle) double-counts the task."""

    def _archive_first_attempt(self, exp_dir: Path, task_id: str, new_score: float) -> None:
        """Simulate _pre_claim: archive the live attempt, then write a fresh one."""
        ep_dir = exp_dir / EPISODES_DIR / f"{task_id}_ep0"
        shutil.copytree(ep_dir, ep_dir.parent / f"{ep_dir.name}.archived_1749000000.0")
        retry = _ep_record(task_id, new_score)
        (ep_dir / "episode_record.json").write_text(retry.model_dump_json(indent=2))

    def test_load_skips_archived_episode_dirs(self, populated_exp_dir: Path) -> None:
        self._archive_first_attempt(populated_exp_dir, "t_success", new_score=0.0)
        log = EvalLog.load(populated_exp_dir)
        assert len(log.episodes) == 4  # one per live attempt, archived copy ignored
        assert sorted(ep.sample_id for ep in log.episodes) == [
            "t_failure",
            "t_max_steps",
            "t_success",
            "t_system_error",
        ]

    def test_journal_score_reflects_only_live_attempts(self, populated_exp_dir: Path) -> None:
        # First attempt of t_success scored 1.0; the retry scored 0.0. Only the
        # retry may count: avg over {0.0, 0.0, 0.0, 0.0} — not 1/5 (stale mixed in).
        self._archive_first_attempt(populated_exp_dir, "t_success", new_score=0.0)
        record = build_journal_record(populated_exp_dir, submitter="alacoste")
        assert record["results"]["avg_score"] == 0.0

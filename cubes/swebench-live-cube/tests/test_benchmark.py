"""Docker-free unit tests for swebench-live-cube — covers the BenchmarkConfig
contract (registry wiring, subsetting, metadata stamping, debug factory,
serialization round-trip).
"""

from __future__ import annotations

from cube.benchmark import BenchmarkConfig
from cube.task import TaskExecutionInfo

from swebench_live_cube.benchmark import SWEBenchLiveBenchmarkConfig
from swebench_live_cube.debug import _TASK_ACTIONS, get_debug_benchmark
from swebench_live_cube.gold_subset import lite_gold_ids
from swebench_live_cube.task import (
    SWEBenchLiveExecutionInfo,
    SWEBenchLiveTaskConfig,
    SWEBenchLiveTaskMetadata,
)


_DEBUG_TASK_IDS = list(_TASK_ACTIONS)


def test_config_roundtrip():
    """``model_dump_json`` → ``model_validate_json`` produces an equivalent config."""
    cfg = SWEBenchLiveBenchmarkConfig(include_hints=True, oracle_mode=True).subset_from_list(_DEBUG_TASK_IDS)
    js = cfg.model_dump_json()
    restored = SWEBenchLiveBenchmarkConfig.model_validate_json(js)
    assert restored.include_hints is True
    assert restored.oracle_mode is True
    assert restored.task_ids == _DEBUG_TASK_IDS
    assert restored.num_tasks == len(_DEBUG_TASK_IDS)
    assert restored.benchmark_metadata.name == "swebench-live-cube"


def test_task_metadata_loaded():
    """``task_metadata`` ClassVar is auto-loaded from task_metadata.json with 1895 entries."""
    cfg = SWEBenchLiveBenchmarkConfig()
    assert cfg.benchmark_metadata.num_tasks == 1895
    assert len(cfg.task_metadata) == 1895
    sample = next(iter(cfg.task_metadata.values()))
    assert isinstance(sample, SWEBenchLiveTaskMetadata)
    # SWE-bench Live specific fields are present
    assert sample.repo
    assert sample.base_commit
    assert isinstance(sample.splits, list) and sample.splits
    assert sample.log_parser


def test_get_task_configs_stamps_metadata():
    """Every emitted ``TaskConfig`` carries the full ``TaskMetadata`` (no task_id-only stub)."""
    cfg = SWEBenchLiveBenchmarkConfig().subset_from_list(_DEBUG_TASK_IDS)
    configs = list(cfg.get_task_configs())
    assert len(configs) == len(_DEBUG_TASK_IDS)
    for tc in configs:
        assert isinstance(tc, SWEBenchLiveTaskConfig)
        assert isinstance(tc.metadata, SWEBenchLiveTaskMetadata)
        assert tc.metadata.id == tc.task_id
        # Stamped metadata carries subclass-specific fields, not just base TaskMetadata.
        assert tc.metadata.repo


def test_subset_from_list():
    """``subset_from_list`` scopes the config to exactly the requested task IDs."""
    cfg = SWEBenchLiveBenchmarkConfig().subset_from_list(_DEBUG_TASK_IDS)
    assert cfg.task_ids == _DEBUG_TASK_IDS
    assert set(cfg.tasks().keys()) == set(_DEBUG_TASK_IDS)
    assert cfg.num_tasks == len(_DEBUG_TASK_IDS)


def test_named_subset_verified():
    """``named_subset('verified')`` returns the 499 SWE-bench Live Verified tasks."""
    cfg = SWEBenchLiveBenchmarkConfig().named_subset("verified")
    assert cfg.num_tasks > 0
    # Every retained task lists 'verified' in its splits field
    for tm in cfg.tasks().values():
        assert "verified" in tm.splits


def test_named_subset_lite_gold():
    """``named_subset('lite-gold')`` returns the 275 root-gold-solvable lite tasks, each a lite task."""
    cfg = SWEBenchLiveBenchmarkConfig().named_subset("lite-gold")
    assert cfg.num_tasks == 275
    assert set(cfg.tasks().keys()) == lite_gold_ids()
    # lite-gold ⊂ lite: every gold task also carries the 'lite' marker.
    for tm in cfg.tasks().values():
        assert "lite-gold" in tm.splits
        assert "lite" in tm.splits


def test_debug_benchmark_type():
    """``get_debug_benchmark()`` returns a ``BenchmarkConfig`` (not a live ``Benchmark``).

    The harness owns ``config.install()`` and ``config.make(infra)``; the debug factory
    must not call either.
    """
    cfg = get_debug_benchmark()
    assert isinstance(cfg, SWEBenchLiveBenchmarkConfig)
    assert isinstance(cfg, BenchmarkConfig)
    assert cfg.oracle_mode is True
    # Scoped to the debug task subset
    assert cfg.task_ids == _DEBUG_TASK_IDS


def test_execution_info_roundtrip():
    """Typed ``SWEBenchLiveExecutionInfo`` round-trips through the TaskExecutionInfo discriminator."""
    ei = SWEBenchLiveExecutionInfo(
        problem_statement="test issue",
        hints_text="useful hint",
        patch="diff --git a/x b/x",
        test_patch="diff --git a/y b/y",
        fail_to_pass=["test_a"],
        pass_to_pass=["test_b"],
        test_cmds=["pytest -x test_a.py"],
    )
    assert isinstance(ei, TaskExecutionInfo)
    restored = SWEBenchLiveExecutionInfo.model_validate_json(ei.model_dump_json())
    assert restored.problem_statement == "test issue"
    assert restored.fail_to_pass == ["test_a"]
    assert restored.test_cmds == ["pytest -x test_a.py"]
    assert restored.eval_timeout == 1800  # default preserved

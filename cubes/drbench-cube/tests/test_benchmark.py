"""Docker-free unit tests for drbench-cube."""

from __future__ import annotations

from cube.benchmark import BenchmarkConfig
from cube.task import TaskMetadata

from drbench_cube.benchmark import DrBenchBenchmarkConfig, DrBenchTaskMetadata
from drbench_cube.debug import _TASK_ACTIONS, get_debug_benchmark
from drbench_cube.task import DrBenchTaskConfig

_DEBUG_TASK_IDS = list(_TASK_ACTIONS)


def test_task_metadata_loaded() -> None:
    cfg = DrBenchBenchmarkConfig()
    assert cfg.benchmark_metadata.num_tasks == 100
    assert len(cfg.task_metadata) == 100
    sample = next(iter(cfg.task_metadata.values()))
    assert isinstance(sample, DrBenchTaskMetadata)
    assert isinstance(sample, TaskMetadata)
    assert sample.domain
    assert sample.difficulty
    assert sample.company_name
    assert sample.insight_count > 0


def test_task_metadata_abstract_description_full() -> None:
    cfg = DrBenchBenchmarkConfig()
    for task_id, meta in cfg.task_metadata.items():
        assert meta.abstract_description, f"{task_id} has empty abstract_description"
        assert len(meta.abstract_description) > 20, f"{task_id} abstract_description looks truncated"


def test_task_metadata_no_extra_info() -> None:
    cfg = DrBenchBenchmarkConfig()
    for task_id, meta in cfg.task_metadata.items():
        assert not hasattr(meta, "extra_info") or not isinstance(getattr(meta, "extra_info", None), dict), (
            f"{task_id} still has untyped extra_info dict"
        )


def test_config_roundtrip() -> None:
    cfg = DrBenchBenchmarkConfig().subset_from_list(_DEBUG_TASK_IDS)
    js = cfg.model_dump_json()
    restored = DrBenchBenchmarkConfig.model_validate_json(js)
    assert restored.task_ids == _DEBUG_TASK_IDS
    assert restored.num_tasks == len(_DEBUG_TASK_IDS)
    assert restored.benchmark_metadata.name == "drbench"


def test_get_task_configs_stamps_metadata() -> None:
    cfg = DrBenchBenchmarkConfig().subset_from_list(_DEBUG_TASK_IDS)
    configs = list(cfg.get_task_configs())
    assert len(configs) == len(_DEBUG_TASK_IDS)
    for tc in configs:
        assert isinstance(tc, DrBenchTaskConfig)
        assert isinstance(tc.metadata, DrBenchTaskMetadata)
        assert tc.metadata.id == tc.task_id
        assert tc.metadata.domain


def test_subset_from_list() -> None:
    cfg = DrBenchBenchmarkConfig().subset_from_list(_DEBUG_TASK_IDS)
    assert cfg.task_ids == _DEBUG_TASK_IDS
    assert set(cfg.tasks().keys()) == set(_DEBUG_TASK_IDS)
    assert cfg.num_tasks == len(_DEBUG_TASK_IDS)


def test_debug_benchmark_type() -> None:
    cfg = get_debug_benchmark()
    assert isinstance(cfg, DrBenchBenchmarkConfig)
    assert isinstance(cfg, BenchmarkConfig)
    assert cfg.task_ids == _DEBUG_TASK_IDS


def test_action_set_non_empty() -> None:
    cfg = DrBenchBenchmarkConfig()
    assert cfg.task_metadata["DR0001"].container_config is not None
    from drbench_cube.tool import DrBenchTool
    import inspect

    action_methods = [
        name
        for name, val in inspect.getmembers(DrBenchTool, predicate=inspect.isfunction)
        if getattr(val, "_is_action", False)
    ]
    assert "search_nextcloud_files" in action_methods
    assert "submit_report" in action_methods
    assert "web_search" in action_methods

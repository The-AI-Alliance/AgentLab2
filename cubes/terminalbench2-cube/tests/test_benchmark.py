"""Docker-free unit tests for terminalbench2-cube BenchmarkConfig migration."""

from __future__ import annotations

from cube.benchmark import BenchmarkConfig
from cube.task import TaskExecutionInfo

from terminalbench2_cube.benchmark import TerminalBench2BenchmarkConfig
from terminalbench2_cube.debug import _TASK_ACTIONS, get_debug_benchmark
from terminalbench2_cube.task import (
    TerminalBench2ExecutionInfo,
    TerminalBench2TaskConfig,
    TerminalBench2TaskMetadata,
)

_DEBUG_TASK_IDS = list(_TASK_ACTIONS)


def test_config_roundtrip() -> None:
    cfg = TerminalBench2BenchmarkConfig(oracle_mode=True).subset_from_list(_DEBUG_TASK_IDS)
    js = cfg.model_dump_json()
    restored = TerminalBench2BenchmarkConfig.model_validate_json(js)
    assert restored.oracle_mode is True
    assert restored.task_ids == _DEBUG_TASK_IDS
    assert restored.num_tasks == len(_DEBUG_TASK_IDS)
    assert restored.benchmark_metadata.name == "terminalbench2-cube"


def test_task_metadata_loaded() -> None:
    cfg = TerminalBench2BenchmarkConfig()
    assert cfg.benchmark_metadata.num_tasks == 89
    assert len(cfg.task_metadata) == 89
    sample = next(iter(cfg.task_metadata.values()))
    assert isinstance(sample, TerminalBench2TaskMetadata)
    assert sample.difficulty
    assert isinstance(sample.tags, list)


def test_get_task_configs_stamps_metadata() -> None:
    cfg = TerminalBench2BenchmarkConfig().subset_from_list(_DEBUG_TASK_IDS)
    configs = list(cfg.get_task_configs())
    assert len(configs) == len(_DEBUG_TASK_IDS)
    for tc in configs:
        assert isinstance(tc, TerminalBench2TaskConfig)
        assert isinstance(tc.metadata, TerminalBench2TaskMetadata)
        assert tc.metadata.id == tc.task_id
        assert tc.metadata.category


def test_subset_from_list() -> None:
    cfg = TerminalBench2BenchmarkConfig().subset_from_list(_DEBUG_TASK_IDS)
    assert cfg.task_ids == _DEBUG_TASK_IDS
    assert set(cfg.tasks().keys()) == set(_DEBUG_TASK_IDS)
    assert cfg.num_tasks == len(_DEBUG_TASK_IDS)


def test_debug_benchmark_type() -> None:
    cfg = get_debug_benchmark()
    assert isinstance(cfg, TerminalBench2BenchmarkConfig)
    assert isinstance(cfg, BenchmarkConfig)
    assert cfg.oracle_mode is True
    assert cfg.task_ids == _DEBUG_TASK_IDS


def test_execution_info_roundtrip() -> None:
    ei = TerminalBench2ExecutionInfo(
        instruction="Follow the steps",
        archive="ZmFrZV90YXI=",
    )
    assert isinstance(ei, TaskExecutionInfo)
    restored = TerminalBench2ExecutionInfo.model_validate_json(ei.model_dump_json())
    assert restored.instruction == "Follow the steps"
    assert restored.archive == "ZmFrZV90YXI="
    assert restored.max_test_timeout_sec == 900

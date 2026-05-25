"""Docker-free unit tests for swebench-verified-cube — covers the BenchmarkConfig
contract (registry wiring, subsetting, metadata stamping, debug factory,
serialization round-trip).
"""

from __future__ import annotations

import pytest

from cube.benchmark import BenchmarkConfig
from cube.resource import IncompatibleInfraError, InfraConfig, ResourceConfig, ResourceHandle
from cube.task import TaskExecutionInfo

from swebench_verified_cube.benchmark import TASKS_REQUIRING_ROOT, SWEBenchVerifiedBenchmarkConfig
from swebench_verified_cube.debug import _TASK_ACTIONS, get_debug_benchmark
from swebench_verified_cube.task import (
    SWEBenchVerifiedExecutionInfo,
    SWEBenchVerifiedTaskConfig,
    SWEBenchVerifiedTaskMetadata,
)


_DEBUG_TASK_IDS = list(_TASK_ACTIONS)


def test_config_roundtrip():
    """``model_dump_json`` → ``model_validate_json`` produces an equivalent config."""
    cfg = SWEBenchVerifiedBenchmarkConfig(oracle_mode=True).subset_from_list(_DEBUG_TASK_IDS)
    js = cfg.model_dump_json()
    restored = SWEBenchVerifiedBenchmarkConfig.model_validate_json(js)
    assert restored.oracle_mode is True
    assert restored.task_ids == _DEBUG_TASK_IDS
    assert restored.num_tasks == len(_DEBUG_TASK_IDS)
    assert restored.benchmark_metadata.name == "swebench-verified-cube"


def test_task_metadata_loaded():
    """``task_metadata`` ClassVar is auto-loaded from task_metadata.json with 500 entries."""
    cfg = SWEBenchVerifiedBenchmarkConfig()
    assert cfg.benchmark_metadata.num_tasks == 500
    assert len(cfg.task_metadata) == 500
    sample = next(iter(cfg.task_metadata.values()))
    assert isinstance(sample, SWEBenchVerifiedTaskMetadata)
    # SWE-bench Verified specific fields are present
    assert sample.repo
    assert sample.base_commit


def test_get_task_configs_stamps_metadata():
    """Every emitted ``TaskConfig`` carries the full ``TaskMetadata`` (no task_id-only stub)."""
    cfg = SWEBenchVerifiedBenchmarkConfig().subset_from_list(_DEBUG_TASK_IDS)
    configs = list(cfg.get_task_configs())
    assert len(configs) == len(_DEBUG_TASK_IDS)
    for tc in configs:
        assert isinstance(tc, SWEBenchVerifiedTaskConfig)
        assert isinstance(tc.metadata, SWEBenchVerifiedTaskMetadata)
        assert tc.metadata.id == tc.task_id
        # Stamped metadata carries subclass-specific fields, not just base TaskMetadata.
        assert tc.metadata.repo


def test_subset_from_list():
    """``subset_from_list`` scopes the config to exactly the requested task IDs."""
    cfg = SWEBenchVerifiedBenchmarkConfig().subset_from_list(_DEBUG_TASK_IDS)
    assert cfg.task_ids == _DEBUG_TASK_IDS
    assert set(cfg.tasks().keys()) == set(_DEBUG_TASK_IDS)
    assert cfg.num_tasks == len(_DEBUG_TASK_IDS)


def test_debug_benchmark_type():
    """``get_debug_benchmark()`` returns a ``BenchmarkConfig`` (not a live ``Benchmark``).

    The harness owns ``config.install()`` and ``config.make(infra)``; the debug factory
    must not call either.
    """
    cfg = get_debug_benchmark()
    assert isinstance(cfg, SWEBenchVerifiedBenchmarkConfig)
    assert isinstance(cfg, BenchmarkConfig)
    assert cfg.oracle_mode is True
    # Scoped to the debug task subset
    assert cfg.task_ids == _DEBUG_TASK_IDS


def test_execution_info_roundtrip():
    """Typed ``SWEBenchVerifiedExecutionInfo`` round-trips through the TaskExecutionInfo discriminator."""
    ei = SWEBenchVerifiedExecutionInfo(
        problem_statement="test issue",
        patch="diff --git a/x b/x",
        test_patch="diff --git a/y b/y",
        fail_to_pass=["test_a", "test_b"],
        pass_to_pass=["test_c"],
    )
    assert isinstance(ei, TaskExecutionInfo)
    restored = SWEBenchVerifiedExecutionInfo.model_validate_json(ei.model_dump_json())
    assert restored.problem_statement == "test issue"
    assert restored.fail_to_pass == ["test_a", "test_b"]
    assert restored.eval_timeout == 1800  # default preserved


# --- container:root requirement (cube-harness#446) ---------------------------


class _FakeInfra(InfraConfig):
    """Minimal InfraConfig stub for capability-gate tests; declares a capability set."""

    caps: set[str] = set()

    def fingerprint(self) -> str:
        return "fake"

    def capabilities(self) -> set[str]:
        return self.caps

    def provision(self, resource: ResourceConfig) -> None: ...

    def launch(self, resource: ResourceConfig) -> ResourceHandle:
        raise NotImplementedError

    def list_active(self, run_id: str | None = None) -> list[ResourceHandle]:
        return []

    def cleanup(self, run_id: str) -> None: ...

    def cleanup_stale(self, max_age_seconds: int | None = None) -> list[str]:
        return []


def test_root_tasks_declare_container_root():
    """The known root-only tasks carry the container:root requirement; ordinary tasks don't."""
    cfg = SWEBenchVerifiedBenchmarkConfig()
    assert len(TASKS_REQUIRING_ROOT) == 6
    for task_id in TASKS_REQUIRING_ROOT:
        cc = cfg.task_metadata[task_id].container_config
        assert cc is not None and "container:root" in cc.requirements(), task_id
    control = cfg.task_metadata["astropy__astropy-12907"].container_config
    assert control is not None and "container:root" not in control.requirements()


def test_nonroot_infra_refuses_root_task_but_serves_others():
    """A non-root infra is gated off a root-only task, runs ordinary ones, and a root
    infra serves both. ``on_incompatible='force'`` bypasses the gate (→ #452 net)."""
    nonroot = _FakeInfra(caps={"docker", "network:egress"})
    rooted = _FakeInfra(caps={"docker", "network:egress", "container:root"})

    root_task = SWEBenchVerifiedBenchmarkConfig().subset_from_list(["psf__requests-1142"])
    with pytest.raises(IncompatibleInfraError):
        root_task._gate_infra_compatibility(nonroot)
    root_task._gate_infra_compatibility(rooted)  # root infra serves it — no raise

    ordinary = SWEBenchVerifiedBenchmarkConfig().subset_from_list(["astropy__astropy-12907"])
    ordinary._gate_infra_compatibility(nonroot)  # ordinary task is fine non-root — no raise

    _FakeInfra(caps={"docker"}, on_incompatible="force")  # force is a valid escape hatch

"""Docker-free unit tests for swegym-lite-cube — covers the BenchmarkConfig
contract (registry wiring, subsetting, metadata stamping, debug factory,
serialization round-trip).
"""

from __future__ import annotations

from cube.benchmark import BenchmarkConfig
from cube.resource import InfraConfig, ResourceConfig, ResourceHandle
from cube.task import TaskExecutionInfo

from swegym_lite_cube.benchmark import TASKS_REQUIRING_ROOT, SWEGymLiteBenchmarkConfig
from swegym_lite_cube.debug import _TASK_ACTIONS, get_debug_benchmark
from swegym_lite_cube.task import (
    SWEGymLiteExecutionInfo,
    SWEGymLiteTaskConfig,
    SWEGymLiteTaskMetadata,
)


_DEBUG_TASK_IDS = list(_TASK_ACTIONS)


def test_config_roundtrip():
    """``model_dump_json`` → ``model_validate_json`` produces an equivalent config."""
    cfg = SWEGymLiteBenchmarkConfig(oracle_mode=True).subset_from_list(_DEBUG_TASK_IDS)
    js = cfg.model_dump_json()
    restored = SWEGymLiteBenchmarkConfig.model_validate_json(js)
    assert restored.oracle_mode is True
    assert restored.task_ids == _DEBUG_TASK_IDS
    assert restored.num_tasks == len(_DEBUG_TASK_IDS)
    assert restored.benchmark_metadata.name == "swegym-lite-cube"


def test_task_metadata_loaded():
    """``task_metadata`` ClassVar is auto-loaded from task_metadata.json with 230 entries."""
    cfg = SWEGymLiteBenchmarkConfig()
    assert cfg.benchmark_metadata.num_tasks == 230
    assert len(cfg.task_metadata) == 230
    sample = next(iter(cfg.task_metadata.values()))
    assert isinstance(sample, SWEGymLiteTaskMetadata)
    # SWE-Gym Lite public fields are present
    assert sample.repo
    assert sample.base_commit
    assert sample.version
    # Images are the published SWE-Gym eval images under the xingyaoww namespace.
    assert sample.container_config is not None
    assert sample.container_config.image.startswith("xingyaoww/sweb.eval.x86_64.")


def test_get_task_configs_stamps_metadata():
    """Every emitted ``TaskConfig`` carries the full ``TaskMetadata`` (no task_id-only stub)."""
    cfg = SWEGymLiteBenchmarkConfig().subset_from_list(_DEBUG_TASK_IDS)
    configs = list(cfg.get_task_configs())
    assert len(configs) == len(_DEBUG_TASK_IDS)
    for tc in configs:
        assert isinstance(tc, SWEGymLiteTaskConfig)
        assert isinstance(tc.metadata, SWEGymLiteTaskMetadata)
        assert tc.metadata.id == tc.task_id
        # Stamped metadata carries subclass-specific fields, not just base TaskMetadata.
        assert tc.metadata.repo


def test_subset_from_list():
    """``subset_from_list`` scopes the config to exactly the requested task IDs."""
    cfg = SWEGymLiteBenchmarkConfig().subset_from_list(_DEBUG_TASK_IDS)
    assert cfg.task_ids == _DEBUG_TASK_IDS
    assert set(cfg.tasks().keys()) == set(_DEBUG_TASK_IDS)
    assert cfg.num_tasks == len(_DEBUG_TASK_IDS)


def test_debug_benchmark_type():
    """``get_debug_benchmark()`` returns a ``BenchmarkConfig`` (not a live ``Benchmark``).

    The harness owns ``config.install()`` and ``config.make(infra)``; the debug factory
    must not call either.
    """
    cfg = get_debug_benchmark()
    assert isinstance(cfg, SWEGymLiteBenchmarkConfig)
    assert isinstance(cfg, BenchmarkConfig)
    assert cfg.oracle_mode is True
    # Scoped to the debug task subset
    assert cfg.task_ids == _DEBUG_TASK_IDS


def test_execution_info_roundtrip():
    """Typed ``SWEGymLiteExecutionInfo`` round-trips through the TaskExecutionInfo discriminator."""
    ei = SWEGymLiteExecutionInfo(
        problem_statement="test issue",
        patch="diff --git a/x b/x",
        test_patch="diff --git a/y b/y",
        fail_to_pass=["test_a", "test_b"],
        pass_to_pass=["test_c"],
    )
    assert isinstance(ei, TaskExecutionInfo)
    restored = SWEGymLiteExecutionInfo.model_validate_json(ei.model_dump_json())
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


def test_no_tasks_require_root_yet():
    """SWE-Gym Lite ships no ``container:root`` tasks yet.

    Unlike swebench-verified-cube, the root-only set starts empty (the offending
    instances must be identified empirically; the runtime probe in reset() protects
    untagged tasks meanwhile — see TASKS_REQUIRING_ROOT). So no task metadata declares
    the requirement.
    """
    cfg = SWEGymLiteBenchmarkConfig()
    assert TASKS_REQUIRING_ROOT == frozenset()
    for tm in cfg.task_metadata.values():
        cc = tm.container_config
        assert cc is not None and "container:root" not in cc.requirements(), tm.id


def test_nonroot_infra_serves_ordinary_tasks():
    """With no root-only tasks, a non-root infra serves every debug task; ``force`` is a
    valid escape hatch on the capability gate."""
    nonroot = _FakeInfra(caps={"docker", "network:egress"})
    ordinary = SWEGymLiteBenchmarkConfig().subset_from_list(_DEBUG_TASK_IDS)
    ordinary._gate_infra_compatibility(nonroot)  # no container:root requirement → no raise
    _FakeInfra(caps={"docker"}, on_incompatible="force")  # force is a valid escape hatch

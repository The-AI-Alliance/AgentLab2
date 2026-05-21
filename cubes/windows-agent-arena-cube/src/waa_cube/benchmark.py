"""WAABenchmark / WAABenchmarkRuntime / WAATaskConfig — CUBE benchmark for WindowsAgentArena.

Two-class layout per cube-standard's BenchmarkConfig/Benchmark split:

  * ``WAABenchmark`` (BenchmarkConfig subclass) — Pydantic, serializable.
    Holds the static registry + user-configurable knobs (use_som, tool_config,
    resources). Vends ``WAATaskConfig`` objects via ``get_task_configs()``.

  * ``WAABenchmarkRuntime`` (Benchmark subclass) — runtime ABC instance returned
    by ``WAABenchmark.make(infra)``. Provides ``_setup`` / ``close`` and the
    ``install()`` hook that populates the per-task execution-info cache.

Usage:

    bench_config = WAABenchmark(tool_config=ComputerConfig(), use_som=False)
    bench = bench_config.make(infra=AzureInfraConfig(...))   # or LocalInfraConfig
    for task_config in bench_config.get_task_configs():
        task = task_config.make()
        ...
        task.close()

Filter by domain (uses lightweight TaskMetadata fields, no extra_info):

    vscode_config = bench_config.subset_from_glob("metadata.id", "vs_code/*")

Heavy per-task execution data (setup steps, evaluator config, related_apps,
…) lives in ``task_execution_info.json`` shipped next to this module, hydrated
into a typed ``WAATaskExecutionInfo`` by ``WAATaskConfig.make()`` so it
arrives on the worker as ``Task.execution_info``. ``task_metadata.json``
stays slim (~350 bytes/task).

Regenerate metadata + execution-info from the upstream WAA eval dir:

    python scripts/create_task_metadata.py --eval-dir /path/to/evaluation_examples_windows --force
"""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from typing import ClassVar

from cube import LocalInfraConfig
from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.resource import InfraConfig, ResourceConfig
from cube.task import TaskConfig, TaskMetadata
from pydantic import Field, SerializeAsAny

from waa_cube.azure import WAA_WINDOWS_RESOURCE
from waa_cube.task import WAATask, WAATaskExecutionInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WAATaskConfig
# ---------------------------------------------------------------------------


class WAATaskConfig(TaskConfig):
    """Serialisable config for a single WAA task.

    Fields:
        task_id:     inherited from TaskConfig (derived from metadata.id)
        metadata:    TaskMetadata embedded so Ray workers don't need to look
                     it up via the class-level registry.
        tool_config: inherited from TaskConfig
        seed:        inherited (ignored — tasks are deterministic)
        infra:       InfraConfig used to launch task VMs.
        use_som:     enable SoM screenshot annotation.
    """

    infra: InfraConfig | None = None
    use_som: bool = False

    def make(
        self,
        runtime_context: dict | None = None,
    ) -> WAATask:
        if self.tool_config is None:
            raise ValueError(
                f"WAATaskConfig for task '{self.task_id}' has no tool_config. "
                "Pass tool_config=ComputerConfig(...) to WAABenchmark."
            )
        exec_info_raw = self.load_task_execution_info()
        execution_info = WAATaskExecutionInfo.model_validate(exec_info_raw)
        return WAATask(
            metadata=self.metadata,
            execution_info=execution_info,
            tool_config=self.tool_config,
            infra=self.infra,
            use_som=self.use_som,
            runtime_context=runtime_context,
        )


# ---------------------------------------------------------------------------
# WAABenchmarkRuntime — runtime pair instantiated by WAABenchmark.make()
# ---------------------------------------------------------------------------


class WAABenchmarkRuntime(Benchmark):
    """Runtime pair for WAABenchmark.

    Holds no shared OS state — each WAA task launches its own VM via
    ``WAATaskConfig.infra.launch()``. ``_setup`` populates the per-task
    execution-info cache from the in-tree ``task_execution_info.json``;
    ``close`` is a no-op because there's no shared resource to release.
    """

    def _setup(self) -> None:
        """Populate per-task execution cache from the shipped
        ``task_execution_info.json`` and sweep any stale VMs from previous runs.

        Reads ``src/waa_cube/task_execution_info.json`` (a dict mapping
        ``task_id`` → execution-info fields) and writes one JSON file per task
        into ``WAATaskConfig.task_execution_cache_dir()`` so workers can hydrate
        ``WAATaskExecutionInfo`` via ``load_task_execution_info(task_id)``.
        """
        from waa_cube import _benchmark_data_dir

        if self._infra is not None:
            self._infra.cleanup_stale()

        exec_info_file = _benchmark_data_dir() / "task_execution_info.json"
        if not exec_info_file.exists():
            logger.warning("install(): %s missing — skipping execution-info cache", exec_info_file)
            return
        all_info = json.loads(exec_info_file.read_text())

        exec_cache_dir = WAATaskConfig.task_execution_cache_dir()
        exec_cache_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for task_id, info in all_info.items():
            cache_file = exec_cache_dir / f"{task_id}.json"
            new_content = json.dumps(info, indent=2)
            if cache_file.exists() and cache_file.read_text() == new_content:
                continue
            cache_file.write_text(new_content)
            written += 1
        logger.info("WAABenchmarkRuntime — wrote %d execution-info cache files to %s", written, exec_cache_dir)

    def close(self) -> None:
        """No shared resources to release — VMs are torn down per-task."""


# ---------------------------------------------------------------------------
# WAABenchmark (config)
# ---------------------------------------------------------------------------


class WAABenchmark(BenchmarkConfig):
    """CUBE benchmark configuration for WindowsAgentArena.

    Reference: https://github.com/microsoft/WindowsAgentArena

    Class-level attributes (required by cube.benchmark.BenchmarkConfig):
        benchmark_metadata:  ClassVar[BenchmarkMetadata]
        task_metadata:       ClassVar[dict[str, TaskMetadata]]  (auto-loaded from task_metadata.json)
        task_config_class:   ClassVar[type[TaskConfig]]         = WAATaskConfig
        benchmark_class:     ClassVar[type[Benchmark]]          = WAABenchmarkRuntime

    Instance fields:
        tool_config: ComputerConfig — applied to every emitted WAATaskConfig
        use_som:     bool           — enable SoM screenshot annotation
    """

    # ── ClassVars ───────────────────────────────────────────────────────────

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="waa",
        version="1.0.0",
        description="WindowsAgentArena: Benchmarking AI Agents on Windows 11",
        authors=["Rogerio Bonatti et al."],
        license="MIT",
        requirements={
            "vm": "Windows 11 (infra-managed VM)",
            "ram_gb": 8,
            "disk_gb": 60,
        },
        num_tasks=154,
        tags=["desktop", "gui", "windows", "multimodal"],
    )

    # task_metadata is auto-loaded from task_metadata.json next to this module
    # by BenchmarkConfig.__init_subclass__. Do not assign a default value here
    # — that would suppress the auto-load.
    task_metadata: ClassVar[dict[str, TaskMetadata]]
    task_config_class: ClassVar[type[TaskConfig]] = WAATaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = WAABenchmarkRuntime

    # ── Instance fields ─────────────────────────────────────────────────────

    use_som: bool = False
    """Enable Set-of-Marks annotation on each observation."""

    tasks_file: str | None = None
    """Optional flat JSON task list for debug overlay (merged on top of shipped metadata)."""

    infra: SerializeAsAny[InfraConfig] = Field(default_factory=LocalInfraConfig)
    """InfraConfig used to launch per-task VMs (defaults to LocalInfraConfig).

    Tasks need this at construction time — each task launches its own VM —
    so we keep it as a config field. ``make()`` defaults its ``infra``
    parameter to this value when none is passed explicitly.
    """

    # Resources are declared so BenchmarkConfig.make(infra) provisions the
    # gallery image idempotently before vending any tasks.
    resources: list[ResourceConfig] = [WAA_WINDOWS_RESOURCE]

    # ── make() — default to self.infra when no explicit infra is passed ─────

    def make(self, infra: InfraConfig | None = None) -> "WAABenchmarkRuntime":
        """Provision resources + return a live ``WAABenchmarkRuntime``.

        If ``infra`` is None, falls back to ``self.infra`` (set on construction)
        so recipe authors can write::

            WAABenchmark(infra=AzureInfraConfig(...), tool_config=...).make()
        """
        return super().make(infra=infra or self.infra)  # type: ignore[return-value]

    # ── get_task_configs() — vend WAATaskConfig with infra + use_som baked in ─

    def get_task_configs(self) -> Generator[WAATaskConfig, None, None]:
        for tm in self.tasks().values():
            yield WAATaskConfig(
                metadata=tm,
                tool_config=self.tool_config,
                seed=None,
                infra=self.infra,
                use_som=self.use_som,
            )

    # ── Debug overlay (used by `cube test waa-cube`) ────────────────────────

    def _load_task_metadata_from_file(self, tasks_file: str) -> dict[str, TaskMetadata]:
        """Load TaskMetadata from a flat JSON list and write per-task
        execution-info cache entries (used by the debug suite)."""
        with open(tasks_file) as f:
            tasks: list[dict] = json.load(f)
        exec_cache_dir = WAATaskConfig.task_execution_cache_dir()
        exec_cache_dir.mkdir(parents=True, exist_ok=True)
        result: dict[str, TaskMetadata] = {}
        for td in tasks:
            task_id = td.get("id", "")
            if not task_id:
                logger.warning("Skipping task with missing 'id' in %s", tasks_file)
                continue
            result[task_id] = TaskMetadata(
                id=task_id,
                abstract_description=td.get("instruction", ""),
            )
            exec_info_dict = {
                "_type": "waa_cube.task.WAATaskExecutionInfo",
                "domain": td.get("domain", "debug"),
                "snapshot": td.get("snapshot", "init_state"),
                "config": td.get("config", []),
                "evaluator": td.get("evaluator", {}),
                "related_apps": td.get("related_apps", []),
                "test_sets": td.get("test_sets", []),
            }
            (exec_cache_dir / f"{task_id}.json").write_text(json.dumps(exec_info_dict, indent=2))
        logger.info("Loaded %d task metadata entries from %s", len(result), tasks_file)
        return result

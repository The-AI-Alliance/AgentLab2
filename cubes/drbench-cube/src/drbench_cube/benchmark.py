"""CUBE Benchmark implementation for DRBench.

DrBenchBenchmarkConfig is the CUBE registry entry (BenchmarkConfig subclass)
that auto-loads benchmark_metadata.json and task_metadata.json from the same
directory via CUBE's __init_subclass__ hook.

DrBenchBenchmark is the runtime pair (Benchmark subclass) that implements
the shared lifecycle (_setup / close).
"""

from __future__ import annotations

from typing import ClassVar

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.task import TaskConfig, TaskMetadata

from drbench_cube.task import DrBenchTaskConfig


class DrBenchTaskMetadata(TaskMetadata):
    """Per-task metadata for DRBench, extending the base with typed fields.

    abstract_description (inherited) holds the full DR question text.
    """

    domain: str = ""
    difficulty: str = ""
    company_name: str = ""
    company_industry: str = ""
    persona_name: str = ""
    persona_role: str = ""
    insight_count: int = 0


class DrBenchBenchmark(Benchmark["DrBenchBenchmarkConfig"]):
    """Runtime benchmark — each task gets its own container; no shared infra."""

    def _setup(self) -> None:
        pass

    def close(self) -> None:
        pass


class DrBenchBenchmarkConfig(BenchmarkConfig[DrBenchTaskMetadata]):
    """
    CUBE-compliant benchmark config wrapping DRBench.

    benchmark_metadata and task_metadata are auto-loaded from the JSON files
    in this directory by CUBE's __init_subclass__ hook. task_metadata.json
    entries carry _type = "drbench_cube.benchmark.DrBenchTaskMetadata" so
    the loader instantiates the typed subclass via TypedBaseModel's discriminator.
    """

    benchmark_metadata: ClassVar[BenchmarkMetadata]
    task_metadata: ClassVar[dict[str, TaskMetadata]]
    task_config_class: ClassVar[type[TaskConfig]] = DrBenchTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = DrBenchBenchmark

    def get_debug_task_configs(self) -> list[DrBenchTaskConfig]:
        """Return debug task configs for CI / smoke tests (DR0001 from val subset)."""
        return [DrBenchTaskConfig(task_id="DR0001")]

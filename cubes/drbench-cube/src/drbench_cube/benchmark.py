"""CUBE Benchmark implementation for DRBench.

DrBenchBenchmark is the CUBE registry entry for DRBench. It auto-loads
benchmark_metadata.json and task_metadata.json from the same directory.
"""

from __future__ import annotations

from typing import ClassVar, Generator

from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.task import TaskConfig, TaskMetadata

from drbench_cube.task import DrBenchTaskConfig


class DrBenchBenchmark(Benchmark):
    """
    CUBE-compliant benchmark wrapping DRBench.

    benchmark_metadata and task_metadata are auto-loaded from the JSON files
    in this directory by CUBE's __init_subclass__ hook.
    """

    # CUBE's __init_subclass__ will populate these from the JSON files
    # next to this file (benchmark_metadata.json, task_metadata.json).
    benchmark_metadata: ClassVar[BenchmarkMetadata]
    task_metadata: ClassVar[dict[str, TaskMetadata]]
    task_config_class: ClassVar[type[TaskConfig]] = DrBenchTaskConfig

    def _setup(self) -> None:
        # No shared infrastructure needed — each task gets its own container.
        pass

    def close(self) -> None:
        pass

    def get_debug_task_configs(self) -> list[DrBenchTaskConfig]:
        """Return debug task configs for CI / smoke tests (DR0001 from val subset)."""
        return [DrBenchTaskConfig(task_id="DR0001")]

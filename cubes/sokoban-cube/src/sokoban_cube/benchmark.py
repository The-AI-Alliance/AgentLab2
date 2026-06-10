"""Benchmark for sokoban_cube.

Tasks are fully self-contained — no shared infrastructure — so the runtime
``_setup`` / ``close`` are no-ops. Metadata uses **Option B**: ``benchmark.py``
defines no inline ``benchmark_metadata`` / ``task_metadata`` ClassVars, so the
framework auto-loads them from the JSON files next to this module:

    src/sokoban_cube/benchmark_metadata.json
    src/sokoban_cube/task_metadata.json   ← regenerate with scripts/create_task_metadata.py

Each task_metadata entry carries ``_type`` =
``sokoban_cube.task.SokobanTaskMetadata`` so it deserialises into the typed
metadata subclass.
"""

from typing import ClassVar

from cube.benchmark import Benchmark, BenchmarkConfig
from cube.task import TaskConfig

from sokoban_cube.task import SokobanTaskConfig, SokobanTaskMetadata


class SokobanBenchmark(Benchmark["SokobanBenchmarkConfig"]):
    """Runtime pair — Sokoban needs no shared infrastructure."""

    def _setup(self) -> None:
        pass

    def close(self) -> None:
        pass


class SokobanBenchmarkConfig(BenchmarkConfig[SokobanTaskMetadata]):
    """Registry of Sokoban levels. Metadata auto-loaded from JSON (Option B)."""

    task_config_class: ClassVar[type[TaskConfig]] = SokobanTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = SokobanBenchmark

    # PipelineRL's cube_rl loop drives one benchmark config through
    # install() + setup() + get_task_configs() + close(). Sokoban has no shared
    # infrastructure, so setup/close are no-ops (install/get_task_configs come
    # from the base).
    def setup(self) -> None:
        pass

    def close(self) -> None:
        pass

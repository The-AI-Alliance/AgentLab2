from typing import ClassVar

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.task import TaskConfig, TaskMetadata

from arithmetic_cube.task import ArithmeticTaskConfig, ArithmeticTaskMetadata


class ArithmeticBenchmark(Benchmark["ArithmeticBenchmarkConfig"]):
    """Runtime pair — arithmetic tasks need no shared infrastructure."""

    def _setup(self) -> None:
        pass

    def close(self) -> None:
        pass


class ArithmeticBenchmarkConfig(BenchmarkConfig[ArithmeticTaskMetadata]):
    """Registry of arithmetic tasks — no shared infrastructure needed."""

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="arithmetic-cube",
        version="0.1.0",
        description="Simple arithmetic tasks — submit the correct integer answer",
        num_tasks=4,
        tags=["example", "arithmetic"],
    )

    # Typed values; declared with the base type to satisfy ClassVar invariance.
    # ``BenchmarkConfig[ArithmeticTaskMetadata]`` narrows the read view via ``tasks()``.
    task_metadata: ClassVar[dict[str, TaskMetadata]] = {
        "add-3-4": ArithmeticTaskMetadata(
            id="add-3-4",
            abstract_description="Compute 3 + 4 and submit the answer",
            recommended_max_steps=2,
            a=3,
            b=4,
            op="+",
            expected=7,
        ),
        "sub-10-3": ArithmeticTaskMetadata(
            id="sub-10-3",
            abstract_description="Compute 10 - 3 and submit the answer",
            recommended_max_steps=2,
            a=10,
            b=3,
            op="-",
            expected=7,
        ),
        "mul-6-7": ArithmeticTaskMetadata(
            id="mul-6-7",
            abstract_description="Compute 6 × 7 and submit the answer",
            recommended_max_steps=2,
            a=6,
            b=7,
            op="*",
            expected=42,
        ),
        "add-100-1": ArithmeticTaskMetadata(
            id="add-100-1",
            abstract_description="Compute 100 + 1 and submit the answer",
            recommended_max_steps=2,
            a=100,
            b=1,
            op="+",
            expected=101,
        ),
    }

    task_config_class: ClassVar[type[TaskConfig]] = ArithmeticTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = ArithmeticBenchmark

    # PipelineRL cube_rl drives one benchmark object through
    # install() + setup() + get_task_configs() + close(). Arithmetic needs no shared
    # infrastructure, so setup/close are no-ops (install/get_task_configs come from the base).
    def setup(self) -> None:
        pass

    def close(self) -> None:
        pass

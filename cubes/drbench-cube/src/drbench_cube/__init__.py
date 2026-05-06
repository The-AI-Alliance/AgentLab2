"""DRBench CUBE compliance layer.

Provides CUBE-compliant Benchmark, Task, Tool, and Container implementations
that wrap DRBench's existing infrastructure.

Install with: pip install -e . (from the drbench-cube directory)
"""

from drbench_cube.benchmark import DrBenchBenchmark, DrBenchBenchmarkConfig, DrBenchTaskMetadata
from drbench_cube.container import DrBenchContainer, DrBenchContainerBackend
from drbench_cube.task import DrBenchTask, DrBenchTaskConfig
from drbench_cube.tool import DrBenchTool, DrBenchToolConfig

__all__ = [
    "DrBenchBenchmark",
    "DrBenchBenchmarkConfig",
    "DrBenchTaskMetadata",
    "DrBenchContainer",
    "DrBenchContainerBackend",
    "DrBenchTask",
    "DrBenchTaskConfig",
    "DrBenchTool",
    "DrBenchToolConfig",
]

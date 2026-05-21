from pathlib import Path

from waa_cube.benchmark import WAABenchmark, WAABenchmarkRuntime, WAATaskConfig
from waa_cube.computer import ComputerConfig
from waa_cube.configs import WAA_CONFIGS
from waa_cube.task import WAATask, WAATaskExecutionInfo


def _benchmark_data_dir() -> Path:
    """Directory containing shipped JSON files (task_metadata.json, …)."""
    return Path(__file__).parent


__all__ = [
    "WAA_CONFIGS",
    "WAABenchmark",
    "WAABenchmarkRuntime",
    "WAATaskConfig",
    "WAATask",
    "WAATaskExecutionInfo",
    "ComputerConfig",
]

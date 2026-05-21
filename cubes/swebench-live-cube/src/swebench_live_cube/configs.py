"""Canonical SWE-bench Live benchmark configs.

from swebench_live_cube import SWEBENCH_LIVE_CONFIGS
benchmark = SWEBENCH_LIVE_CONFIGS["default"]
"""

from cube.core import ConfigRegistry
from swebench_live_cube.benchmark import SWEBenchLiveBenchmarkConfig

SWEBENCH_LIVE_CONFIGS: ConfigRegistry[SWEBenchLiveBenchmarkConfig] = ConfigRegistry(
    {
        "default": SWEBenchLiveBenchmarkConfig(),
    }
)

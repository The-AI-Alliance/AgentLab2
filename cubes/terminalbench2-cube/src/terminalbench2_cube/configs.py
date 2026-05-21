"""Canonical Terminal-Bench 2 benchmark configs.

from terminalbench2_cube import TERMINALBENCH2_CONFIGS
benchmark = TERMINALBENCH2_CONFIGS["default"]
"""

from cube.core import ConfigRegistry
from terminalbench2_cube.benchmark import TerminalBench2BenchmarkConfig

TERMINALBENCH2_CONFIGS: ConfigRegistry[TerminalBench2BenchmarkConfig] = ConfigRegistry(
    {
        "default": TerminalBench2BenchmarkConfig(),
    }
)

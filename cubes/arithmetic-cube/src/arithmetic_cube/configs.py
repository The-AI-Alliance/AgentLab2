"""Canonical Arithmetic benchmark configs.

from arithmetic_cube import ARITHMETIC_CONFIGS
benchmark = ARITHMETIC_CONFIGS["default"]
"""

from cube.core import ConfigRegistry
from arithmetic_cube.benchmark import ArithmeticBenchmarkConfig

ARITHMETIC_CONFIGS: ConfigRegistry[ArithmeticBenchmarkConfig] = ConfigRegistry(
    {
        "default": ArithmeticBenchmarkConfig(),
    }
)

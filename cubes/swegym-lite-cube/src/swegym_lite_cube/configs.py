"""Canonical SWE-Gym Lite benchmark configs.

    from swegym_lite_cube import SWEGYM_LITE_CONFIGS
    benchmark = SWEGYM_LITE_CONFIGS["default"]

Only the standard, score-comparable configuration is canonical. For a
non-canonical task subset, clone the recipe and chain the existing
BenchmarkConfig helpers (subset_from_list / subset_from_glob / named_subset).
"""

from cube.core import ConfigRegistry
from swegym_lite_cube.benchmark import SWEGymLiteBenchmarkConfig

SWEGYM_LITE_CONFIGS: ConfigRegistry[SWEGymLiteBenchmarkConfig] = ConfigRegistry(
    {
        "default": SWEGymLiteBenchmarkConfig(),
    }
)

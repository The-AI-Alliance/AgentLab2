"""Canonical SWE-Gym benchmark configs.

    from swegym_cube import SWEGYM_CONFIGS
    benchmark = SWEGYM_CONFIGS["default"]   # full 2438-task SWE-Gym
    benchmark = SWEGYM_CONFIGS["lite"]      # official 230-task lite subset

``default`` is the full 2438-task SWE-Gym training set; ``lite`` is the official
230-task score-comparable subset (the upstream SWE-Gym-Lite split). For any other
task subset, clone the recipe and chain the existing BenchmarkConfig helpers
(subset_from_list / subset_from_glob / named_subset).
"""

from cube.core import ConfigRegistry
from swegym_cube.benchmark import SWEGymBenchmarkConfig

SWEGYM_CONFIGS: ConfigRegistry[SWEGymBenchmarkConfig] = ConfigRegistry(
    {
        "default": SWEGymBenchmarkConfig(),
        "lite": SWEGymBenchmarkConfig().named_subset("lite"),
    }
)

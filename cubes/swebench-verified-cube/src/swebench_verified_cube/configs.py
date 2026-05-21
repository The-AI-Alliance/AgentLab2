"""Canonical SWE-bench Verified benchmark configs.

    from swebench_verified_cube import SWEBENCH_CONFIGS
    benchmark = SWEBENCH_CONFIGS["default"]

Only the standard, score-comparable configuration is canonical. For a
non-canonical task subset, clone the recipe and chain the existing
BenchmarkConfig helpers (subset_from_list / subset_from_glob / named_subset).
"""

from cube.core import ConfigRegistry
from swebench_verified_cube.benchmark import SWEBenchVerifiedBenchmarkConfig

SWEBENCH_CONFIGS: ConfigRegistry[SWEBenchVerifiedBenchmarkConfig] = ConfigRegistry(
    {
        "default": SWEBenchVerifiedBenchmarkConfig(),
    }
)

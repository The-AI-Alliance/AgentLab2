"""Canonical BrowseComp benchmark configs.

    from browsercomp_cube import BROWSECOMP_CONFIGS
    benchmark = BROWSECOMP_CONFIGS["default"]

The ``default`` config grades with ``scorer_model="openai/gpt-4o"`` (override
per-experiment, e.g. ``BrowseCompBenchmarkConfig(scorer_model=...)``). The cube's
intrinsic tool is answer submission; richer browsing tools are a recipe-side
concern (clone the recipe and extend the ToolboxConfig).
"""

from cube.core import ConfigRegistry
from browsercomp_cube.benchmark import BrowseCompBenchmarkConfig

BROWSECOMP_CONFIGS: ConfigRegistry[BrowseCompBenchmarkConfig] = ConfigRegistry(
    {"default": BrowseCompBenchmarkConfig()}
)

"""Canonical BrowseComp benchmark configs.

    from browsercomp_cube import BrowseCompBenchmarkConfig
    benchmark = BrowseCompBenchmarkConfig(scorer_model="openai/gpt-4o")

``scorer_model`` is required — the judge model used to grade answers — and has no
sensible default, so the cube ships no pre-built default config. The cube's
intrinsic tool is answer submission; richer browsing tools are a recipe-side
concern (clone the recipe and extend the ToolboxConfig).
"""

from cube.core import ConfigRegistry
from browsercomp_cube.benchmark import BrowseCompBenchmarkConfig

BROWSECOMP_CONFIGS: ConfigRegistry[BrowseCompBenchmarkConfig] = ConfigRegistry({})

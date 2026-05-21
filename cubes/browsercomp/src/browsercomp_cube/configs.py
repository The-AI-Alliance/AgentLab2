"""Canonical BrowseComp benchmark configs.

    from browsercomp_cube import BROWSECOMP_CONFIGS
    benchmark = BROWSECOMP_CONFIGS["default"]

The cube's intrinsic tool is answer submission; richer browsing tools are a
recipe-side concern (clone the recipe and extend the ToolboxConfig).
"""

from cube.core import ConfigRegistry
from cube.tool import ToolboxConfig
from browsercomp_cube.benchmark import BrowseCompBenchmarkConfig
from browsercomp_cube.tool import SubmitAnswerToolConfig

BROWSECOMP_CONFIGS: ConfigRegistry[BrowseCompBenchmarkConfig] = ConfigRegistry(
    {
        "default": BrowseCompBenchmarkConfig(tool_config=ToolboxConfig(tool_configs=[SubmitAnswerToolConfig()])),
    }
)

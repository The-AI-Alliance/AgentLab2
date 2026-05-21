"""Canonical WebArena-Verified benchmark configs.

    from webarena_verified_cube import WEBARENA_CONFIGS
    benchmark = WEBARENA_CONFIGS["default"]

WEBARENA_ALL is the primary resource (one provision/launch for all sites).
Runs on BrowserGym with axtree + screenshot (the canonical web observation).
"""

from cube.core import ConfigRegistry
from cube_browser_tool.bgym_tool import BgymToolConfig
from webarena_verified_cube.benchmark import WebArenaVerifiedBenchmarkConfig
from webarena_verified_cube.resources import WEBARENA_ALL

WEBARENA_CONFIGS: ConfigRegistry[WebArenaVerifiedBenchmarkConfig] = ConfigRegistry(
    {
        "default": WebArenaVerifiedBenchmarkConfig(
            tool_config=BgymToolConfig(use_html=False, use_axtree=True, use_screenshot=True),
            resources=[WEBARENA_ALL],
        ),
    }
)

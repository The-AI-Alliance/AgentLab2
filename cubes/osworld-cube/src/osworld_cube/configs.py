"""Canonical OSWorld benchmark configs.

    from osworld_cube import OSWORLD_CONFIGS
    benchmark = OSWORLD_CONFIGS["default"]

"default" uses zero-arg OSWorldBenchmarkConfig (ComputerConfig() + the
OSWORLD_UBUNTU resource). "a11y" is the accessibility-tree + pyautogui
setup used for the reference OSWorld runs.
"""

from cube.core import ConfigRegistry
from osworld_cube.benchmark import OSWorldBenchmarkConfig
from osworld_cube.computer import ComputerConfig

OSWORLD_CONFIGS: ConfigRegistry[OSWorldBenchmarkConfig] = ConfigRegistry(
    {
        "default": OSWorldBenchmarkConfig(),
        "a11y": OSWorldBenchmarkConfig(
            tool_config=ComputerConfig(
                action_space="pyautogui",
                require_a11y_tree=True,
                observe_after_action=True,
            ),
            use_som=False,
        ),
    }
)

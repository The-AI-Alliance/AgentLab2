"""Canonical KNOWS benchmark configs.

    from knows_cube import KNOWS_CONFIGS
    benchmark = KNOWS_CONFIGS["default"]

All configs run the BrowserGym browser tool with axtree + screenshot (the
canonical web observation) plus the submit/infeasible terminator.

``"default"`` excludes tasks whose upstream evaluator cannot run; use ``"all"``
to include them.

Note: ``ConfigRegistry.__getitem__`` deep-copies on every lookup, so
``KNOWS_CONFIGS["default"].field = x`` silently no-ops. Bind to a variable first.
"""

from cube.core import ConfigRegistry

from knows_cube.benchmark import KnowsBenchmarkConfig
from knows_cube.task import default_tool_config

KNOWS_CONFIGS: ConfigRegistry[KnowsBenchmarkConfig] = ConfigRegistry(
    {
        "all": KnowsBenchmarkConfig(tool_config=default_tool_config()),
        "default": KnowsBenchmarkConfig(tool_config=default_tool_config()).named_subset("gradeable"),
        "docs": KnowsBenchmarkConfig(tool_config=default_tool_config()).named_subset("docs"),
        "sheets": KnowsBenchmarkConfig(tool_config=default_tool_config()).named_subset("sheets"),
        "slides": KnowsBenchmarkConfig(tool_config=default_tool_config()).named_subset("slides"),
    }
)

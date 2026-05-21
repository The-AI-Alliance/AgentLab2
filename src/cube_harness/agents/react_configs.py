"""Canonical ReAct configs.

    from cube_harness.agents.react_configs import REACT_CONFIGS
    agent = REACT_CONFIGS["default"]
    agent.llm_config.temperature = 0.5

Every lookup returns a fresh deep copy (see `ConfigRegistry`).
"""

from cube.core import ConfigRegistry

from cube_harness.agents.react import ReactAgentConfig
from cube_harness.llm import LLMConfig

_DEFAULT = ReactAgentConfig(llm_config=LLMConfig(model_name="gpt-5-mini", temperature=1.0))

REACT_CONFIGS: ConfigRegistry[ReactAgentConfig] = ConfigRegistry({"default": _DEFAULT})

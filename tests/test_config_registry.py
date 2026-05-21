"""ConfigRegistry copy-on-access + canonical registry validity."""

import pytest
from cube.core import ConfigRegistry
from pydantic import ValidationError

from cube_harness.agents.genny_configs import GENNY_CONFIGS
from cube_harness.agents.react_configs import REACT_CONFIGS
from cube_harness.infra import INFRA_CONFIGS
from cube_harness.llm import LLMConfig


def test_lookup_returns_independent_deep_copy() -> None:
    reg: ConfigRegistry[LLMConfig] = ConfigRegistry({"x": LLMConfig(model_name="m")})
    a = reg["x"]
    b = reg["x"]
    assert a is not b
    a.temperature = 0.123
    assert reg["x"].temperature != 0.123  # shared instance untouched


def test_unknown_name_lists_available() -> None:
    reg: ConfigRegistry[LLMConfig] = ConfigRegistry({"x": LLMConfig(model_name="m")})
    with pytest.raises(KeyError, match=r"Unknown config 'y'. Available: \['x'\]"):
        reg["y"]


def test_canonical_registries_are_constructible() -> None:
    for reg in (GENNY_CONFIGS, REACT_CONFIGS, INFRA_CONFIGS):
        assert len(reg) >= 1
        for name in reg:
            assert reg[name] is not None
    assert "local" in INFRA_CONFIGS
    assert {"default", "swe"} <= set(GENNY_CONFIGS)


def test_validated_assignment_on_canonical_config() -> None:
    agent = GENNY_CONFIGS["swe"]
    agent.budget.cost_limit = 2.0  # nested ValidatedConfig
    assert agent.budget.cost_limit == 2.0
    with pytest.raises(ValidationError):
        agent.budget.cost_limit = "free"  # type: ignore[assignment]

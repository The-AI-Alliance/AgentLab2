"""Config field-validation: fat-fingered values must fail FAST (at construction),
not silently or only at run time.

Before these constraints, e.g. `max_obs_chars=0` silently blinded the agent every
step, `obs_format="typo"` was silently treated as "raw", and `Experiment(max_steps=0)`
ran a zero-step episode — all with no error.
"""

import pytest
from pydantic import ValidationError

from cube_harness.agents.genny import GennyConfig
from cube_harness.agents.genny_configs import GENNY_CONFIGS
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig
from tests.conftest import MockAgentConfig, MockCubeBenchmarkConfig


def _swe() -> GennyConfig:
    return GENNY_CONFIGS["swe"]


def test_llm_model_name_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        LLMConfig(model_name="")
    LLMConfig(model_name="azure/gpt-5.4-mini")  # valid still constructs


@pytest.mark.parametrize(
    "field, bad",
    [
        ("obs_format", "nonsense"),  # not a Literal member → was silently treated as "raw"
        ("max_obs_chars", 0),  # 0 truncates every obs to "… [truncated]" → blinds the agent
        ("compact_threshold_chars", -5),
        ("display_budget_every_k", -1),
        ("max_format_errors", -1),
    ],
)
def test_genny_config_rejects_bad_values(field: str, bad: object) -> None:
    # validate_assignment=True (inherited) → assignment is the recipe pattern, must validate.
    with pytest.raises(ValidationError):
        setattr(_swe(), field, bad)


def test_genny_config_valid_values_still_work() -> None:
    a = _swe()
    a.obs_format = "output_tag"
    a.max_obs_chars = 1
    a.compact_threshold_chars = 10_000
    a.display_budget_every_k = 0  # 0 = disable (allowed)
    a.max_format_errors = 0  # 0 = no retry (allowed)


@pytest.mark.parametrize(
    "field, bad",
    [
        ("max_steps", 0),
        ("max_steps", -1),
        ("max_cost_usd", 0),
        ("max_cost_usd", -1),
        ("max_retries", -1),
    ],
)
def test_experiment_rejects_bad_budget(field: str, bad: object) -> None:
    with pytest.raises(ValidationError):
        Experiment(
            name="t",
            agent_config=MockAgentConfig(),
            benchmark_config=MockCubeBenchmarkConfig(),
            **{field: bad},
        )


def test_experiment_valid_budget_still_works() -> None:
    Experiment(
        name="t",
        agent_config=MockAgentConfig(),
        benchmark_config=MockCubeBenchmarkConfig(),
        max_steps=1,
        max_cost_usd=0.5,
        max_retries=0,
    )

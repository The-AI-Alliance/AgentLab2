"""Tests for cube_harness.analyze.inspect_results module."""

import numpy as np
import pytest
from cube.core import EnvironmentOutput, Observation, StepError

from cube_harness.analyze.inspect_results import (
    _extract_error_from_trajectory,
    agent_configs_to_df,
    error_report,
    format_agent_comparison,
    get_constants_and_variables,
    trajectories_to_df,
)
from cube_harness.core import (
    AgentOutput,
    Trajectory,
    TrajectoryStep,
)


def _make_trajectory(
    agent_name: str,
    task_name: str,
    model: str,
    reward: float,
    episode: int,
    error: StepError | None = None,
) -> Trajectory:
    steps = [
        TrajectoryStep(output=EnvironmentOutput(obs=Observation.from_text("start"), reward=0.0, done=False)),
        TrajectoryStep(output=AgentOutput(actions=[], error=error)),
        TrajectoryStep(output=EnvironmentOutput(obs=Observation.from_text("end"), reward=reward, done=True)),
    ]
    return Trajectory(
        id=f"{task_name}_ep{episode}",
        steps=steps,
        metadata={"agent_name": agent_name, "task_name": task_name, "model": model, "max_steps": 10},
        start_time=1000.0,
        end_time=1010.0,
        reward_info={"reward": reward},
    )


@pytest.fixture
def single_agent_trajectories() -> list[Trajectory]:
    return [
        _make_trajectory("react", "miniwob.click-test", "gpt-4", 1.0, 0),
        _make_trajectory("react", "miniwob.click-test", "gpt-4", 0.0, 1),
        _make_trajectory("react", "miniwob.login", "gpt-4", 1.0, 2),
        _make_trajectory("react", "miniwob.login", "gpt-4", 1.0, 3),
    ]


@pytest.fixture
def multi_agent_trajectories() -> list[Trajectory]:
    return [
        _make_trajectory("react", "miniwob.click-test", "gpt-4", 1.0, 0),
        _make_trajectory("react", "miniwob.click-test", "gpt-4", 1.0, 1),
        _make_trajectory("react", "miniwob.click-test", "claude-3", 0.0, 2),
        _make_trajectory("react", "miniwob.click-test", "claude-3", 0.0, 3),
    ]


@pytest.fixture
def trajectories_with_errors() -> list[Trajectory]:
    err = StepError(
        error_type="RuntimeError",
        exception_str="Connection timed out",
        stack_trace="Traceback...\nRuntimeError: Connection timed out",
    )
    return [
        _make_trajectory("react", "miniwob.click-test", "gpt-4", 1.0, 0),
        _make_trajectory("react", "miniwob.login", "gpt-4", 0.0, 1, error=err),
        _make_trajectory("react", "miniwob.login", "gpt-4", 0.0, 2, error=err),
    ]


class TestTrajectoriesToDf:
    def test_returns_none_for_empty_list(self):
        assert trajectories_to_df([]) is None

    def test_columns_present(self, single_agent_trajectories):
        df = trajectories_to_df(single_agent_trajectories)
        for col in ["trajectory_id", "agent_name", "task_name", "model", "cum_reward", "n_steps", "status", "err_msg"]:
            assert col in df.columns

    def test_row_count_matches_trajectories(self, single_agent_trajectories):
        df = trajectories_to_df(single_agent_trajectories)
        assert len(df) == 4

    def test_metadata_spread_as_columns(self, single_agent_trajectories):
        df = trajectories_to_df(single_agent_trajectories)
        assert df["agent_name"].iloc[0] == "react"
        assert df["max_steps"].iloc[0] == 10

    def test_rewards_computed(self, single_agent_trajectories):
        df = trajectories_to_df(single_agent_trajectories)
        assert list(df["cum_reward"]) == [1.0, 0.0, 1.0, 1.0]

    def test_metadata_only_stubs(self):
        traj = Trajectory(id="stub", metadata={"agent_name": "x"}, start_time=0.0, end_time=1.0)
        df = trajectories_to_df([traj])
        assert np.isnan(df["cum_reward"].iloc[0])
        assert not df["done"].iloc[0]


class TestExtractError:
    def test_no_error(self, single_agent_trajectories):
        msg, trace = _extract_error_from_trajectory(single_agent_trajectories[0])
        assert msg is None
        assert trace is None

    def test_extracts_last_error(self, trajectories_with_errors):
        msg, trace = _extract_error_from_trajectory(trajectories_with_errors[1])
        assert msg == "Connection timed out"
        assert "Traceback" in trace


class TestGetConstantsAndVariables:
    def test_single_agent_constants(self, single_agent_trajectories):
        df = trajectories_to_df(single_agent_trajectories)
        constants, variables, _ = get_constants_and_variables(df)
        assert "agent_name" in constants
        assert "model" in constants
        assert constants["agent_name"] == "react"
        assert "task_name" in variables

    def test_multi_agent_model_is_variable(self, multi_agent_trajectories):
        df = trajectories_to_df(multi_agent_trajectories)
        constants, variables, _ = get_constants_and_variables(df)
        assert "model" in variables
        assert "agent_name" in constants

    def test_drop_constants(self, single_agent_trajectories):
        df = trajectories_to_df(single_agent_trajectories)
        _, _, filtered = get_constants_and_variables(df, drop_constants=True)
        assert "agent_name" not in filtered.columns


class TestErrorReport:
    def test_no_errors(self, single_agent_trajectories):
        df = trajectories_to_df(single_agent_trajectories)
        result = error_report(df)
        assert result == "No errors found."

    def test_groups_errors(self, trajectories_with_errors):
        df = trajectories_to_df(trajectories_with_errors)
        result = error_report(df)
        assert "2x" in result
        assert "Connection timed out" in result
        assert "Traceback" in result


class TestFormatAgentComparison:
    _gpt4o_cfg = {
        "_type": "ReactAgentConfig",
        "llm_config": {"model_name": "gpt-4o", "temperature": 1.0, "max_tokens": 8192},
        "max_actions": 10,
        "can_finish": True,
    }
    _mini_cfg = {
        "_type": "ReactAgentConfig",
        "llm_config": {"model_name": "gpt-4o-mini", "temperature": 1.0, "max_tokens": 8192},
        "max_actions": 10,
        "can_finish": True,
    }

    def test_constants_are_shared_params(self) -> None:
        df = agent_configs_to_df([("Agent-gpt-4o", self._gpt4o_cfg), ("Agent-gpt-4o-mini", self._mini_cfg)])
        assert df is not None
        const_df, _ = format_agent_comparison(df)
        assert "parameter" in const_df.columns
        assert "value" in const_df.columns
        # temperature, max_actions, can_finish are shared
        assert set(const_df["parameter"]).issuperset({"llm_config.temperature", "max_actions", "can_finish"})

    def test_variables_are_differing_params(self) -> None:
        df = agent_configs_to_df([("Agent-gpt-4o", self._gpt4o_cfg), ("Agent-gpt-4o-mini", self._mini_cfg)])
        assert df is not None
        _, var_df = format_agent_comparison(df)
        assert "parameter" in var_df.columns
        # model_name differs between the two agents
        assert "llm_config.model_name" in var_df["parameter"].values

    def test_variables_pivoted_one_col_per_agent(self) -> None:
        df = agent_configs_to_df([("Agent-gpt-4o", self._gpt4o_cfg), ("Agent-gpt-4o-mini", self._mini_cfg)])
        assert df is not None
        _, var_df = format_agent_comparison(df)
        assert "Agent-gpt-4o" in var_df.columns
        assert "Agent-gpt-4o-mini" in var_df.columns
        row = var_df[var_df["parameter"] == "llm_config.model_name"].iloc[0]
        assert row["Agent-gpt-4o"] == "gpt-4o"
        assert row["Agent-gpt-4o-mini"] == "gpt-4o-mini"

    def test_single_agent_all_constants(self) -> None:
        df = agent_configs_to_df([("Agent-gpt-4o", self._gpt4o_cfg)])
        assert df is not None
        const_df, var_df = format_agent_comparison(df)
        # With one agent everything is a constant
        assert not const_df.empty
        assert var_df.empty or "parameter" in var_df.columns

    def test_empty_returns_empty_dfs(self) -> None:
        df = agent_configs_to_df([])
        assert df is None

    def test_variables_correct_when_df_rows_reversed(self) -> None:
        # Regression: iloc-based lookup assigned values to the wrong agent when
        # DataFrame row order differed from agent_names list order.
        df = agent_configs_to_df([("Agent-gpt-4o", self._gpt4o_cfg), ("Agent-gpt-4o-mini", self._mini_cfg)])
        assert df is not None
        # Reverse the row order to simulate non-sequential indexing.
        df = df.iloc[::-1].reset_index(drop=True)
        _, var_df = format_agent_comparison(df)
        row = var_df[var_df["parameter"] == "llm_config.model_name"].iloc[0]
        assert row["Agent-gpt-4o"] == "gpt-4o"
        assert row["Agent-gpt-4o-mini"] == "gpt-4o-mini"

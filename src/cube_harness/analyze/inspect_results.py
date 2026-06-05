import re
from typing import Any

import numpy as np
import pandas as pd
from cube.core import EnvironmentOutput

from cube_harness.core import Trajectory

TASK_KEY = "task_name"


def _extract_error_from_trajectory(traj: Trajectory) -> tuple[str | None, str | None]:
    for step in reversed(traj.steps):
        if hasattr(step.output, "error") and step.output.error is not None:
            return step.output.error.exception_str, step.output.error.stack_trace
    return None, None


def trajectories_to_df(trajectories: list[Trajectory]) -> pd.DataFrame | None:
    from cube_harness.analyze.xray_utils import compute_trajectory_stats, trajectory_status

    records: list[dict[str, Any]] = []
    for traj in trajectories:
        record: dict[str, Any] = {"trajectory_id": traj.id}
        record.update(traj.metadata)

        if traj.steps or traj.summary_stats:
            stats = compute_trajectory_stats(traj)
            record["cum_reward"] = stats["final_reward"]
            record["n_steps"] = stats["n_env_steps"]
            record["duration"] = stats["duration"]
            record["cost"] = stats["cost"]
            record["prompt_tokens"] = stats["prompt_tokens"]
            record["completion_tokens"] = stats["completion_tokens"]

            if traj.steps:
                last_env = None
                for step in reversed(traj.steps):
                    if isinstance(step.output, EnvironmentOutput):
                        last_env = step.output
                        break
                record["done"] = last_env.done if last_env else False
            else:
                reward = stats.get("final_reward", 0.0)
                record["done"] = traj.reward_info.get("done", reward > 0)
        else:
            record["cum_reward"] = np.nan
            record["n_steps"] = np.nan
            record["duration"] = None
            record["cost"] = np.nan
            record["prompt_tokens"] = 0
            record["completion_tokens"] = 0
            record["done"] = False

        record["status"] = trajectory_status(traj)
        err_msg, stack_trace = _extract_error_from_trajectory(traj)
        record["err_msg"] = err_msg
        record["stack_trace"] = stack_trace
        records.append(record)

    if not records:
        return None
    return pd.DataFrame(records)


def get_constants_and_variables(df: pd.DataFrame, drop_constants: bool = False) -> tuple[dict, list[str], pd.DataFrame]:
    constants: dict[str, Any] = {}
    variable_keys: list[str] = []
    for col in df.columns:
        try:
            nuniq = df[col].nunique(dropna=False)
        except TypeError:
            nuniq = 0
        if nuniq == 1:
            val = df[col].iloc[0]
            if isinstance(val, np.generic):
                val = val.item()
            constants[col] = val
            if drop_constants:
                df = df.drop(col, axis=1)
        else:
            variable_keys.append(col)
    return constants, variable_keys, df


def map_err_key(err_msg: str | None) -> str | None:
    if err_msg is None:
        return None
    err_msg = err_msg[: err_msg.find("=== logs ===")] if "=== logs ===" in err_msg else err_msg
    err_msg = err_msg.rstrip()
    regex_replacements = [
        (r"your messages resulted in \d+ tokens", "your messages resulted in x tokens"),
        (r"(?<=Exception uncaught by agent or environment in task\s)([^\s]+)", "<task_name>."),
    ]
    for pattern, replacement in regex_replacements:
        err_msg = re.sub(pattern, replacement, err_msg)
    return err_msg


def error_report(df: pd.DataFrame, max_stack_trace: int = 10) -> str:
    if "err_key" not in df.columns:
        df = df.copy()
        df["err_key"] = None
        err_mask = df["err_msg"].notnull()
        df.loc[err_mask, "err_key"] = df.loc[err_mask, "err_msg"].map(map_err_key)

    err_df = df[df["err_key"].notnull()]
    if err_df.empty:
        return "No errors found."

    unique_counts = err_df["err_key"].value_counts().sort_values(ascending=False)
    report: list[str] = []

    for err_key, count in unique_counts.items():
        report.append("---")
        report.append(f"### {count}x: {err_key}")
        sub_df = err_df[err_df["err_key"] == err_key]

        task_col = TASK_KEY if TASK_KEY in sub_df.columns else None
        if task_col is None and TASK_KEY in (sub_df.index.names or []):
            sub_df = sub_df.reset_index()
            task_col = TASK_KEY if TASK_KEY in sub_df.columns else None

        shown = 0
        for _, row in sub_df.iterrows():
            if shown >= max_stack_trace:
                break
            task_name = (
                row.get(task_col, row.get("trajectory_id", "unknown"))
                if task_col
                else row.get("trajectory_id", "unknown")
            )
            report.append(f"\n**Task:** `{task_name}`")
            st = row.get("stack_trace")
            if st:
                report.append(f"```\n{st}\n```")
            shown += 1

    return "\n".join(report)


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Recursively flatten a nested dictionary using dot-separated keys."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def agent_configs_to_df(agents: list[tuple[str, dict]]) -> pd.DataFrame | None:
    """Build a DataFrame with one row per agent from (agent_name, config_dict) pairs.

    Config dicts are recursively flattened so nested keys become dot-separated columns,
    e.g. ``llm_config.model_name``.
    """
    if not agents:
        return None
    rows = [{"agent_name": name, **_flatten_dict(cfg)} for name, cfg in agents]
    return pd.DataFrame(rows)


def format_agent_comparison(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split an agent config DataFrame into shared parameters and varying parameters.

    ``df`` must have an ``agent_name`` column plus flattened config parameter columns,
    as produced by :func:`agent_configs_to_df`.

    Returns:
        const_df: parameters that are identical across all agents (columns: parameter, value).
        var_df: parameters that differ, pivoted so each agent is its own column
                (columns: parameter, <agent_name>, ...).
    """
    param_cols = [c for c in df.columns if c != "agent_name"]
    constants, variable_keys, _ = get_constants_and_variables(df[param_cols])

    const_records = [{"parameter": k, "value": str(v)} for k, v in constants.items()]
    const_df = pd.DataFrame(const_records) if const_records else pd.DataFrame(columns=["parameter", "value"])

    if not variable_keys or "agent_name" not in df.columns:
        return const_df, pd.DataFrame(columns=["parameter"])

    var_records = [
        {"parameter": var, **{row["agent_name"]: str(row[var]) for _, row in df.iterrows()}} for var in variable_keys
    ]
    var_df = pd.DataFrame(var_records) if var_records else pd.DataFrame(columns=["parameter"])
    return const_df, var_df

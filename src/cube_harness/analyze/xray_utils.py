"""Pure utility functions for the cube-harness XRay viewer.

All functions in this module are pure (or near-pure) — no Gradio imports, no global state.
This makes them independently testable without any UI framework.
"""

import datetime
import html as html_lib
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from cube.benchmark import BenchmarkConfig
from cube.core import EnvironmentOutput
from PIL import Image
from pydantic import BaseModel

from cube_harness.agent import AgentConfig
from cube_harness.analyze.stats import reward_mean_stderr
from cube_harness.core import AgentOutput, Trajectory, TrajectoryStep
from cube_harness.episode_status import STATUS_FILENAME, EpisodeStatus, should_sweep_running_to_stale
from cube_harness.episode_status import TERMINAL_STATUSES as _EPISODE_TERMINAL_STATUSES
from cube_harness.exp_runner import DEFAULT_CANCEL_GRACE_S, DEFAULT_STEP_TIMEOUT_S
from cube_harness.experiment_status import EXPERIMENT_STATUS_FILENAME, ExperimentStatus, is_driver_alive
from cube_harness.llm import LLMCall

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.

    Examples: 800ms, 4.2s, 3m 12s, 1h 5m
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def trajectory_status(traj: Trajectory) -> str:
    """Return a lifecycle status string for a trajectory.

    Reads ``_episode_status`` injected by :meth:`FileStorage._maybe_inject_episode_status`
    when a ``status.json`` file exists alongside the trajectory (post-PR#315 experiments).
    Falls back to :func:`_infer_status_legacy` for older experiments that pre-date the
    episode-status RFC.

    Values (canonical set — driven by status.json):
      'queued'         — QUEUED: claimed but not yet started
      'running'        — RUNNING: worker actively executing
      'success'        — COMPLETED with reward > 0
      'fail'           — COMPLETED with reward = 0
      'max_steps'      — MAX_STEPS_REACHED: step budget exhausted
      'failed'         — FAILED: worker crashed / abnormal termination
      'stale'          — STALE: heartbeat timeout, dead worker
      'cancelled'      — CANCELLED: deliberately stopped

    Legacy values (heuristic fallback — no status.json):
      'system_error'   — crashed before trajectory was written (legacy heuristic)
    """
    raw = traj.metadata.get("_episode_status")
    if raw is not None:
        return _map_episode_status(raw, traj)
    return _infer_status_legacy(traj)


_RAW_STATUS_MAP: dict[str, str] = {
    "QUEUED": "queued",
    "RUNNING": "running",
    "COMPLETED": "success",  # reward not available here; folds to ✓ either way
    "MAX_STEPS_REACHED": "max_steps",
    "FAILED": "failed",
    "STALE": "stale",
    "CANCELLED": "cancelled",
    "INVALID_CONFIG": "failed",  # permanent provider error — render as a failure
}


def _map_episode_status(raw: str, traj: Trajectory) -> str:
    """Map a raw Status string from status.json to an xray display status."""
    if raw == "COMPLETED":
        return "success" if (traj.reward_info and traj.reward_info.get("reward", 0) > 0) else "fail"
    mapped = _RAW_STATUS_MAP.get(raw)
    return mapped if mapped is not None else _infer_status_legacy(traj)


def _infer_status_legacy(traj: Trajectory) -> str:
    # DEPRECATED: remove once status.json is guaranteed present for all loaded experiments.
    # Used only when traj.metadata has no "_episode_status" key (pre-PR#315 experiments).
    if traj.metadata.get("_missing"):
        return "system_error" if traj.metadata.get("_failure_text") else "queued"
    if traj.end_time is None:
        return "system_error" if traj.metadata.get("_failure_text") else "running"
    if traj.reward_info and traj.reward_info.get("reward", 0) > 0:
        return "success"
    return "fail"


# Statuses that count as terminal outcomes for reward/step statistics.
TERMINAL_OUTCOME_STATUSES: frozenset[str] = frozenset({"success", "fail", "max_steps"})

# Statuses that represent in-flight work (not yet terminal).
IN_FLIGHT_STATUSES: frozenset[str] = frozenset({"queued", "running"})

# All statuses that are terminal (episode will not run again).
TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"success", "fail", "max_steps", "failed", "stale", "cancelled", "system_error"}
)

_STATUS_HTML: dict[str, str] = {
    # Canonical statuses (from status.json)
    "queued": "<span title='Queued — not yet started'>🕐</span>",
    "running": "<span title='Running'>▶️</span>",
    "success": "<span title='Completed — reward > 0'>🟢</span>",
    "fail": "<span title='Completed — no reward'>⚫</span>",
    "max_steps": "<span title='Max steps reached — step budget exhausted'>🎬</span>",
    "failed": "<span title='Failed — worker crashed'>⛔</span>",
    "stale": "<span title='Stale — heartbeat lost, dead worker'>👻</span>",
    "cancelled": "<span title='Cancelled'>🚫</span>",
    # Legacy heuristic (no status.json — pre-PR#315 experiments)
    "system_error": "<span title='System error — crashed (legacy inferred status)' style='color:#dc3545;font-weight:bold;font-size:14px'>✕</span>",
}

# Plain-text labels for the header bar and other non-HTML contexts.
_STATUS_LABEL: dict[str, str] = {
    "queued": "🕐 Queued",
    "running": "▶️ Running",
    "success": "🟢 Success",
    "fail": "⚫ Completed (no reward)",
    "max_steps": "🎬 Max steps reached",
    "failed": "⛔ Failed",
    "stale": "👻 Stale",
    "cancelled": "🚫 Cancelled",
    "system_error": "✕ System error (legacy)",
}

# Bare symbols for inline use (agent table status cell).
# Terminal-outcome statuses collapsed to ✔ in the agent-level aggregate view.
# success + fail + max_steps are all "ran to completion"; avg_reward captures the breakdown.
_COMPLETED_AGGREGATE_HTML = "<span title='Terminal — success, fail, or max steps'>✅</span>"


def _build_status_cell(statuses: list[str]) -> str:
    """Build the agent-table status cell: ``(15✓ + 4▶️) / 19`` or ``15✓ / 15``.

    All terminal-outcome statuses (success, fail, max_steps) collapse to ✓ so the
    agent row stays readable. Per-status detail lives in the Trajectories tab.
    Total always equals len(statuses). Parentheses only added when there are multiple parts.
    """
    n_terminal = sum(1 for s in statuses if s in TERMINAL_OUTCOME_STATUSES)
    counts: dict[str, int] = {}
    for s in statuses:
        if s not in TERMINAL_OUTCOME_STATUSES:
            counts[s] = counts.get(s, 0) + 1

    order = ["running", "queued", "stale", "cancelled", "failed", "system_error"]
    parts = []
    if n_terminal:
        parts.append(f"{n_terminal}{_COMPLETED_AGGREGATE_HTML}")
    for key in order:
        n = counts.get(key, 0)
        if n:
            parts.append(f"{n}{_STATUS_HTML.get(key, key)}")

    total = len(statuses)
    inner = " + ".join(parts)
    if len(parts) > 1:
        inner = f"({inner})"
    return f"{inner} / {total}"


def build_progress_html(
    n_completed: int,
    n_total: int,
    n_running: int,
    per_agent: list[tuple[str, int, int, int]] | None = None,
    exp_names: list[str] | None = None,
) -> str:
    """Return an HTML progress bar + label for experiment completion status.

    Args:
        n_completed: Total completed trajectories across all agents.
        n_total: Total trajectories across all agents.
        n_running: Total currently running trajectories.
        per_agent: Optional list of (agent_name, n_completed, n_total, n_running).
                   When provided with > 1 entry, a per-agent breakdown is appended.
        exp_names: Names of selected experiment directories being monitored.
    """
    header = ""
    if exp_names:
        label_text = "Monitoring"
        names_html = "".join(
            f'<code style="font-size:11px;background:#e5e7eb;border-radius:3px;padding:1px 5px;">'
            f"{html_lib.escape(n)}</code> "
            for n in exp_names
        )
        header = f'<div style="margin-bottom:6px;color:#666;font-size:11px;">{label_text}: {names_html}</div>'

    pct = (n_completed / n_total * 100) if n_total > 0 else 0
    bar = (
        f'<div style="background:#e5e7eb;border-radius:6px;height:14px;overflow:hidden;margin-bottom:4px;">'
        f'<div style="background:linear-gradient(90deg,#22c55e,#16a34a);height:100%;width:{pct:.1f}%;'
        f'transition:width 0.5s;"></div></div>'
    )
    label = f'<div style="font-size:12px;color:#555;">{n_completed}/{n_total} episodes completed'
    if n_running > 0:
        label += f", {n_running} running ⏳"
    label += "</div>"

    if not per_agent or len(per_agent) <= 1:
        return header + bar + label

    rows_html = ""
    for agent_name, agent_done, agent_total, agent_running in per_agent:
        agent_pct = (agent_done / agent_total * 100) if agent_total > 0 else 0
        mini_bar = (
            f'<div style="background:#e5e7eb;border-radius:4px;height:8px;overflow:hidden;flex:1;">'
            f'<div style="background:#22c55e;height:100%;width:{agent_pct:.1f}%;transition:width 0.5s;"></div></div>'
        )
        running_str = f" ⏳{agent_running}" if agent_running > 0 else ""
        rows_html += (
            f'<div style="display:flex;align-items:center;gap:8px;margin-top:4px;font-size:11px;color:#555;">'
            f'<div style="min-width:140px;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'
            f'font-family:monospace;" title="{html_lib.escape(agent_name)}">{html_lib.escape(agent_name)}</div>'
            f"{mini_bar}"
            f'<div style="min-width:60px;text-align:right;white-space:nowrap;">{agent_done}/{agent_total}{running_str}</div>'
            f"</div>"
        )
    return header + bar + label + rows_html


def archive_experiment(results_dir: Path, exp_name: str) -> None:
    """Move an experiment directory into results_dir/_archive/.

    Creates _archive/ if it does not exist. No-ops silently if the source does not exist.
    """
    src = results_dir / exp_name
    if not src.exists():
        return
    archive_dir = results_dir / "_archive"
    archive_dir.mkdir(exist_ok=True)
    shutil.move(str(src), str(archive_dir / exp_name))


def _is_experiment_dir(dir_path: Path) -> bool:
    """Return True if dir_path is a valid (non-archived) experiment directory."""
    if not dir_path.is_dir() or dir_path.name.startswith("_"):
        return False
    if (dir_path / "episodes").exists():
        return True
    if (dir_path / "trajectories").exists():
        return True
    return any(
        f.name.endswith(".metadata.json") and ".archived_" not in f.name for f in dir_path.glob("*.metadata.json")
    )


def _count_episodes(dir_path: Path) -> int:
    episodes_dir = dir_path / "episodes"
    if episodes_dir.exists():
        return sum(1 for d in episodes_dir.iterdir() if d.is_dir() and ".archived_" not in d.name)
    n = len([f for f in dir_path.glob("*.metadata.json") if ".archived_" not in f.name])
    if n > 0:
        return n
    traj_dir = dir_path / "trajectories"
    if traj_dir.exists():
        return len([f for f in traj_dir.glob("*.metadata.json") if ".archived_" not in f.name])
    return 0


def get_directory_contents(results_dir: Path) -> list[str]:
    """Return sorted list of experiment directory names with trajectory counts.

    Returns ["Select experiment directory"] + names sorted most-recent first.
    Includes directories that have trajectory metadata in the same dir (flat layout)
    or under a ``trajectories/`` subdirectory (legacy).
    Directories whose names start with '_' (e.g. _archive) are excluded.
    """
    sentinel = "Select experiment directory"
    if not results_dir or not results_dir.exists():
        return [sentinel]

    exp_descriptions = []
    for dir_path in results_dir.iterdir():
        if not _is_experiment_dir(dir_path):
            continue
        n_trajs = _count_episodes(dir_path)
        exp_descriptions.append(f"{dir_path.name} ({n_trajs} trajectories)")

    return [sentinel] + sorted(exp_descriptions, reverse=True)


def _parse_exp_date(dir_path: Path) -> str:
    """Extract a datetime string from the directory name, fall back to mtime.

    Returns "YYYY-MM-DD HH:MM:SS" when a full timestamp is found.

    Recognises common timestamp patterns in directory names:
      - YYYY-MM-DD[_HH-MM] or YYYY-MM-DDTHH:MM  (ISO-like)
      - YYYYMMDD_HHMMSS or YYYYMMDD_HHMM or YYYYMMDD  (compact, e.g. exp_20260221_074349)
    Falls back to the directory's mtime formatted as YYYY-MM-DD HH:MM:SS.
    """
    name = dir_path.name
    # ISO-like: YYYY-MM-DD optionally followed by _HH-MM or THH:MM
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})(?:[_T](\d{2})[-:](\d{2}))?", name)
    if m:
        date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        if m.group(4) and m.group(5):
            date_str += f" {m.group(4)}:{m.group(5)}"
        return date_str
    # Compact: YYYYMMDD optionally followed by _HHMMSS or _HHMM
    m = re.search(r"(\d{4})(\d{2})(\d{2})(?:_(\d{2})(\d{2})(\d{2})?)?(?!\d)", name)
    if m:
        date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        if m.group(4) and m.group(5):
            date_str += f" {m.group(4)}:{m.group(5)}"
            if m.group(6):
                date_str += f":{m.group(6)}"
        return date_str
    dt = datetime.datetime.fromtimestamp(dir_path.stat().st_mtime)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def agent_name_from_config(agent_cfg: dict) -> str:
    """Resolve `AgentConfig.agent_name` from a serialized agent_config dict.

    `agent_name` is a `@property` on AgentConfig (not a Pydantic field), so it is
    NOT present in the JSON dump. We re-instantiate the concrete subclass via
    TypedBaseModel's `_type` dispatch and read the property. When the module is
    unavailable locally (e.g. an experiment was produced on a branch that defines
    a custom agent) we synthesize a name that follows the common
    `<AgentName>-<model_name>` convention used by built-in agents — both so the
    column stays informative and so multi-experiment loads remain
    distinguishable when they only differ in model.
    """
    if not agent_cfg:
        return ""
    try:
        return AgentConfig.model_validate(dict(agent_cfg)).agent_name
    except Exception as exc:
        logger.debug("Could not instantiate AgentConfig (%s); synthesizing fallback name", exc)
        cls_name = (agent_cfg.get("_type") or "").rsplit(".", 1)[-1]
        # Drop the trailing "Config" so e.g. GennyConfig -> Genny, matching the
        # display convention of agent classes whose .agent_name we couldn't reach.
        if cls_name.endswith("Config"):
            cls_name = cls_name[: -len("Config")]
        model = (agent_cfg.get("llm_config") or {}).get("model_name") or ""
        if cls_name and model:
            return f"{cls_name}-{model}".replace("/", "_")
        return cls_name or model


def benchmark_name_from_config(bench_cfg: dict) -> str:
    """Resolve `BenchmarkConfig.benchmark_metadata.name` from a benchmark_config dict.

    `benchmark_metadata` is a `ClassVar` on BenchmarkConfig subclasses (declared at
    class definition time), so it is NOT serialized. We re-instantiate via `_type`
    dispatch and read the class-level metadata. Falls back to the class short name
    when the type can't be imported.
    """
    if not bench_cfg:
        return ""
    try:
        return BenchmarkConfig.model_validate(dict(bench_cfg)).benchmark_metadata.name
    except Exception as exc:
        logger.debug("Could not instantiate BenchmarkConfig (%s); falling back to _type", exc)
        return (bench_cfg.get("_type") or "").rsplit(".", 1)[-1]


def parse_exp_config(exp_cfg: dict) -> dict[str, str]:
    """Return {agent, model, benchmark} display strings from a parsed experiment_config dict.

    Single source of truth for extracting human-readable agent/model/benchmark
    identifiers from a serialized Experiment config. Used by both the experiments
    selector table (xray_utils) and the XRay viewer state (xray.XRayState) so the
    agent identifier shown in every UI surface comes from the same logic.

    Accepts both the current (`benchmark_config`) and the pre-split (`benchmark`)
    key names so historical experiment dirs still resolve a benchmark name.
    """
    agent_cfg = exp_cfg.get("agent_config") or {}
    bench_cfg = exp_cfg.get("benchmark_config") or exp_cfg.get("benchmark") or {}
    return {
        "agent": agent_name_from_config(agent_cfg),
        "model": (agent_cfg.get("llm_config") or {}).get("model_name") or "",
        "benchmark": benchmark_name_from_config(bench_cfg),
    }


def _parse_experiment_config(exp_dir: Path) -> dict[str, str]:
    """Return {agent, model, benchmark} from experiment_config.json.

    Falls back to scanning the first episode.metadata.json for agent_name when
    experiment_config.json is absent (e.g. synthetic test data).
    """
    config_path = exp_dir / "experiment_config.json"
    info = {"agent": "", "model": "", "benchmark": ""}

    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            info = parse_exp_config(cfg)
        except Exception as exc:
            logger.debug("Failed to parse experiment_config.json at %s: %s", config_path, exc)

    if not info["agent"]:
        # Fallback: read agent_name from first episode metadata (legacy / synthetic data)
        episodes_dir = exp_dir / "episodes"
        if episodes_dir.exists():
            for ep_dir in episodes_dir.iterdir():
                meta = ep_dir / "episode.metadata.json"
                if meta.exists():
                    try:
                        with open(meta) as f:
                            data = json.load(f)
                        agent = data.get("metadata", {}).get("agent_name", "")
                        if agent:
                            info["agent"] = agent
                            break
                    except Exception as exc:
                        logger.debug("Failed to parse %s: %s", meta, exc)

    return info


GHOST_TIMEOUT = DEFAULT_STEP_TIMEOUT_S + DEFAULT_CANCEL_GRACE_S  # mirrors runner's kill threshold
_XRAY_CACHE_FILENAME = ".xray_summary.json"


def _promote_ghost_episodes(exp_dir: Path) -> None:
    """Write STALE into status.json for in-flight episodes whose driver is dead.

    RUNNING + ray mode: promoted by per-episode heartbeat age (GHOST_TIMEOUT).
    RUNNING + sequential mode + driver dead: promoted immediately — driver is the
      worker, so process death kills both.
    QUEUED + driver dead: promoted regardless of mode — no worker will ever pick
      these up if the scheduler that queued them is gone.
    """
    episodes_dir = exp_dir / "episodes"
    if not episodes_dir.exists():
        return
    exp_status = ExperimentStatus.read(exp_dir / EXPERIMENT_STATUS_FILENAME)
    driver_dead = not is_driver_alive(exp_status, exp_dir, timeout_s=GHOST_TIMEOUT)
    now = time.time()
    for ep_dir in episodes_dir.iterdir():
        if not ep_dir.is_dir() or ".archived_" in ep_dir.name:
            continue
        status = EpisodeStatus.read(ep_dir / STATUS_FILENAME)
        if status is None:
            continue
        is_stale = False
        if status.status == "RUNNING":
            if driver_dead and exp_status is not None and exp_status.mode == "sequential":
                is_stale = True  # driver == worker; both dead
            else:
                is_stale = should_sweep_running_to_stale(
                    status,
                    now=now,
                    step_timeout_s=DEFAULT_STEP_TIMEOUT_S,
                    cancel_grace_s=DEFAULT_CANCEL_GRACE_S,
                )
        elif status.status == "QUEUED" and driver_dead:
            is_stale = True
        if not is_stale:
            continue
        status.status = "STALE"
        if status.ended_at is None:
            status.ended_at = status.last_heartbeat_at or status.started_at
        try:
            status.write(ep_dir / STATUS_FILENAME)
        except OSError:
            pass  # best-effort: race with runner archiving the dir is harmless


def _all_episodes_terminal(exp_dir: Path) -> bool:
    """Return True when every non-archived episode has a terminal status.

    Episodes with no status.json but an episode.metadata.json are treated as
    terminal (pre-PR#315 legacy format — always completed by definition).
    V1 experiments (no episodes/ dir) return True unconditionally.
    """
    episodes_dir = exp_dir / "episodes"
    if not episodes_dir.exists():
        return True  # V1 flat layout — always historical/done
    for ep_dir in episodes_dir.iterdir():
        if not ep_dir.is_dir() or ".archived_" in ep_dir.name:
            continue
        status = EpisodeStatus.read(ep_dir / STATUS_FILENAME)
        if status is None:
            # No status.json: terminal only if episode.metadata.json exists (legacy done)
            if not (ep_dir / "episode.metadata.json").exists():
                return False
        elif status.status not in _EPISODE_TERMINAL_STATUSES:
            return False
    return True


def _is_cache_valid(exp_dir: Path, cache_mtime: float) -> bool:
    """Return False if episodes/ or any episode dir was modified after the cache was written.

    Uses only stat() calls (no file reads). Catches:
    - New episode dirs created (episodes/ dir mtime changes).
    - Episode relaunched: runner archives old dir and creates new one → episodes/ mtime.
    - Status.json written: EpisodeStatus.write() creates a .tmp sibling first, which
      updates the episode dir mtime via the tmp-file creation step.
    """
    episodes_dir = exp_dir / "episodes"
    if not episodes_dir.exists():
        return True
    if episodes_dir.stat().st_mtime > cache_mtime:
        return False
    for ep_dir in episodes_dir.iterdir():
        if ep_dir.is_dir() and ".archived_" not in ep_dir.name:
            if ep_dir.stat().st_mtime > cache_mtime:
                return False
    return True


def _compute_exp_row(exp_dir: Path) -> dict[str, Any]:
    """Compute display fields for one experiment by reading per-episode status.json files.

    V2 (episodes/ dir): reads status.json per episode — O(N_episodes) JSON reads.
    V1 (flat *.metadata.json): reads trajectory metadata per episode.
    Returns: {date, agent, model, benchmark, status, avg_reward}.
    """
    cfg_info = _parse_experiment_config(exp_dir)
    statuses: list[str] = []
    rewards: list[float] = []

    episodes_dir = exp_dir / "episodes"
    if episodes_dir.exists():
        for ep_dir in episodes_dir.iterdir():
            if not ep_dir.is_dir() or ".archived_" in ep_dir.name:
                continue
            status = EpisodeStatus.read(ep_dir / STATUS_FILENAME)
            if status is not None:
                statuses.append(_RAW_STATUS_MAP.get(status.status, "system_error"))
                if status.status in ("COMPLETED", "MAX_STEPS_REACHED") and status.reward is not None:
                    rewards.append(float(status.reward))
            elif (ep_dir / "episode.metadata.json").exists():
                statuses.append("success")  # pre-status.json legacy episode
            else:
                statuses.append("queued")
    else:
        # V1: read flat *.metadata.json for status and reward
        for search_dir in (exp_dir, exp_dir / "trajectories"):
            if not search_dir.exists():
                continue
            for meta_file in search_dir.glob("*.metadata.json"):
                if ".archived_" in meta_file.name:
                    continue
                try:
                    with open(meta_file) as f:
                        data = json.load(f)
                    reward_info = data.get("reward_info") or {}
                    reward = reward_info.get("reward")
                    if data.get("end_time") is not None:
                        statuses.append("success" if (reward or 0) > 0 else "fail")
                        if reward is not None:
                            rewards.append(float(reward))
                    else:
                        statuses.append("queued")
                except Exception as exc:
                    logger.debug("Failed to parse %s: %s", meta_file, exc)
                    statuses.append("system_error")

    status_html = _build_status_cell(statuses) if statuses else "—"
    mean, stderr = _reward_mean_stderr(rewards)
    avg_reward_str = f"{mean:.3f} ± {stderr:.3f}" if rewards else "—"

    return {
        "date": _parse_exp_date(exp_dir),
        "agent": cfg_info["agent"],
        "model": cfg_info["model"],
        "benchmark": cfg_info["benchmark"],
        "status": status_html,
        "avg_reward": avg_reward_str,
    }


def get_experiments_table_rows(results_dir: Path) -> list[dict[str, Any]]:
    """Return one row per experiment directory for the Experiments selector table.

    Columns: selected, experiment, date, agent, model, benchmark, status, avg_reward.
    Uses a per-experiment .xray_summary.json cache. Cache is written once all episodes
    are terminal (including after ghost promotion) and invalidated via episode dir mtime.
    Sorted most-recent first.
    """
    if not results_dir or not results_dir.exists():
        return []

    rows = []
    for dir_path in results_dir.iterdir():
        if not _is_experiment_dir(dir_path):
            continue
        cache_path = dir_path / _XRAY_CACHE_FILENAME
        if cache_path.exists():
            try:
                cache_mtime = cache_path.stat().st_mtime
                if _is_cache_valid(dir_path, cache_mtime):
                    with open(cache_path) as f:
                        cached = json.load(f)
                    rows.append({"selected": False, "experiment": dir_path.name, **cached})
                    continue
            except Exception as exc:
                logger.debug("Cache read failed for %s: %s", cache_path, exc)
        _promote_ghost_episodes(dir_path)
        summary = _compute_exp_row(dir_path)
        if _all_episodes_terminal(dir_path):
            try:
                tmp = cache_path.parent / (cache_path.name + ".tmp")
                tmp.write_text(json.dumps(summary, indent=2))
                os.replace(tmp, cache_path)
            except Exception as exc:
                logger.debug("Cache write failed for %s: %s", cache_path, exc)
        rows.append({"selected": False, "experiment": dir_path.name, **summary})
    rows.sort(key=lambda r: r["date"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Screenshot extraction
# ---------------------------------------------------------------------------


def get_screenshot_from_step(step: EnvironmentOutput | AgentOutput | None) -> Image.Image | None:
    """Extract the first PIL Image from an EnvironmentOutput's observation contents.

    Returns None if no image is found, step is None, or step is an AgentOutput.
    """
    if not isinstance(step, EnvironmentOutput):
        return None
    for content in step.obs.contents:
        if isinstance(content.data, Image.Image):
            return content.data
    return None


def get_current_screenshot(
    step: EnvironmentOutput | AgentOutput | None,
    prev_step: EnvironmentOutput | AgentOutput | None,
) -> Image.Image | None:
    """Get the best screenshot for the current step.

    If current step is EnvironmentOutput: return its screenshot.
    If current step is AgentOutput: fall back to screenshot from prev_step.
    """
    img = get_screenshot_from_step(step)
    if img is None and prev_step is not None:
        img = get_screenshot_from_step(prev_step)
    return img


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------


def extract_obs_content(step: EnvironmentOutput | None, name_pattern: str) -> str | None:
    """Find text content in EnvironmentOutput.obs.contents by name substring match.

    Performs case-insensitive substring match against content.name.
    Returns the first matching str content.data, or None if not found.

    Examples:
        extract_obs_content(step, "axtree")  -> accessibility tree text
        extract_obs_content(step, "html")    -> page HTML
        extract_obs_content(step, "pruned")  -> pruned HTML
    """
    if not isinstance(step, EnvironmentOutput):
        return None
    pattern_lower = name_pattern.lower()
    for content in step.obs.contents:
        if isinstance(content.data, str) and pattern_lower in (content.name or "").lower():
            return content.data
    return None


# ---------------------------------------------------------------------------
# LLM prompt / chat rendering
# ---------------------------------------------------------------------------


_COLLAPSE_THRESHOLD = 2000  # chars (~20 lines) — messages longer than this start collapsed


def _msg_to_dict(msg: object) -> dict:
    """Normalise a message to a plain dict."""
    if isinstance(msg, dict):
        return msg
    if hasattr(msg, "model_dump"):
        return msg.model_dump()
    if hasattr(msg, "__dict__"):
        return dict(msg.__dict__)
    return {"role": "unknown", "content": str(msg)}


def _preview(text: str, max_chars: int = 80) -> str:
    """Return first non-empty line of text, truncated to max_chars."""
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:max_chars] + ("…" if len(line) > max_chars else "")
    return ""


def _details_block(label: str, body: str, icon: str = "📄") -> str:
    """Wrap body in a <details> block. Short content is open by default.

    Summary shows: icon + label + first-line preview (when collapsed).
    """
    open_attr = " open" if len(body) <= _COLLAPSE_THRESHOLD else ""
    preview = _preview(body)
    preview_html = (
        f" <span style='color:#888;font-weight:normal'>{html_lib.escape(preview)}</span>"
        if preview and not open_attr
        else ""
    )
    escaped = html_lib.escape(body)
    return (
        f"<details{open_attr}>"
        f"<summary>{icon} <strong>{html_lib.escape(label)}</strong>{preview_html}</summary>"
        f"<pre style='white-space:pre-wrap;overflow-wrap:anywhere;margin:4px 0'>{escaped}</pre>"
        f"</details>\n"
    )


def _render_text_content(text: str) -> str:
    """Render a plain text content string.

    Handles the '##name\\nbody' convention used by Content.to_message() for named
    text/dict content, but also works for any plain string.
    """
    if text.startswith("##"):
        newline = text.find("\n")
        if newline != -1:
            name = text[2:newline].strip()
            body = text[newline + 1 :]
            return _details_block(name, body)
    return _details_block("text", text)


def _render_content_items(content: str | list | None) -> str:
    """Render a message's content field as HTML.

    Handles the common content types found in LLM message dicts:
      - str:            plain text or '##name\\nbody' encoded text
      - list of items:  multimodal content list with typed items:
          {"type": "text",      "text": ...}
          {"type": "image_url", "image_url": {"url": ...}}
          {"type": "image",     "url": ...}          # alternate image format
          {"type": "audio",     ...}                 # future / other modalities
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return _render_text_content(content)

    # Multimodal list — iterate items, grouping a text label with a following image
    parts: list[str] = []
    items = [i for i in content if isinstance(i, dict)]
    idx = 0
    while idx < len(items):
        item = items[idx]
        item_type = item.get("type", "")
        next_item = items[idx + 1] if idx + 1 < len(items) else None

        if item_type == "text":
            text = item.get("text", "")
            # If the next item is an image, this text is a label for it
            if next_item is not None and next_item.get("type") in ("image_url", "image"):
                url = next_item.get("image_url", {}).get("url", "") or next_item.get("url", "")
                img = f"<img src='{url}' style='max-width:100%;border-radius:4px;margin:4px 0'>"
                parts.append(
                    f"<details open><summary>📷 <strong>{html_lib.escape(text or 'screenshot')}</strong></summary>{img}</details>\n"
                )
                idx += 2
            else:
                parts.append(_render_text_content(text))
                idx += 1
        elif item_type in ("image_url", "image"):
            url = item.get("image_url", {}).get("url", "") or item.get("url", "")
            img = f"<img src='{url}' style='max-width:100%;border-radius:4px;margin:4px 0'>"
            parts.append(f"<details open><summary>📷 <strong>screenshot</strong></summary>{img}</details>\n")
            idx += 1
        else:
            # Unknown / future type — show type name as a placeholder
            parts.append(f"<em>[{html_lib.escape(item_type)}]</em>\n")
            idx += 1

    return "".join(parts)


def _render_assistant_content(msg: dict) -> str:
    """Render assistant message: text content + tool calls as HTML."""
    parts: list[str] = []
    content = msg.get("content") or ""
    if content:
        parts.append(_details_block("reasoning", str(content), icon="💭"))
    tool_calls = msg.get("tool_calls") or []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            tc = tc.model_dump() if hasattr(tc, "model_dump") else vars(tc)
        fn = tc.get("function", {})
        name = fn.get("name", "?")
        args = fn.get("arguments", "")
        if isinstance(args, str):
            try:
                args = json.dumps(json.loads(args), indent=2)
            except (json.JSONDecodeError, ValueError):
                pass
        parts.append(_details_block(f"tool call: {name}", str(args), icon="🔧"))
    return "".join(parts)


_ROLE_STYLE = {
    "system": "background:#f0f4ff;border-left:3px solid #6c8ebf",
    "user": "background:#f5f5f5;border-left:3px solid #aaa",
    "tool": "background:#fff8e7;border-left:3px solid #e6a817",
    "assistant": "background:#f0fff4;border-left:3px solid #5cb85c",
}


def _render_llm_call_html(llm_call: LLMCall) -> str:
    """Render a single LLM call (prompt + response) as HTML message blocks."""
    config_json = html_lib.escape(llm_call.llm_config.model_dump_json(indent=2))
    config_html = (
        f"<details><summary>⚙️ <strong>llm_config</strong></summary>"
        f"<pre style='white-space:pre-wrap;overflow-wrap:anywhere;margin:4px 0'>{config_json}</pre>"
        f"</details>\n"
    )
    if llm_call.prompt.tools:
        tools_json = html_lib.escape(json.dumps(llm_call.prompt.tools, indent=2))
        n = len(llm_call.prompt.tools)
        tools_html = (
            f"<details><summary>🔧 <strong>tools</strong> ({n})</summary>"
            f"<pre style='white-space:pre-wrap;overflow-wrap:anywhere;margin:4px 0'>{tools_json}</pre>"
            f"</details>\n"
        )
    else:
        tools_html = ""
    messages = list(llm_call.prompt.messages) + [llm_call.output]
    blocks: list[str] = [config_html, tools_html]

    for i, msg in enumerate(messages):
        msg_dict = _msg_to_dict(msg)
        role = msg_dict.get("role", "unknown")
        tool_call_id = msg_dict.get("tool_call_id")

        label = f"[{i + 1}] {role}"
        if tool_call_id:
            label += f" · tool_result for {tool_call_id}"

        if role == "assistant":
            body_html = _render_assistant_content(msg_dict)
        else:
            body_html = _render_content_items(msg_dict.get("content"))

        style = _ROLE_STYLE.get(role, "background:#fafafa;border-left:3px solid #ccc")
        blocks.append(
            f"<div style='margin:6px 0;padding:8px 12px;border-radius:4px;{style}'>"
            f"<strong>{html_lib.escape(label)}</strong><br>{body_html}</div>\n"
        )

    return "".join(blocks)


def get_chat_branches(step: EnvironmentOutput | AgentOutput | None) -> dict[str, str]:
    """Return {label: html} for each LLMCall in an agent step.

    The tab label is call.tag when set, otherwise call.id.
    Returns empty dict for non-AgentOutput steps or steps with no llm_calls.
    """
    if not isinstance(step, AgentOutput):
        return {}
    return {(call.tag or call.id): _render_llm_call_html(call) for call in step.llm_calls}


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len characters with a trailing indicator."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... [truncated]"


# ---------------------------------------------------------------------------
# Step detail markdown
# ---------------------------------------------------------------------------


def get_step_details_markdown(
    step: EnvironmentOutput | AgentOutput | None,
    traj_step: TrajectoryStep | None,
) -> str:
    """Produce a context-aware markdown summary of the current step."""
    if step is None:
        return "No step selected"

    duration_info = ""
    if traj_step and traj_step.start_time is not None and traj_step.end_time is not None:
        duration = traj_step.end_time - traj_step.start_time
        duration_info = f" │ ⏱️ {format_duration(duration)}"

    if isinstance(step, EnvironmentOutput):
        return _format_env_step_details(step, duration_info)
    elif isinstance(step, AgentOutput):
        return _format_agent_step_details(step, duration_info)
    return "Unknown step type"


def _format_env_step_details(step: EnvironmentOutput, duration_info: str) -> str:
    """Format EnvironmentOutput details as markdown."""
    sections = [f"## 🌍 Environment Output{duration_info}\n"]

    if step.done:
        status = "✅ **Success**" if step.reward > 0 else "❌ **Failed**"
        sections.append(f"**Status:** {status} │ **Reward:** {step.reward:.2f}\n")
    else:
        sections.append(f"**Reward:** {step.reward:.2f} │ **Done:** No\n")

    for content in step.obs.contents:
        if isinstance(content.data, str):
            name = content.name or "Content"
            data = _truncate(content.data, 200000)
            sections.append(f"### {name}\n```\n{data}\n```\n")
        elif isinstance(content.data, Image.Image):
            sections.append(f"**{content.name or 'Screenshot'}:** {content.data.size[0]}x{content.data.size[1]}\n")
        elif isinstance(content.data, (dict, list)):
            name = content.name or "Data"
            data_str = json.dumps(content.data, indent=2)
            data_str = _truncate(data_str, 100000)
            sections.append(f"### {name}\n```json\n{data_str}\n```\n")
        elif isinstance(content.data, BaseModel):
            name = content.name or "Data"
            data_str = content.data.model_dump_json(indent=2)
            data_str = _truncate(data_str, 100000)
            sections.append(f"### {name}\n```json\n{data_str}\n```\n")

    if step.info.get("error"):
        sections.append(f"\n### ⚠️ Error\n```\n{step.info['error']}\n```\n")

    return "\n".join(sections)


def _format_agent_step_details(step: AgentOutput, duration_info: str) -> str:
    """Format AgentOutput details as markdown."""
    sections = [f"## 🤖 Agent Output{duration_info}\n"]

    if step.llm_calls:
        llm_call = step.llm_calls[0]
        usage = llm_call.usage
        if usage and usage.prompt_tokens > 0:
            token_parts = [f"📊 **Tokens:** prompt: {usage.prompt_tokens:,}"]
            token_parts.append(f"completion: {usage.completion_tokens:,}")
            if usage.cached_tokens > 0:
                cache_pct = usage.cached_tokens / usage.prompt_tokens * 100
                token_parts.append(f"cached: {usage.cached_tokens:,} ({cache_pct:.0f}%)")
            if usage.cache_creation_tokens > 0:
                token_parts.append(f"cache_created: {usage.cache_creation_tokens:,}")
            if usage.cost > 0:
                token_parts.append(f"💰 **${usage.cost:.4f}**")
            sections.append(" │ ".join(token_parts) + "\n")

    if step.thoughts:
        sections.append(f"### Rationale\n{_truncate(step.thoughts, 150000)}\n")

    if step.actions:
        sections.append("### Actions\n")
        for i, action in enumerate(step.actions):
            args_str = json.dumps(action.arguments, indent=2)
            sections.append(f"**{i + 1}. {action.name}**\n```json\n{args_str}\n```\n")
    else:
        sections.append("*No actions taken*\n")

    if step.llm_calls:
        llm_call = step.llm_calls[0]
        if llm_call.output:
            msg = llm_call.output
            content = getattr(msg, "content", None)
            if content:
                reasoning = _truncate(str(content), 150000)
                sections.append(f"### Agent Reasoning\n{reasoning}\n")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Always-visible step summary (task goal + agent action)
# ---------------------------------------------------------------------------


def get_task_goal(trajectory: Trajectory | None) -> str:
    """Extract the task goal text from the first EnvironmentOutput's text content.

    The goal is typically the first text content item in the first env step,
    set by Task.setup() via Observation.from_text(goal).
    """
    if trajectory is None:
        return "*No trajectory loaded*"
    if not trajectory.steps:
        return "*No goal text found*"
    for ts in trajectory.steps:
        if isinstance(ts.output, EnvironmentOutput):
            for content in ts.output.obs.contents:
                if isinstance(content.data, str) and content.data.strip():
                    return content.data
    return "*No goal text found*"


def get_agent_action_markdown(agent_out: AgentOutput | None) -> str:
    """Return a compact markdown summary of the agent's actions as function call syntax.

    Renders each action as: `name(key="value", key2=123)`
    Long string values are truncated to 200 chars.
    Returns a placeholder for terminal steps.
    """
    if agent_out is None:
        return "*Terminal step — no agent action*"
    if not agent_out.actions:
        return "*No actions taken*"
    parts = []
    for action in agent_out.actions:
        args_parts = []
        for k, v in (action.arguments or {}).items():
            if isinstance(v, str):
                v_display = v if len(v) <= 200 else v[:200] + "…"
                args_parts.append(f'{k}="{v_display}"')
            else:
                args_parts.append(f"{k}={v!r}")
        call_str = f"{action.name}({', '.join(args_parts)})"
        parts.append(f"`{call_str}`")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Paired step rendering (env + agent shown together)
# ---------------------------------------------------------------------------


def get_paired_step_details_markdown(
    env_out: EnvironmentOutput | None,
    agent_out: AgentOutput | None,
    env_ts: TrajectoryStep | None,
    agent_ts: TrajectoryStep | None,
) -> str:
    """Produce a combined markdown summary showing both env observation and agent action."""
    if env_out is None:
        return "No step selected"

    env_duration = ""
    if env_ts and env_ts.start_time is not None and env_ts.end_time is not None:
        env_duration = f" │ ⏱️ {format_duration(env_ts.end_time - env_ts.start_time)}"

    env_section = _format_env_step_details(env_out, env_duration)

    if agent_out is None:
        agent_section = "\n---\n\n## 🤖 Agent Action\n\n*No agent action — terminal observation.*\n"
    else:
        agent_duration = ""
        if agent_ts and agent_ts.start_time is not None and agent_ts.end_time is not None:
            agent_duration = f" │ ⏱️ {format_duration(agent_ts.end_time - agent_ts.start_time)}"
        agent_section = "\n---\n\n" + _format_agent_step_details(agent_out, agent_duration)

    return env_section + agent_section


def get_paired_error_markdown(
    env_out: EnvironmentOutput | None,
    agent_out: AgentOutput | None,
) -> str:
    """Show errors from both the env output and the agent output for this UI step."""
    parts = []

    if env_out is not None:
        env_err = _extract_error_markdown(env_out, "Environment")
        if env_err:
            parts.append(env_err)

    if agent_out is not None:
        agent_err = _extract_error_markdown(agent_out, "Agent")
        if agent_err:
            parts.append(agent_err)

    return "\n\n---\n\n".join(parts) if parts else "No errors in this step"


def _extract_error_markdown(step: EnvironmentOutput | AgentOutput, label: str) -> str:
    """Extract error string from a single step, returning empty string if none."""
    if step.error is not None:
        err = step.error
        return (
            f"### ⚠️ {label}: {err.error_type}\n"
            f"**Message:** {err.exception_str}\n\n"
            f"**Stack Trace:**\n```\n{err.stack_trace}\n```"
        )
    if isinstance(step, EnvironmentOutput):
        info_error = step.info.get("error")
        if info_error:
            return f"### ⚠️ {label} error (from info)\n```\n{info_error}\n```"
    return ""


# ---------------------------------------------------------------------------
# Error & logs (legacy single-step, kept for backward compatibility)
# ---------------------------------------------------------------------------


def get_step_error_markdown(step: EnvironmentOutput | AgentOutput | None) -> str:
    """Extract error information from a single step as markdown.

    Checks step.error (StepError) for both step types.
    Also checks EnvironmentOutput.info.get('error') as a fallback.
    """
    if step is None:
        return "No errors in this step"
    result = _extract_error_markdown(step, "Error")
    return result if result else "No errors in this step"


def get_step_logs_markdown(
    step: EnvironmentOutput | AgentOutput | None,
    traj: Trajectory | None,
) -> str:
    """Extract log information from a step and trajectory metadata.

    Shows failure stack trace prominently when present (_failure_text in metadata),
    then EnvironmentOutput.info entries, then trajectory metadata.
    For missing stubs, returns early after showing failure info.
    """
    parts = []

    if traj:
        failure_text = traj.metadata.get("_failure_text", "")
        if failure_text:
            parts.append(f"### ❌ System Error\n```\n{failure_text}\n```\n")
        elif traj.metadata.get("_missing"):
            parts.append(
                "### ❌ Missing Trajectory\n\n"
                "This task has no trajectory data. It may have crashed before any steps were recorded.\n"
            )
        # For missing stubs there are no steps, so return now
        if traj.metadata.get("_missing"):
            return "\n".join(parts)

    if isinstance(step, EnvironmentOutput) and step.info:
        log_entries = {k: v for k, v in step.info.items() if k not in ("error", "message")}
        if log_entries:
            parts.append("### Step Info\n")
            for k, v in log_entries.items():
                parts.append(f"**{k}**: `{v}`\n")

    if traj and traj.metadata:
        # Exclude internal keys already displayed above
        meta = {k: v for k, v in traj.metadata.items() if k not in ("_failure_text", "_missing")}
        if meta:
            meta_str = json.dumps(meta, indent=2)
            parts.append(f"\n### Trajectory Metadata\n```json\n{meta_str}\n```\n")

    return "\n".join(parts) if parts else "No log information available."


def get_logs_tab_markdown(traj: Trajectory | None, log_content: str) -> str:
    """Render the Logs tab: episode log file content, with system error banner when present."""
    if traj is None:
        return "No trajectory selected."

    parts = []

    retry_count = traj.metadata.get("_retry_count", 0)
    error_type = traj.metadata.get("_error_type")
    error_message = traj.metadata.get("_error_message")
    if retry_count or error_type or error_message:
        detail_parts = []
        if retry_count:
            detail_parts.append(f"**Attempt:** {retry_count + 1} (retried {retry_count}×)")
        if error_type:
            detail_parts.append(f"**Error type:** `{error_type}`")
        if error_message:
            detail_parts.append(f"**Error message:** `{error_message}`")
        parts.append("### Episode Status\n" + "\n\n".join(detail_parts) + "\n")

    failure_text = traj.metadata.get("_failure_text", "")
    if failure_text:
        parts.append(f"### ❌ System Error\n```\n{failure_text}\n```\n")
    elif traj.metadata.get("_missing"):
        parts.append(
            "### ❌ Missing Trajectory\n\n"
            "This task has no trajectory data. It may have crashed before any steps were recorded.\n"
        )

    if log_content:
        parts.append(f"```\n{log_content}\n```")
    else:
        parts.append("No episode log found.")

    return "\n".join(parts)


def load_retry_history(ep_dir: Path) -> list[dict[str, Any]]:
    """Load error info from archived copies of an episode directory.

    Archived dirs live at ``{ep_dir.parent}/{ep_dir.name}.archived_{timestamp}``.
    Returns a list sorted by timestamp (oldest first), each entry containing:
      timestamp, status, error_type, error_message, failure_text.
    """
    history: list[dict[str, Any]] = []
    archived_prefix = f"{ep_dir.name}.archived_"
    for candidate in ep_dir.parent.iterdir():
        if not candidate.is_dir() or not candidate.name.startswith(archived_prefix):
            continue
        try:
            ts = float(candidate.name[len(archived_prefix) :])
        except ValueError:
            ts = 0.0
        entry: dict[str, Any] = {
            "timestamp": ts,
            "status": None,
            "error_type": None,
            "error_message": None,
            "failure_text": None,
        }
        status_path = candidate / "status.json"
        if status_path.exists():
            try:
                data = json.loads(status_path.read_text())
                entry["status"] = data.get("status")
                entry["error_type"] = data.get("error_type")
                entry["error_message"] = data.get("error_message")
            except Exception:
                pass
        failure_path = candidate / "failure.txt"
        if failure_path.exists():
            try:
                entry["failure_text"] = failure_path.read_text()
            except Exception:
                pass
        history.append(entry)
    history.sort(key=lambda e: e["timestamp"])
    return history


def render_retry_history_md(history: list[dict[str, Any]], traj: Trajectory) -> str:
    """Render retry history as markdown for the Retries tab."""
    retry_count = traj.metadata.get("_retry_count", 0)
    if not retry_count:
        return "No retries for this trajectory."
    if not history:
        return f"This trajectory was retried {retry_count}× but no archived attempt data was found on disk."

    parts: list[str] = [f"**{retry_count} retry attempt(s)** — showing oldest first.\n"]
    for i, entry in enumerate(history, start=1):
        status = entry["status"] or "unknown"
        parts.append(f"---\n### Attempt {i} — `{status}`")
        if entry["error_type"]:
            parts.append(f"**Error type:** `{entry['error_type']}`")
        if entry["error_message"]:
            parts.append(f"**Error message:** {entry['error_message']}")
        if entry["failure_text"]:
            parts.append(f"**Stack trace:**\n```\n{entry['failure_text'].strip()}\n```")
        if not entry["error_type"] and not entry["error_message"] and not entry["failure_text"]:
            parts.append("*(No error detail recorded for this attempt.)*")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _compute_token_stats_for_trajectory(traj: Trajectory) -> dict[str, int | float]:
    """Sum token usage across all AgentOutput LLM calls in one trajectory."""
    stats: dict[str, int | float] = {
        "prompt": 0,
        "completion": 0,
        "cached": 0,
        "cache_created": 0,
        "cost": 0.0,
    }
    for traj_step in traj.steps:
        if isinstance(traj_step.output, AgentOutput):
            for llm_call in traj_step.output.llm_calls:
                if llm_call.usage:
                    stats["prompt"] = int(stats["prompt"]) + llm_call.usage.prompt_tokens
                    stats["completion"] = int(stats["completion"]) + llm_call.usage.completion_tokens
                    stats["cached"] = int(stats["cached"]) + llm_call.usage.cached_tokens
                    stats["cache_created"] = int(stats["cache_created"]) + llm_call.usage.cache_creation_tokens
                    stats["cost"] = float(stats["cost"]) + llm_call.usage.cost
    return stats


def compute_trajectory_stats(traj: Trajectory) -> dict[str, Any]:
    """Compute per-trajectory statistics.

    Returns dict with: n_env_steps, n_agent_steps, total_actions, total_llm_calls,
    duration, prompt_tokens, completion_tokens, cached_tokens, cache_creation_tokens,
    cost, final_reward.
    """
    if traj.summary_stats:
        return traj.summary_stats
    n_env_steps = 0
    n_agent_steps = 0
    total_actions = 0
    total_llm_calls = 0

    for traj_step in traj.steps:
        if isinstance(traj_step.output, EnvironmentOutput):
            n_env_steps += 1
        elif isinstance(traj_step.output, AgentOutput):
            n_agent_steps += 1
            total_actions += len(traj_step.output.actions)
            total_llm_calls += len(traj_step.output.llm_calls)

    duration = None
    if traj.start_time is not None and traj.end_time is not None:
        duration = traj.end_time - traj.start_time

    final_reward = 0.0
    if traj.reward_info:
        final_reward = traj.reward_info.get("reward", 0.0)
    else:
        for traj_step in reversed(traj.steps):
            if isinstance(traj_step.output, EnvironmentOutput):
                final_reward = traj_step.output.reward
                break

    token_stats = _compute_token_stats_for_trajectory(traj)

    return {
        "n_env_steps": n_env_steps,
        "n_agent_steps": n_agent_steps,
        "total_actions": total_actions,
        "total_llm_calls": total_llm_calls,
        "duration": duration,
        "prompt_tokens": token_stats["prompt"],
        "completion_tokens": token_stats["completion"],
        "cached_tokens": token_stats["cached"],
        "cache_creation_tokens": token_stats["cache_created"],
        "cost": token_stats["cost"],
        "final_reward": final_reward,
    }


def _finished_rewards(trajectories: list[Trajectory]) -> list[float]:
    """Return final rewards for trajectories that ran to completion.

    Includes success, fail, and max_steps — all terminal outcomes where reward is meaningful.
    """
    return [
        compute_trajectory_stats(t)["final_reward"]
        for t in trajectories
        if trajectory_status(t) in ("success", "fail", "max_steps")
    ]


def _reward_mean_stderr(rewards: list[float]) -> tuple[float, float]:
    """Return (mean, stderr) for a list of rewards.

    Delegates to :func:`cube_harness.analyze.stats.reward_mean_stderr` so XRay
    and ``scripts/experiments_report.py`` produce identical CIs for the same data
    (auto-selects binomial vs sample SE by data shape).
    """
    return reward_mean_stderr(rewards)


def compute_experiment_stats(trajectories: list[Trajectory]) -> str:
    """Aggregate statistics across all trajectories and return as markdown."""
    if not trajectories:
        return ""

    finished_rewards: list[float] = []
    finished_steps: list[int] = []
    finished_durations: list[float] = []
    n_in_flight = 0
    n_max_steps = 0
    n_stale = 0
    n_cancelled = 0
    n_errored = 0

    total_prompt = 0
    total_completion = 0
    total_cached = 0
    total_cache_created = 0
    total_cost = 0.0

    for traj in trajectories:
        stats = compute_trajectory_stats(traj)
        status = trajectory_status(traj)

        if status in ("success", "fail"):
            finished_rewards.append(stats["final_reward"])
            finished_steps.append(stats["n_env_steps"])
            if stats["duration"] is not None:
                finished_durations.append(stats["duration"])
        elif status in IN_FLIGHT_STATUSES:
            n_in_flight += 1
        elif status == "max_steps":
            n_max_steps += 1
        elif status == "stale":
            n_stale += 1
        elif status == "cancelled":
            n_cancelled += 1
        else:  # "failed" or legacy "system_error"
            n_errored += 1

        total_prompt += stats["prompt_tokens"]
        total_completion += stats["completion_tokens"]
        total_cached += stats["cached_tokens"]
        total_cache_created += stats["cache_creation_tokens"]
        total_cost += stats["cost"]

    n_finished = len(finished_rewards)
    n_completed = n_finished + n_max_steps  # all terminal outcomes
    n_total = n_completed + n_in_flight + n_stale + n_cancelled + n_errored

    stats_parts = [f"📊 **{n_total}** trajectories"]
    summary_parts = []
    if n_completed > 0:
        summary_parts.append(f"✓ Completed: **{n_completed}**")
    if n_in_flight > 0:
        summary_parts.append(f"▶️ Running: **{n_in_flight}**")
    if n_stale > 0:
        summary_parts.append(f"👻 Stale: **{n_stale}**")
    if n_cancelled > 0:
        summary_parts.append(f"🚫 Cancelled: **{n_cancelled}**")
    if n_errored > 0:
        summary_parts.append(f"⛔ Failed: **{n_errored}**")
    if summary_parts:
        stats_parts.append("│ " + " │ ".join(summary_parts))

    if n_finished > 0:
        avg_reward = sum(finished_rewards) / n_finished
        avg_steps = sum(finished_steps) / n_finished
        success_rate = sum(1 for r in finished_rewards if r > 0) / n_finished * 100
        stats_parts.append(f"│ Avg Reward: **{avg_reward:.2f}**")
        stats_parts.append(f"│ Success Rate: **{success_rate:.0f}%**")
        stats_parts.append(f"│ Avg Steps: **{avg_steps:.1f}**")
        if finished_durations:
            avg_duration = sum(finished_durations) / len(finished_durations)
            stats_parts.append(f"│ Avg Duration: **{format_duration(avg_duration)}**")

    result = " ".join(stats_parts)

    if total_prompt > 0:
        token_parts = [f"📊 prompt: **{total_prompt:,}**"]
        token_parts.append(f"completion: **{total_completion:,}**")
        token_parts.append(f"total: **{total_prompt + total_completion:,}**")
        if total_cached > 0:
            cache_pct = total_cached / total_prompt * 100
            token_parts.append(f"cached: **{total_cached:,}** ({cache_pct:.0f}%)")
        if total_cache_created > 0:
            token_parts.append(f"cache_created: **{total_cache_created:,}**")
        if total_cost > 0:
            token_parts.append(f"💰 **${total_cost:.4f}**")
        result += "\n\n" + " │ ".join(token_parts)

    return result


# ---------------------------------------------------------------------------
# Agent / Task / Seed hierarchy tables
# ---------------------------------------------------------------------------
#
# NOTE: The per-trajectory stats shown in the task and seed tables (n_steps, tokens, cost)
# are derived by iterating over all loaded TrajectoryStep objects.  When trajectories are
# loaded as metadata stubs (steps=[]), these values will show "-" until the full trajectory
# is loaded — either by the user clicking on a seed, or by the background bulk-loading
# thread in xray.py.
#
# This is a temporary workaround.  The long-term solution is to have the evaluation loop
# persist per-episode summary stats (n_steps, prompt_tokens, completion_tokens, total_cost,
# duration) directly into the *.metadata.json file as each episode completes.  That would
# make all table columns immediately available at experiment-open time without any bulk loading.
# ---------------------------------------------------------------------------


def build_agent_table(trajectories: list[Trajectory]) -> list[dict[str, Any]]:
    """Build one row per unique agent for the top-level agent table.

    Groups trajectories by metadata.get('agent_name', 'unknown').
    Columns: agent_name, avg_reward, status, total_cost

    status — ``[count][symbol] + ... = total`` cell, e.g. ``15✓ + 4▶️ + 2🎬 = 21``.
             success and fail both collapse to ✓ (avg_reward already captures the breakdown).
    total_cost shows "-" when no cost data is available (e.g. unloaded trajectory stubs).
    """
    groups: dict[str, list[Trajectory]] = {}
    for traj in trajectories:
        agent_key = traj.metadata.get("agent_name", "unknown")
        groups.setdefault(agent_key, []).append(traj)

    rows = []
    for agent_key in sorted(groups.keys()):
        agent_trajs = groups[agent_key]
        all_stats = [compute_trajectory_stats(t) for t in agent_trajs]
        statuses = [trajectory_status(t) for t in agent_trajs]
        finished = _finished_rewards(agent_trajs)
        total_cost = sum(float(s["cost"]) for s in all_stats)
        mean, stderr = _reward_mean_stderr(finished)
        cost_str = f"${total_cost:.4f}" if total_cost > 0 else "-"

        rows.append(
            {
                "agent_name": agent_key,
                "avg_reward": f"{mean:.3f} ± {stderr:.3f}",
                "status": _build_status_cell(statuses),
                "total_cost": cost_str,
            }
        )
    return rows


def build_trajectory_table(trajectories: list[Trajectory], agent_key: str) -> list[dict[str, Any]]:
    """Build one row per trajectory for a selected agent.

    Filters trajectories to those matching agent_key.
    Displayed columns: status, task_id, [seed,] n_steps, duration, tokens, cost
    The seed column is omitted when all trajectories have seed=None.
    Hidden key _traj_id carries the full trajectory ID for selection.
    Sorted by task_id then start_time within a task.
    """
    agent_trajs = [t for t in trajectories if t.metadata.get("agent_name", "unknown") == agent_key]
    agent_trajs.sort(key=lambda t: (t.metadata.get("task_id", "unknown"), t.start_time is None, t.start_time or 0))

    include_seed = any(t.metadata.get("seed") is not None for t in agent_trajs)

    rows = []
    for traj in agent_trajs:
        stats = compute_trajectory_stats(traj)
        task_id = traj.metadata.get("task_id", "unknown")
        status = trajectory_status(traj)
        retry_count = traj.metadata.get("_retry_count", 0)
        retry_badge = f" <sup style='color:#888;font-size:9px'>×{retry_count}</sup>" if retry_count else ""
        duration_str = format_duration(stats["duration"]) if stats["duration"] is not None else "-"
        total_tokens = int(stats["prompt_tokens"]) + int(stats["completion_tokens"])
        tokens_str = f"{total_tokens:,}" if total_tokens > 0 else "-"
        cost_str = f"${float(stats['cost']):.4f}" if float(stats["cost"]) > 0 else "-"
        row: dict[str, Any] = {
            "_traj_id": traj.id,
            "status": _STATUS_HTML[status] + retry_badge,
            "task_id": html_lib.escape(task_id),
        }
        if include_seed:
            row["seed"] = traj.metadata.get("seed")
        row["n_steps"] = stats["n_env_steps"]
        row["duration"] = duration_str
        row["tokens"] = tokens_str
        row["cost"] = cost_str
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Timeline HTML generation
# ---------------------------------------------------------------------------

_ENV_COLOR = "#a1c9f4"
_AGENT_COLOR = "#8de5a1"
_CURRENT_BORDER_COLOR = "#ffd700"
_SUCCESS_BORDER_COLOR = "#32cd32"
_FAILURE_BORDER_COLOR = "#dc3545"
_MIN_WIDTH = 12
_MAX_WIDTH = 240

# Muted palette for profiling labels — distinct from env blue and agent green.
_PROFILING_PALETTE = [
    "#f9c784",  # warm amber
    "#d4a8e0",  # soft violet
    "#f4a3a8",  # dusty rose
    "#80cbc4",  # teal
    "#fde68a",  # pale gold
    "#ff9a6c",  # soft orange
]


def _compute_step_width(
    duration: float | None,
    min_duration: float,
    max_duration: float,
    min_width: int = _MIN_WIDTH,
    max_width: int = _MAX_WIDTH,
) -> int:
    """Compute pixel width for a timeline segment based on its duration."""
    if duration is None or max_duration <= min_duration:
        return min_width
    normalized = (duration - min_duration) / (max_duration - min_duration)
    return int(min_width + normalized * (max_width - min_width))


def _assign_profiling_colors(labels: list[str]) -> dict[str, str]:
    """Map profiling labels to palette colors in a stable, insertion-order manner."""
    return {label: _PROFILING_PALETTE[i % len(_PROFILING_PALETTE)] for i, label in enumerate(labels)}


def _build_profiling_strip_css(
    profiling: dict[str, tuple[float, float]],
    label_colors: dict[str, str],
    agent_t_start: float | None = None,
    agent_t_end: float | None = None,
    env_pct: float = 0.0,
) -> str | None:
    """Build a CSS linear-gradient for the top profiling strip.

    The strip is anchored to the agent portion of the bar (env_pct% → 100%).
    When agent_t_start/end are provided, profiling entries are positioned using the
    agent step's absolute timestamps so the strip aligns with the green agent zone.
    Falls back to spanning the profiling window if agent timing is unavailable.
    Returns None if profiling is empty or the total duration is zero.
    """
    if not profiling:
        return None
    entries = sorted(
        [(start, end, label) for label, (start, end) in profiling.items() if end > start],
        key=lambda x: x[0],
    )
    if not entries:
        return None

    agent_span = 100.0 - env_pct  # bar-% width of the agent zone

    if agent_t_start is not None and agent_t_end is not None and agent_t_end > agent_t_start:
        ref_start = agent_t_start
        ref_duration = agent_t_end - agent_t_start
    else:
        # Fallback: map relative to the profiling window, over the agent zone only.
        ref_start = entries[0][0]
        ref_end = max(e for _, e, _ in entries)
        ref_duration = ref_end - ref_start

    if ref_duration <= 0:
        return None

    stops: list[str] = []
    # Cover the env portion with transparent so colors stay in the agent zone.
    if env_pct > 0.5:
        stops.append(f"transparent 0% {env_pct:.1f}%")

    cursor_pct = env_pct
    for seg_start, seg_end, label in entries:
        pct_start = env_pct + (seg_start - ref_start) / ref_duration * agent_span
        pct_end = env_pct + (seg_end - ref_start) / ref_duration * agent_span
        # Clamp to the agent portion.
        pct_start = max(env_pct, min(100.0, pct_start))
        pct_end = max(env_pct, min(100.0, pct_end))
        if pct_end <= pct_start:
            continue
        color = label_colors.get(label, _AGENT_COLOR)
        if pct_start > cursor_pct + 0.5:
            stops.append(f"transparent {cursor_pct:.1f}% {pct_start:.1f}%")
        stops.append(f"{color} {pct_start:.1f}% {pct_end:.1f}%")
        cursor_pct = pct_end
    if cursor_pct < 99.5:
        stops.append(f"transparent {cursor_pct:.1f}% 100%")
    if not stops or stops == [f"transparent 0% {env_pct:.1f}%"]:
        return None
    return f"linear-gradient(to right, {', '.join(stops)})"


def _build_segment_html(
    step_idx: int,
    is_current: bool,
    total_width: int,
    tooltip: str,
    env_frac: float,
    done: bool = False,
    reward: float = 0.0,
    profiling_strip_css: str | None = None,
) -> str:
    """Build the HTML div for one timeline segment.

    The main bar uses a two-color env/agent gradient. When profiling_strip_css is provided,
    a thin colored strip is rendered at the top of the bar (like the done border at the bottom).
    """
    border = f"3px solid {_CURRENT_BORDER_COLOR}" if is_current else "1px solid #ccc"
    box_shadow = "0 0 8px rgba(255, 215, 0, 0.8)" if is_current else "none"

    done_border = ""
    if done:
        done_color = _SUCCESS_BORDER_COLOR if reward > 0 else _FAILURE_BORDER_COLOR
        done_border = f"border-bottom: 4px solid {done_color};"

    step_num = step_idx + 1
    env_pct = int(env_frac * 100)
    gradient = (
        _ENV_COLOR
        if env_pct == 100
        else (f"linear-gradient(to right, {_ENV_COLOR} {env_pct}%, {_AGENT_COLOR} {env_pct}%)")
    )

    strip_html = ""
    if profiling_strip_css:
        strip_html = (
            f'<div style="position: absolute; top: 0; left: 0; right: 0; height: 5px;'
            f' background: {profiling_strip_css}; border-radius: 3px 3px 0 0; pointer-events: none;"></div>'
        )

    # Use native setter to properly trigger Gradio's change detection
    onclick = (
        f"const inp = document.querySelector('#timeline_click_input input, #timeline_click_input textarea');"
        f" if(inp) {{"
        f" const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;"
        f" nativeSetter.call(inp, {step_idx});"
        f" inp.dispatchEvent(new Event('input', {{bubbles: true}}));"
        f" inp.dispatchEvent(new Event('change', {{bubbles: true}}));"
        f" }}"
    )

    return (
        f'<div class="timeline-step" data-step="{step_idx}" title="{tooltip}" onclick="{onclick}" style="'
        f"position: relative; display: inline-flex; align-items: center; justify-content: center;"
        f" min-width: {total_width}px; height: 36px; margin: 2px;"
        f" background: {gradient}; border: {border}; border-radius: 4px;"
        f" cursor: pointer; font-size: 11px; font-weight: bold; color: #333;"
        f" box-shadow: {box_shadow}; {done_border} transition: transform 0.1s;"
        f'"'
        f" onmouseover=\"this.style.transform='scale(1.1)'\" onmouseout=\"this.style.transform='scale(1)'\">"
        f"{strip_html}{step_num}</div>"
    )


def _collect_profiling_labels(trajectory: Trajectory) -> list[str]:
    """Return unique profiling labels across all AgentOutputs, in first-seen order."""
    seen: dict[str, None] = {}
    for ts in trajectory.steps:
        if isinstance(ts.output, AgentOutput):
            for label in ts.output.profiling:
                seen.setdefault(label, None)
    return list(seen.keys())


def generate_timeline_html(trajectory: Trajectory | None, current_step: int) -> str:
    """Generate an HTML timeline with one segment per UI step (EnvironmentOutput).

    current_step is a UI step index (0-based index into env steps).
    Each segment's width scales with total (env+agent) duration.
    The segment bar is split left→right: env time (blue) then agent time (green), with the
    agent portion further subdivided by AgentOutput.profiling intervals when present.
    """
    if trajectory is None or not trajectory.steps:
        return "<div style='padding: 10px; color: #666;'>No trajectory loaded</div>"

    env_steps: list[tuple[int, EnvironmentOutput]] = [
        (i, ts.output)  # type: ignore[misc]
        for i, ts in enumerate(trajectory.steps)
        if isinstance(ts.output, EnvironmentOutput)
    ]

    if not env_steps:
        return "<div style='padding: 10px; color: #666;'>No environment steps found</div>"

    # Assign stable colors to every profiling label found in this trajectory.
    profiling_labels = _collect_profiling_labels(trajectory)
    label_colors = _assign_profiling_colors(profiling_labels)

    def _raw_duration(raw_idx: int) -> float | None:
        ts = trajectory.steps[raw_idx]
        if ts.start_time is not None and ts.end_time is not None:
            return ts.end_time - ts.start_time
        return None

    # Pre-compute per-UI-step env and agent durations
    env_durs: list[float | None] = []
    agent_durs: list[float | None] = []
    total_durs: list[float | None] = []
    for raw_idx, _ in env_steps:
        ed = _raw_duration(raw_idx)
        next_idx = raw_idx + 1
        has_agent = next_idx < len(trajectory.steps) and isinstance(trajectory.steps[next_idx].output, AgentOutput)
        ad = _raw_duration(next_idx) if has_agent else None
        env_durs.append(ed)
        agent_durs.append(ad)
        total = (ed or 0.0) + (ad or 0.0)
        total_durs.append(total if (ed is not None or ad is not None) else None)

    valid_totals = [d for d in total_durs if d is not None and d > 0]
    min_total = min(valid_totals) if valid_totals else 0.0
    max_total = max(valid_totals) if valid_totals else 1.0

    steps_html = []
    for ui_idx, (raw_idx, env_out) in enumerate(env_steps):
        is_current = ui_idx == current_step
        total_width = _compute_step_width(total_durs[ui_idx], min_total, max_total)

        ed = env_durs[ui_idx] or 0.0
        ad = agent_durs[ui_idx] or 0.0
        total = ed + ad
        env_frac = (ed / total) if total > 0 else 1.0

        # Collect agent step profiling for the top strip.
        agent_out: AgentOutput | None = None
        agent_ts_start: float | None = None
        agent_ts_end: float | None = None
        next_idx = raw_idx + 1
        if next_idx < len(trajectory.steps):
            next_ts = trajectory.steps[next_idx]
            if isinstance(next_ts.output, AgentOutput):
                agent_out = next_ts.output
                agent_ts_start = next_ts.start_time
                agent_ts_end = next_ts.end_time

        profiling = agent_out.profiling if agent_out is not None else {}
        profiling_strip_css = _build_profiling_strip_css(
            profiling, label_colors, agent_ts_start, agent_ts_end, env_frac * 100.0
        )

        # Build tooltip as a timing tree.
        timing_parts = []
        if ed > 0:
            timing_parts.append(f"env: {format_duration(ed)}")
        if ad > 0:
            timing_parts.append(f"agent: {format_duration(ad)}")
        tooltip = f"Step {ui_idx + 1}"
        if timing_parts:
            tooltip += f" ({' + '.join(timing_parts)})"
        tree_lines: list[str] = []
        if ed > 0:
            tree_lines.append(f"  env: {format_duration(ed)}")
        if ad > 0:
            tree_lines.append(f"  agent: {format_duration(ad)}")
            for lbl, (start, end) in profiling.items():
                tree_lines.append(f"    {lbl}: {format_duration(end - start)}")
        if env_out.done:
            tree_lines.append(f"  reward: {env_out.reward:.2f}")
        if tree_lines:
            tooltip += "\n" + "\n".join(tree_lines)

        steps_html.append(
            _build_segment_html(
                ui_idx,
                is_current,
                total_width,
                tooltip,
                env_frac,
                env_out.done,
                env_out.reward,
                profiling_strip_css,
            )
        )

    # Legend row 1: bar colors and step indicators (always shown).
    row1_parts = [
        f'<div style="display: flex; align-items: center; gap: 4px;">'
        f'<div style="width: 22px; height: 14px;'
        f" background: linear-gradient(to right, {_ENV_COLOR} 50%, {_AGENT_COLOR} 50%);"
        f' border-radius: 3px;"></div>'
        f"<span>Env | Agent time</span></div>",
        f'<div style="display: flex; align-items: center; gap: 4px;">'
        f'<div style="width: 16px; height: 16px; border: 2px solid {_CURRENT_BORDER_COLOR}; border-radius: 3px;"></div>'
        f"<span>Current</span></div>",
        f'<div style="display: flex; align-items: center; gap: 4px;">'
        f'<div style="width: 16px; height: 16px; border-bottom: 3px solid {_SUCCESS_BORDER_COLOR}; background: #ddd; border-radius: 3px;"></div>'
        f"<span>Success</span></div>",
        f'<div style="display: flex; align-items: center; gap: 4px;">'
        f'<div style="width: 16px; height: 16px; border-bottom: 3px solid {_FAILURE_BORDER_COLOR}; background: #ddd; border-radius: 3px;"></div>'
        f"<span>Failure</span></div>",
    ]
    legend_html = (
        '<div style="display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 4px; font-size: 12px; color: #666;">'
        + "".join(row1_parts)
        + "</div>"
    )

    # Legend row 2: profiling labels (only when profiling data is present).
    if profiling_labels:
        row2_parts = [
            f'<div style="display: flex; align-items: center; gap: 4px;">'
            f'<div style="width: 16px; height: 5px; background: {label_colors[lbl]}; border-radius: 2px; border: 1px solid #bbb;"></div>'
            f"<span>{lbl}</span></div>"
            for lbl in profiling_labels
        ]
        legend_html += (
            '<div style="display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 4px; font-size: 12px; color: #666;">'
            '<span style="color:#999;">Agent profiling:</span>' + "".join(row2_parts) + "</div>"
        )

    return (
        f'<div style="padding: 10px; background: #f8f9fa; border-radius: 8px;">'
        f"{legend_html}"
        f'<div id="timeline-container" style="'
        f"display: flex; flex-wrap: wrap; align-items: center; padding: 8px;"
        f" background: white; border-radius: 6px; border: 1px solid #dee2e6;"
        f' max-height: 120px; overflow-y: auto;">'
        f"{''.join(steps_html)}"
        f"</div></div>"
    )

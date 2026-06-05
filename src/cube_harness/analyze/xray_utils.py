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
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Optional Tk fallback for the directory picker on non-macOS platforms.
# Imported at module level (EX-001); guarded since headless hosts may lack Tk.
try:
    import tkinter as _tkinter
    from tkinter import filedialog as _tk_filedialog
except Exception:  # pragma: no cover - platform without Tk
    _tkinter = None
    _tk_filedialog = None

from cube.benchmark import BenchmarkConfig
from cube.core import EnvironmentOutput, Observation
from PIL import Image

from cube_harness.agent import AgentConfig
from cube_harness.analyze.stats import reward_mean_stderr
from cube_harness.analyze.xray_events import EpisodeEvents, EventGroup
from cube_harness.core import AgentOutput, Trajectory
from cube_harness.episode_status import STATUS_FILENAME, EpisodeStatus, should_sweep_running_to_stale
from cube_harness.episode_status import TERMINAL_STATUSES as _EPISODE_TERMINAL_STATUSES
from cube_harness.exp_runner import DEFAULT_CANCEL_GRACE_S, DEFAULT_STEP_TIMEOUT_S
from cube_harness.experiment_status import EXPERIMENT_STATUS_FILENAME, ExperimentStatus, is_driver_alive
from cube_harness.llm import LLMCall
from cube_harness.reproducibility import scan, submissions

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
    "cancelled": "<span title='Cancelled — deliberately stopped'>⏹️</span>",
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
    "cancelled": "⏹️ Cancelled",
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
    ray_dashboard_urls: list[tuple[str, str]] | None = None,
) -> str:
    """Return an HTML progress bar + label for experiment completion status.

    Args:
        n_completed: Total completed trajectories across all agents.
        n_total: Total trajectories across all agents.
        n_running: Total currently running trajectories.
        per_agent: Optional list of (agent_name, n_completed, n_total, n_running).
                   When provided with > 1 entry, a per-agent breakdown is appended.
        exp_names: Names of selected experiment directories being monitored.
        ray_dashboard_urls: Optional list of (exp_name, url). Each becomes a clickable
                   link to the live Ray dashboard for that experiment.
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

    links = _build_ray_dashboard_links_html(ray_dashboard_urls, multi=bool(exp_names and len(exp_names) > 1))

    if not per_agent or len(per_agent) <= 1:
        return header + bar + label + links

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
    return header + bar + label + links + rows_html


def _build_ray_dashboard_links_html(ray_dashboard_urls: list[tuple[str, str]] | None, *, multi: bool) -> str:
    """Render clickable Ray-dashboard link(s). Empty string when no URLs are available.

    ``multi`` prefixes each link with its experiment name (only useful when several
    experiments are being monitored at once).
    """
    if not ray_dashboard_urls:
        return ""
    parts = []
    for name, url in ray_dashboard_urls:
        safe_url = html_lib.escape(url, quote=True)
        prefix = f"{html_lib.escape(name)}: " if multi else ""
        parts.append(
            f'{prefix}<a href="{safe_url}" target="_blank" rel="noopener" '
            f'style="color:#1d4ed8;text-decoration:none;">🔗 Ray dashboard</a>'
        )
    return '<div style="font-size:11px;margin-top:4px;">' + " &nbsp;·&nbsp; ".join(parts) + "</div>"


def pick_directory(start: Path) -> Path | None:
    """Open a native folder picker rooted at `start`; return the chosen dir.

    Runs on the XRay host (a local tool), so the dialog appears on the user's
    own desktop. Uses macOS `osascript` (no GUI-thread constraints, unlike Tk in
    a Gradio worker thread) and falls back to Tk elsewhere. Returns None if the
    user cancels, the picker is unavailable, or the choice isn't a directory.
    """
    chosen: str | None = None
    if sys.platform == "darwin":
        script = (
            f'set startDir to POSIX file "{start}"\n'
            'set d to choose folder with prompt "Select XRay results directory" default location startDir\n'
            "POSIX path of d"
        )
        try:
            result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode == 0:  # non-zero == user cancelled
            chosen = result.stdout.strip()
    elif _tkinter is not None and _tk_filedialog is not None:
        try:
            root = _tkinter.Tk()
            root.withdraw()
            chosen = _tk_filedialog.askdirectory(initialdir=str(start))
            root.destroy()
        except Exception:  # pragma: no cover - display-less host
            return None

    if not chosen:
        return None
    path = Path(chosen).expanduser()
    return path if path.is_dir() else None


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
# Bump when the cached row schema/semantics change so stale caches recompute.
# v3: added _category (eligibility) + _ran/_total for the incomplete badge.
# v4: added _subs_mtime so the cache invalidates when submissions.json is
#     written, updated, OR deleted (a submit, or a rollback that clears it).
# v5: dropped the `incomplete` category (#491 records subset n_tasks at the
#     source); replaced _ran/_total with _is_official (drives Archive auto-select).
_EXP_ROW_VERSION = 5


def _subs_mtime(exp_dir: Path) -> float:
    """mtime of submissions.json, or 0.0 when absent. Stored on the cached row so
    it invalidates whenever the submission state changes — including deletion
    (absent → 0.0 ≠ the stored mtime), which an mtime-vs-cache check would miss."""
    try:
        return (exp_dir / submissions.SUBMISSIONS_FILENAME).stat().st_mtime
    except OSError:
        return 0.0


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

    Does NOT cover submissions.json (it lives outside episodes/ and can be
    *deleted*, which an mtime-vs-cache comparison misses) — that's handled
    separately in get_experiments_table_rows via the cached `_subs_mtime`.
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


# --- Submission eligibility (clean + submit) -------------------------------
# Compact badges for the Experiments-table "eligibility" column. The raw
# category (stored on the hidden `_category` key) drives the Garbage-collect /
# Check-submit auto-selection.

_ELIGIBILITY_BADGES: dict[str, str] = {
    "submittable": "<span title='Clean run — ready to submit'>🟢 submittable</span>",
    "subset_review": "<span title='Passed integrity checks but the subset shape needs a human look (submit with review / mark is_official)'>🔍 review</span>",
    "unfinished": "<span title='Episodes still queued/running, or tasks missing a status file — state may change'>⏳ unfinished</span>",
    "broken": "<span title='Cannot produce a meaningful score'>💥 broken</span>",
    # Neutral fallback only: a real journal decision is always resolved to ✅
    # submitted or 🚫 rejected by eligibility_badge's fresh submissions read, so
    # this never renders in practice — keep it neutral (not a green ✅) so a
    # stray decided-but-unresolved state can't be mistaken for a success.
    "already_submitted": "<span title='Has a prior submission decision' style='color:#888'>• decided</span>",
}


def scan_category(exp_dir: Path, *, sweep_stale: bool = False) -> str:
    """The `ScanCategory` value for one experiment (``"broken"`` on any error).

    The expensive part (reads `experiment_record.json` + per-episode statuses),
    so it's cached on the row's hidden `_category` key. `sweep_stale=True` lets a
    cleanup pass reclassify dead RUNNING/QUEUED episodes as broken first."""
    try:
        return scan.classify(exp_dir, sweep_stale=sweep_stale).category.value
    except Exception:  # pragma: no cover - defensive
        return "broken"


def eligibility_badge(exp_dir: Path, category: str) -> str:
    """Badge for the eligibility column. The persisted submission state (read
    fresh, cheap) takes precedence over the cached scan `category`: a successful
    submission shows ✅; a recorded rejection shows 🚫 rejected (with its reason),
    so a previously-rejected/broken run is never mistaken for a success."""
    subs = submissions.read(exp_dir)
    submitted = [d for d in ("journal", "eee") if subs.get(d, {}).get("status") == "submitted"]
    if submitted:
        names = " + ".join({"journal": "registry", "eee": "eee"}[d] for d in submitted)
        return f"<span title='Submitted to {names}'>✅ {names}</span>"
    rejected = next((subs[d] for d in ("journal", "eee") if subs.get(d, {}).get("status") == "rejected"), None)
    if rejected is not None:
        reason = html_lib.escape(rejected.get("reason", "previously rejected"))
        return f"<span title='{reason}'>🚫 rejected</span>"
    if any(subs.get(d, {}).get("status") == "pending" for d in ("journal", "eee")):
        return "<span title='Submission in progress'>📤 submitting…</span>"
    failed = next((subs[d] for d in ("journal", "eee") if subs.get(d, {}).get("status") == "failed"), None)
    if failed is not None:
        reason = html_lib.escape(failed.get("reason", "submission failed"))
        return f"<span title='Last submit attempt failed (retryable): {reason}'>❌ submit failed</span>"
    return _ELIGIBILITY_BADGES.get(category, f"<span>{html_lib.escape(category)}</span>")


def is_archivable(exp_dir: Path, category: str, is_official: bool | None = None) -> bool:
    """True for runs the Archive auto-select should tick: a broken scan, a run
    recorded as rejected (e.g. an all-ghost run), or one explicitly marked debug
    (``is_official is False``).

    ``is_official is True`` is an absolute keep — the operator vouched for / pinned
    the run (e.g. a reference submission), so it is *never* auto-archived, even if
    broken or rejected. A bare ``subset_review`` is also kept (it may be a legit
    subset awaiting a `--yes` submission)."""
    if is_official is True:
        return False
    if category == "broken" or is_official is False:
        return True
    subs = submissions.read(exp_dir)
    return any(subs.get(d, {}).get("status") == "rejected" for d in ("journal", "eee"))


def is_submittable_pick(exp_dir: Path, category: str) -> bool:
    """True for the Submit auto-select: a submittable run that has NOT already
    been submitted and is NOT mid-submission. Reads submissions.json *fresh* so a
    run submitted earlier in this session is never re-ticked — the cached scan
    `category` can lag a submit (submissions.json lives outside episodes/)."""
    if category != "submittable":
        return False
    subs = submissions.read(exp_dir)
    return not any(subs.get(d, {}).get("status") in ("submitted", "pending") for d in ("journal", "eee"))


def persist_broken_rejection(exp_dir: Path) -> bool:
    """If *exp_dir* classifies as broken and has no prior journal decision, stamp
    a rejection into submissions.json (mirrors ``scan_experiments.py
    --persist-broken``). Called on archive so a broken run's verdict is durable
    and travels with the dir into ``_archive/``. Returns True when newly written."""
    try:
        result = scan.classify(exp_dir, sweep_stale=False)
    except Exception:  # pragma: no cover - defensive
        return False
    if result.category is not scan.ScanCategory.broken or submissions.has_decision(exp_dir, "journal"):
        return False
    reason = result.reasons[0] if result.reasons else "broken (archived from XRay)"
    submissions.record_rejected(exp_dir, "journal", reason=f"broken: {reason}")
    return True


def _compute_exp_row(exp_dir: Path) -> dict[str, Any]:
    """Compute display fields for one experiment by reading per-episode status.json files.

    V2 (episodes/ dir): reads status.json per episode — O(N_episodes) JSON reads.
    V1 (flat *.metadata.json): reads trajectory metadata per episode.
    Returns: {date, agent, model, benchmark, status, avg_reward}.
    """
    cfg_info = _parse_experiment_config(exp_dir)

    # Single source of truth: one classification pass reads the per-episode
    # statuses AND yields the eligibility category. For record-bearing runs it
    # also returns the status breakdown + rewards, so we don't re-read the status
    # files here. Record-less / V1 layouts (which the scanner can't classify) fall
    # back to a direct read.
    result = scan.classify(exp_dir, sweep_stale=False)
    if result.status_counts:
        statuses = [
            display
            for raw, n in result.status_counts.items()
            for display in [_RAW_STATUS_MAP.get(raw, "system_error")] * n
        ]
        rewards = list(result.rewards)
    else:
        statuses, rewards = _read_display_statuses(exp_dir)

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
        # Cached scan category (stable for terminal runs); the displayed badge is
        # derived fresh in get_experiments_table_rows so submissions stay current.
        "_category": result.category.value,
        # Operator run-intent (None/True/False) — drives the Archive auto-select
        # (is_official=False ⇒ explicit debug ⇒ archivable).
        "_is_official": result.is_official,
        # Submission state at compute time, so the cache invalidates on a later
        # submit / rollback (see _subs_mtime).
        "_subs_mtime": _subs_mtime(exp_dir),
        "_v": _EXP_ROW_VERSION,
    }


def _read_display_statuses(exp_dir: Path) -> tuple[list[str], list[float]]:
    """Fallback per-episode read for layouts the scanner can't classify
    (record-less V2, or V1 flat ``*.metadata.json``). Returns
    ``(display_statuses, rewards)``."""
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
    return statuses, rewards


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
        row: dict[str, Any] | None = None
        if cache_path.exists():
            try:
                cache_mtime = cache_path.stat().st_mtime
                # Older-schema caches are treated as stale (recompute + rewrite).
                if _is_cache_valid(dir_path, cache_mtime):
                    with open(cache_path) as f:
                        cached = json.load(f)
                    # Valid only if the schema matches AND the submission state is
                    # unchanged since the cache was written (a submit or a rollback
                    # that deletes submissions.json must force a reclassify).
                    if cached.get("_v") == _EXP_ROW_VERSION and cached.get("_subs_mtime") == _subs_mtime(dir_path):
                        row = {"selected": False, "experiment": dir_path.name, **cached}
            except Exception as exc:
                logger.debug("Cache read failed for %s: %s", cache_path, exc)
        if row is None:
            _promote_ghost_episodes(dir_path)
            summary = _compute_exp_row(dir_path)
            if _all_episodes_terminal(dir_path):
                try:
                    tmp = cache_path.parent / (cache_path.name + ".tmp")
                    tmp.write_text(json.dumps(summary, indent=2))
                    os.replace(tmp, cache_path)
                except Exception as exc:
                    logger.debug("Cache write failed for %s: %s", cache_path, exc)
            row = {"selected": False, "experiment": dir_path.name, **summary}
        # Derive the eligibility badge fresh (submissions.json is cheap and can
        # change after a submit without invalidating the mtime-based cache).
        row["eligibility"] = eligibility_badge(dir_path, row.get("_category", "broken"))
        rows.append(row)
    rows.sort(key=lambda r: r["date"], reverse=True)
    return rows


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


_EMPTY_TRAJECTORY_STATS: dict[str, Any] = {
    "n_env_steps": 0,
    "n_agent_steps": 0,
    "total_actions": 0,
    "total_llm_calls": 0,
    "duration": None,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "cached_tokens": 0,
    "cache_creation_tokens": 0,
    "cost": 0.0,
    "final_reward": 0.0,
}


def compute_trajectory_stats(traj: Trajectory) -> dict[str, Any]:
    """Per-trajectory statistics, sourced from the trajectory's ``summary_stats``.

    Keys: n_env_steps, n_agent_steps, total_actions, total_llm_calls, duration,
    prompt_tokens, completion_tokens, cached_tokens, cache_creation_tokens, cost,
    final_reward.

    XRay's metadata stubs carry ``summary_stats`` (aggregated by ``EventStreamer``),
    so that's a pure lookup. Legacy in-memory ``Trajectory`` objects with ``steps``
    (still built by ``inspect_results`` / the Global Report and ``experiments_report``)
    have no ``summary_stats`` — fall back to counting steps and reading the reward
    from ``reward_info`` (or the last env step). Token stats stay zero on this path
    (the legacy ``AgentOutput.llm_calls`` are gone).
    """
    if traj.summary_stats:
        return traj.summary_stats
    if not traj.steps:
        return dict(_EMPTY_TRAJECTORY_STATS)

    stats = dict(_EMPTY_TRAJECTORY_STATS)
    stats["n_env_steps"] = sum(1 for s in traj.steps if isinstance(s.output, EnvironmentOutput))
    stats["n_agent_steps"] = sum(1 for s in traj.steps if isinstance(s.output, AgentOutput))
    stats["total_actions"] = sum(len(s.output.actions) for s in traj.steps if isinstance(s.output, AgentOutput))
    if traj.start_time is not None and traj.end_time is not None:
        stats["duration"] = traj.end_time - traj.start_time
    if traj.reward_info:
        stats["final_reward"] = traj.reward_info.get("reward", 0.0)
    else:
        for s in reversed(traj.steps):
            if isinstance(s.output, EnvironmentOutput):
                stats["final_reward"] = s.output.reward
                break
    return stats


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

        if status in TERMINAL_OUTCOME_STATUSES:
            finished_rewards.append(stats["final_reward"])
            finished_steps.append(stats["n_env_steps"])
            if stats["duration"] is not None:
                finished_durations.append(stats["duration"])
            if status == "max_steps":
                n_max_steps += 1
        elif status in IN_FLIGHT_STATUSES:
            n_in_flight += 1
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
    n_success_fail = n_finished - n_max_steps
    n_total = n_finished + n_in_flight + n_stale + n_cancelled + n_errored

    stats_parts = [f"📊 **{n_total}** trajectories"]
    summary_parts = []
    if n_success_fail > 0:
        summary_parts.append(f"✓ Completed: **{n_success_fail}**")
    if n_max_steps > 0:
        summary_parts.append(f"🎬 Max steps: **{n_max_steps}**")
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
# Per-trajectory stats (n_steps, tokens, cost) come from `compute_trajectory_stats`,
# which reads `Trajectory.summary_stats` — aggregated by EventStreamer and persisted on
# the metadata stub. Stubs with no summary_stats render "-".
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


# ===========================================================================
# Event-stream rendering (agent-owns-loop XRay rewrite)
#
# Everything below consumes the flat `EpisodeEvents` view model (Observation /
# LLMCall / EvaluationEvent payloads) — never EnvironmentOutput / AgentOutput /
# Trajectory.steps. The card rail replaces the horizontal timeline; the detail
# panes render a resolved `EventGroup` instead of a lone step.
# ===========================================================================


# --- Observation extractors ------------------------------------------------


def screenshot_from_obs(obs: Observation | None) -> Image.Image | None:
    """First PIL image in an Observation's contents, or None."""
    if obs is None:
        return None
    for content in obs.contents:
        if isinstance(content.data, Image.Image):
            return content.data
    return None


def text_content_from_obs(obs: Observation | None, name_pattern: str) -> str | None:
    """First text content whose name contains `name_pattern` (case-insensitive)."""
    if obs is None:
        return None
    needle = name_pattern.lower()
    for content in obs.contents:
        if isinstance(content.data, str) and needle in (content.name or "").lower():
            return content.data
    return None


def goal_from_events(events: EpisodeEvents | None) -> str:
    """Task goal text — the first non-empty text content of the first observation."""
    if events is None or len(events) == 0:
        return "*No trajectory loaded*"
    for i in range(len(events)):
        obs = events.observation(i)
        if obs is None:
            continue
        for content in obs.contents:
            if isinstance(content.data, str) and content.data.strip():
                return content.data
    return "*No goal text found*"


# --- Card rail -------------------------------------------------------------

_RAIL_MIN_H = 44  # px — shortest card
_RAIL_MAX_H = 140  # px — tallest card


def _card_height(duration: float | None, min_dur: float, max_dur: float) -> int:
    """Map an event duration to a clamped card height in pixels."""
    if duration is None or max_dur <= min_dur:
        return _RAIL_MIN_H
    frac = (duration - min_dur) / (max_dur - min_dur)
    return int(_RAIL_MIN_H + frac * (_RAIL_MAX_H - _RAIL_MIN_H))


def _card_onclick(index: int) -> str:
    """JS that writes `index` into the hidden #timeline_click_input Number.

    Reuses the native-setter pattern the old timeline used so Gradio's change
    detection fires on every click (even when re-selecting the same value).
    """
    return (
        "const inp = document.querySelector('#timeline_click_input input, #timeline_click_input textarea');"
        " if(inp) {"
        " const s = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;"
        f" s.call(inp, {index});"
        " inp.dispatchEvent(new Event('input', {bubbles: true}));"
        " inp.dispatchEvent(new Event('change', {bubbles: true}));"
        " }"
    )


def _event_profiling(events: EpisodeEvents, index: int) -> dict | None:
    """`profiling` dict of an event's payload, if it carries one (LLM calls)."""
    return getattr(events.output(index), "profiling", None) or None


def render_event_rail_html(events: EpisodeEvents | None, selected: int) -> str:
    """Vertical, scrollable column of `.xray-event-card` divs — one per event.

    The active card (index == selected) gets a solid border in its kind colour;
    its group-mates get a muted (dashed) border. Card height scales with the
    event's wall-clock duration. A left-edge stripe marks profiled events.
    """
    if events is None or len(events) == 0:
        return "<div style='padding:10px;color:#666;'>No events to display</div>"

    durations: list[float | None] = []
    for ev in events.events:
        d = ev.end_time - ev.start_time if (ev.start_time is not None and ev.end_time is not None) else None
        durations.append(d)
    valid = [d for d in durations if d is not None and d > 0]
    min_dur, max_dur = (min(valid), max(valid)) if valid else (0.0, 1.0)

    accompanying: set[int] = set(events.accompanying_indices(selected)) if 0 <= selected < len(events) else set()

    cards_html: list[str] = []
    for card in events.cards():
        i = card.index
        is_active = i == selected
        is_accomp = i in accompanying
        if is_active:
            border = f"2px solid {card.color}"
            shadow = f"box-shadow:0 0 0 2px {card.color}33;"
        elif is_accomp:
            border = f"2px dashed {card.color}99"
            shadow = ""
        else:
            border = "1px solid #e2e8f0"
            shadow = ""
        height = _card_height(durations[i], min_dur, max_dur)
        stripe = ""
        if _event_profiling(events, i):
            stripe = (
                "<div style='position:absolute;left:0;top:0;bottom:0;width:4px;"
                f"background:{card.color};border-radius:6px 0 0 6px;'></div>"
            )
        dur_label = format_duration(durations[i]) if durations[i] is not None else ""
        title = html_lib.escape(card.title)
        subtitle = html_lib.escape(card.subtitle)
        cards_html.append(
            f"<div class='xray-event-card' data-index='{i}' onclick=\"{_card_onclick(i)}\" "
            f"style='position:relative;display:flex;flex-direction:column;justify-content:center;"
            f"min-height:{height}px;margin:4px 0;padding:6px 10px 6px 14px;border:{border};{shadow}"
            f"border-radius:6px;background:{card.color}0d;cursor:pointer;overflow:hidden;'>"
            f"{stripe}"
            f"<div style='display:flex;align-items:center;gap:6px;font-weight:600;font-size:13px;color:#1f2937;'>"
            f"<span style='font-size:15px;'>{card.icon}</span>"
            f"<span style='overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>{title}</span>"
            f"<span style='margin-left:auto;font-size:10px;color:#9ca3af;font-weight:400;'>#{i}{(' · ' + dur_label) if dur_label else ''}</span>"
            f"</div>"
            f"<div style='font-size:11px;color:#6b7280;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-top:2px;'>{subtitle}</div>"
            f"</div>"
        )

    # No scroll wrapper here — the overflow lives on the stable `#xray_rail`
    # Gradio container (see _CSS), so re-rendering these cards on selection does
    # not reset the scroll position.
    return "<div id='xray-event-rail'>" + "".join(cards_html) + "</div>"


# --- Grouped detail panes --------------------------------------------------


def render_group_chat_html(events: EpisodeEvents, group: EventGroup) -> str:
    """Chat pane: the group's LLM call rendered with the existing helper."""
    call = events.llm_call(group.llm_index)
    if call is None:
        return "<em>No LLM call in this group (e.g. the reset observation).</em>"
    return _render_llm_call_html(call)


def _message_text(msg: object) -> str:
    """Plain text of an LLM output message (str content, or text parts of a list)."""
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def render_group_reasoning_html(events: EpisodeEvents, group: EventGroup) -> str:
    """Reasoning pane: the LLM's thinking for the group, shown beside the action.

    Prefers extended-thinking `reasoning_content` (Anthropic exposes this);
    otherwise the assistant message text. When neither is present but the model
    consumed reasoning tokens, note that the provider hid the chain-of-thought
    (OpenAI/Azure return `reasoning_tokens` but not the text)."""
    call = events.llm_call(group.llm_index)
    if call is None:
        return "<em>No LLM call in this group.</em>"
    text = getattr(call.output, "reasoning_content", None) or _message_text(call.output)
    if text and text.strip():
        return (
            f"<div style='white-space:pre-wrap;font-size:13px;line-height:1.4;'>{html_lib.escape(text.strip())}</div>"
        )
    reasoning_tokens = getattr(call.usage, "reasoning_tokens", 0) or 0
    if reasoning_tokens > 0:
        return (
            f"<em>🔒 The model used {reasoning_tokens:,} reasoning token(s), but the provider "
            "did not return the reasoning text (hidden chain-of-thought).</em>"
        )
    return "<em>No reasoning text — the model emitted only tool call(s).</em>"


def render_group_observation_html(events: EpisodeEvents, group: EventGroup) -> tuple[list[Image.Image], str]:
    """Observation pane: screenshots + text contents for every observation in
    the group. Parallel siblings are stacked with per-observation labels.

    Returns `(images, html)` — the gallery images and the text/labels HTML.
    """
    images: list[Image.Image] = []
    blocks: list[str] = []
    obs_indices = group.observation_indices
    multiple = len(obs_indices) > 1
    for n, idx in enumerate(obs_indices):
        obs = events.observation(idx)
        action = events.action(idx)
        label = action.name if action is not None else f"observation #{idx}"
        if multiple:
            blocks.append(
                f"<h4 style='margin:8px 0 4px;'>🖥️ Sibling {n + 1}/{len(obs_indices)} — {html_lib.escape(label)}</h4>"
            )
        else:
            blocks.append(f"<h4 style='margin:8px 0 4px;'>🖥️ {html_lib.escape(label)}</h4>")
        img = screenshot_from_obs(obs)
        if img is not None:
            images.append(img)
            blocks.append(
                f"<div style='font-size:12px;color:#6b7280;'>screenshot {img.size[0]}×{img.size[1]} → see gallery</div>"
            )
        if obs is not None:
            for content in obs.contents:
                if isinstance(content.data, str) and content.data.strip():
                    name = html_lib.escape(content.name or "text")
                    blocks.append(_details_block(name, content.data))
    if not blocks:
        return [], "<em>No observation in this group.</em>"
    return images, "".join(blocks)


def render_group_evaluation_md(events: EpisodeEvents, group: EventGroup) -> str:
    """Evaluation pane: reward + info for every EvaluationEvent in the group."""
    if not group.evaluation_indices:
        return "*No evaluation recorded for this group.*"
    parts: list[str] = []
    for idx in group.evaluation_indices:
        out = events.output(idx)
        scope = "Terminal" if getattr(out, "is_terminal", False) else "Step-wise"
        reward = getattr(out, "reward", 0.0)
        info = getattr(out, "info", {}) or {}
        parts.append(f"### 🏁 {scope} evaluation\n\n**Reward:** {reward:g}\n")
        if info:
            parts.append("```json\n" + json.dumps(info, indent=2, default=str) + "\n```")
    return "\n".join(parts)


def render_group_error_md(events: EpisodeEvents, group: EventGroup) -> str:
    """Error pane: every error in the group (LLM, tool, or agent error)."""
    if not group.error_indices:
        return "No errors in this group."
    parts: list[str] = []
    for idx in group.error_indices:
        err = events.error(idx)
        if err is None:
            continue
        parts.append(
            f"### ⚠️ {err.error_type}\n**Message:** {err.exception_str}\n\n**Stack Trace:**\n```\n{err.stack_trace}\n```"
        )
    return "\n\n---\n\n".join(parts) if parts else "No errors in this group."


def render_group_debug_json(events: EpisodeEvents, group: EventGroup) -> str:
    """Debug pane: raw JSON dump of every event in the group.

    Dumps `event.output` with its concrete type (rather than the
    `TrajectoryEvent` wrapper, whose union `output` field triggers noisy
    Pydantic serializer warnings on every non-matching arm).
    """
    dump = []
    for idx in group.members:
        ev = events[idx]
        dump.append(
            {
                "index": idx,
                "kind": type(ev.output).__name__,
                "start_time": ev.start_time,
                "end_time": ev.end_time,
                "output": ev.output.model_dump(mode="json"),
            }
        )
    return json.dumps(dump, indent=2, default=str)


def render_group_action_html(events: EpisodeEvents, group: EventGroup) -> str:
    """Always-visible action panel: the action(s) the group's LLM dispatched."""
    actions = [events.action(i) for i in group.observation_indices]
    actions = [a for a in actions if a is not None]
    if not actions:
        return "<em>No action — observation-only or terminal group.</em>"
    lines: list[str] = []
    for action in actions:
        args = ", ".join(f"{k}={json.dumps(v, default=str)}" for k, v in (action.arguments or {}).items())
        lines.append(f"<code>{html_lib.escape(action.name)}({html_lib.escape(args)})</code>")
    return "<br>".join(lines)

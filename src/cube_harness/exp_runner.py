"""Run experiments with Ray or sequentially."""

import logging
import os
import signal
import socket
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import ray
from cube.resource import InfraConfig

from cube_harness.core import Trajectory
from cube_harness.episode import Episode
from cube_harness.episode_logs import LOG_FORMAT, get_log_path, redirect_output_to_log, trajectory_log_id
from cube_harness.episode_status import RETRIABLE_STATUSES, TERMINAL_STATUSES, EpisodeStatus, next_retry_count
from cube_harness.experiment import (
    DEFAULT_ORPHAN_THRESHOLD_S,
    Experiment,
    ExpResult,
    sweep_stale_statuses,
)
from cube_harness.experiment_status import EXPERIMENT_STATUS_FILENAME, ExperimentStatus
from cube_harness.metrics.tracer import get_trace_env_vars, get_tracer
from cube_harness.storage import FileStorage, Storage

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Wall-clock budget for ``infra.cleanup_stale()`` at lifecycle exit. The Azure
# implementation lists the resource group and parallel-deletes expired VMs;
# 30s covers a several-hundred-orphan sweep with headroom. If exceeded the
# lifecycle exits anyway — half-done cleanup is recovered by the next
# benchmark setup's L3 sweep.
_CLEANUP_GRACE_TIMEOUT_S: float = 30.0


# Default timeouts shared with xray_utils for ghost-episode detection.
DEFAULT_STEP_TIMEOUT_S: float = 9000.0
DEFAULT_SETUP_TIMEOUT_S: float = 4 * DEFAULT_STEP_TIMEOUT_S  # container scheduling can queue
DEFAULT_CANCEL_GRACE_S: float = 120.0

# How often the driver updates experiment_status.json in Ray mode.
_EXP_HEARTBEAT_INTERVAL_S: float = 30.0


def _warn_if_ephemeral_venv() -> None:
    """Warn when VIRTUAL_ENV is set but the interpreter lives elsewhere.

    This happens when a recipe is launched with bare `uv run recipe.py` while VIRTUAL_ENV
    already points to a project venv (e.g. inside a Claude worktree or after `source .venv/bin/activate`).
    uv ignores the mismatch and creates an ephemeral env in ~/.cache/uv/environments-v2/ whose .pth
    files can point to stale or deleted paths. Ray workers inherit that environment and may fail
    with ImportError on any editable-installed cube package.

    Fix: launch with `.venv/bin/python recipe.py`  or  `uv run --active recipe.py`.
    """
    venv = os.environ.get("VIRTUAL_ENV", "")
    if venv and not sys.executable.startswith(venv):
        logger.warning(
            "VIRTUAL_ENV=%s but the interpreter is %s — uv may have created an ephemeral "
            "environment whose .pth files point to stale paths. Ray workers will inherit "
            "this environment and may raise ImportError. "
            "Launch with:  .venv/bin/python <recipe>.py  or  uv run --active <recipe>.py",
            venv,
            sys.executable,
        )


def _extract_model(exp: Experiment) -> str | None:
    """Return the LLM model name from the agent config, if any (for tracing/labeling)."""
    llm_config = getattr(exp.agent_config, "llm_config", None)
    return llm_config.model_name if llm_config else None


def _trajectory_id(episode: Episode) -> str:
    """Build the canonical trajectory id `{task_id}_ep{episode_id}` for an Episode."""
    return trajectory_log_id(episode.config.task_config.task_id, episode.config.id)


@contextmanager
def _experiment_lifecycle(
    exp_dir: Path,
    mode: Literal["ray", "sequential"],
    infra: InfraConfig | None = None,
) -> Iterator[tuple[ExperimentStatus, Path]]:
    """Manage `experiment_status.json` from RUNNING through terminal write.

    Writes RUNNING on entry. On normal exit, writes COMPLETED; on exception,
    writes INTERRUPTED. Both initial and terminal writes log a warning on
    failure rather than swallowing silently — these bracket the run, so a
    failure here is something an operator should see.

    When ``infra`` is provided, also installs a SIGTERM handler that raises
    SystemExit so finally blocks run on orchestrator-driven shutdowns
    (``kubectl delete``, ``docker stop``, systemd unit stop), and sweeps stale
    cloud resources via ``infra.cleanup_stale()`` on lifecycle exit. This is
    best-effort: Ray captures signals delivered to the main thread (see TODO
    in ``_run_with_ray_impl``), so signal-driven cleanup mainly helps the
    sequential path. Hard kills (SIGKILL, OOM) cannot be intercepted — rely
    on the external scheduled sweeper for those.

    Ctrl+C (KeyboardInterrupt) still runs cleanup so Ray workers get their
    teardown window and stale infra resources get reclaimed — but cleanup
    is bounded by ``_CLEANUP_GRACE_TIMEOUT_S`` and pressing Ctrl+C a second
    time force-exits with a message. The daemon thread keeps running on
    timeout/force-exit but dies when the main thread exits; residue is
    recovered by the next benchmark setup's L3 sweep.
    """
    now = time.time()
    exp_status = ExperimentStatus(
        status="RUNNING",
        mode=mode,
        pid=os.getpid(),
        host=socket.gethostname(),
        started_at=now,
        last_heartbeat_at=now,
        total_episodes=0,
    )
    exp_status_path = exp_dir / EXPERIMENT_STATUS_FILENAME
    try:
        exp_status.write(exp_status_path)
    except Exception:
        logger.warning("Failed to write initial experiment status", exc_info=True)

    prev_sigterm = None
    if infra is not None:
        try:
            prev_sigterm = signal.signal(signal.SIGTERM, _raise_systemexit_on_sigterm)
        except ValueError:
            # signal.signal only works in the main thread — non-fatal if we're not there.
            logger.debug("Could not install SIGTERM handler (non-main thread); skipping")

    completed = False
    try:
        yield exp_status, exp_status_path
        completed = True
    finally:
        exp_status.status = "COMPLETED" if completed else "INTERRUPTED"
        exp_status.ended_at = time.time()
        exp_status.last_heartbeat_at = exp_status.ended_at
        try:
            exp_status.write(exp_status_path)
        except Exception:
            logger.warning("Failed to write terminal experiment status", exc_info=True)

        if prev_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, prev_sigterm)
            except ValueError:
                pass

        if infra is not None:
            _run_cleanup_with_grace(infra, timeout_s=_CLEANUP_GRACE_TIMEOUT_S)


def _raise_systemexit_on_sigterm(signum: int, frame: object) -> None:
    """Convert SIGTERM into SystemExit so the lifecycle's finally blocks run.

    Without this, Python's default SIGTERM behaviour is to terminate the process
    immediately, bypassing context managers and ``finally`` clauses — meaning a
    ``kubectl delete pod`` or ``docker stop`` would leak any in-flight cloud
    resources. Raising SystemExit lets the lifecycle exit gracefully and run
    ``cleanup_stale()`` on the way out.
    """
    logger.warning("Received SIGTERM — propagating as SystemExit for graceful cleanup")
    raise SystemExit(128 + signum)


def _run_cleanup_with_grace(infra: InfraConfig, timeout_s: float) -> Literal["OK", "ERROR", "TIMEOUT", "FORCED"]:
    """Run ``infra.cleanup_stale()`` with a timeout and Ctrl+C escalation.

    Cleanup runs in a daemon thread; the main thread polls for completion and
    drives the escalation. The first Ctrl+C is what got us here (caught by
    the enclosing context). A second Ctrl+C prints an escalation hint; a
    third sets a force-exit flag and stops polling. The daemon thread keeps
    running but dies when the process exits — half-done cleanup is recovered
    by the next benchmark setup's L3 sweep.

    Returns one of:
      - ``OK``      cleanup completed and reclaimed N (possibly zero) resources
      - ``ERROR``   ``cleanup_stale()`` raised; logged at WARNING, not propagated
      - ``TIMEOUT`` exceeded ``timeout_s`` seconds
      - ``FORCED``  user pressed Ctrl+C past the threshold
    """
    result: dict[str, Any] = {}

    def _worker() -> None:
        try:
            result["deleted"] = infra.cleanup_stale()
        except Exception as exc:
            result["error"] = exc

    thread = threading.Thread(target=_worker, daemon=True, name="cube-cleanup-stale")

    forced = threading.Event()
    presses = {"count": 0}

    def _on_sigint(signum: int, frame: object) -> None:
        presses["count"] += 1
        if presses["count"] == 1:
            print(
                "\n[cube] Cleanup in progress — letting Ray workers shut down and sweeping stale "
                "infra. Press Ctrl+C again to force exit (next launch will sweep any residue).",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                "\n[cube] Forcing exit. Cleanup incomplete — next launch will sweep.",
                file=sys.stderr,
                flush=True,
            )
            forced.set()

    prev_sigint = None
    try:
        prev_sigint = signal.signal(signal.SIGINT, _on_sigint)
    except ValueError:
        # signal.signal only works in the main thread; skip the escalation install otherwise.
        logger.debug("Could not install SIGINT handler (non-main thread); cleanup runs without escalation")

    thread.start()
    try:
        start = time.time()
        while thread.is_alive():
            thread.join(timeout=0.5)
            if forced.is_set():
                return "FORCED"
            if time.time() - start > timeout_s:
                logger.warning(
                    "Lifecycle exit: cleanup_stale exceeded %.0fs — exiting; next launch will sweep",
                    timeout_s,
                )
                return "TIMEOUT"
        if "error" in result:
            logger.warning("Lifecycle exit: cleanup_stale failed: %s", result["error"])
            return "ERROR"
        deleted = result.get("deleted", [])
        if deleted:
            logger.info("Lifecycle exit: cleanup_stale reclaimed %d resource(s)", len(deleted))
        return "OK"
    finally:
        if prev_sigint is not None:
            try:
                signal.signal(signal.SIGINT, prev_sigint)
            except ValueError:
                pass


def _pre_claim(storage: Storage, episode: Episode) -> None:
    """Write `QUEUED` for an episode before submitting it to Ray.

    `QUEUED` distinguishes "submitted to Ray, waiting for a worker" from
    `RUNNING` (a worker has actually picked it up and is heartbeating). The
    worker overwrites `QUEUED` → `RUNNING` when it enters `_open_status`.

    If the previous attempt left a terminal status, archive its directory first
    so the per-attempt history (including the terminal `status.json`) is preserved.
    """
    traj_id = _trajectory_id(episode)
    prior = storage.read_episode_status(traj_id)
    if prior is not None and prior.status in TERMINAL_STATUSES:
        storage.archive_episode(traj_id)
    storage.write_episode_status(
        traj_id,
        EpisodeStatus(
            status="QUEUED",
            task_id=episode.config.task_config.task_id,
            episode_id=episode.config.id,
            started_at=time.time(),
            last_heartbeat_at=None,
            current_step=0,
            retry_count=next_retry_count(prior),
        ),
    )


def run_with_ray(
    exp: Experiment,
    *,
    n_cpus: int = 4,
    ray_poll_timeout: float = 2.0,
    step_timeout_s: float = DEFAULT_STEP_TIMEOUT_S,
    setup_timeout_s: float = DEFAULT_SETUP_TIMEOUT_S,
    cancel_grace_s: float = DEFAULT_CANCEL_GRACE_S,
    orphan_threshold_s: float = DEFAULT_ORPHAN_THRESHOLD_S,
    max_retry_rounds: int = 3,
    otlp_endpoint: str | None = None,
    model: str | None = None,
    agent_name: str | None = None,
) -> ExpResult:
    """Run `exp` in parallel on Ray, with auto-retry and step-timeout enforcement."""
    model = model or _extract_model(exp)
    tracer = get_tracer(
        exp.name,
        otlp_endpoint=otlp_endpoint,
        model=model,
        agent_name=agent_name,
    )
    try:
        with (
            tracer.benchmark(exp.name),
            _experiment_lifecycle(exp.output_dir, mode="ray", infra=exp.infra) as (exp_status, exp_status_path),
        ):
            return _run_with_retries(
                exp,
                n_cpus=n_cpus,
                ray_poll_timeout=ray_poll_timeout,
                step_timeout_s=step_timeout_s,
                setup_timeout_s=setup_timeout_s,
                cancel_grace_s=cancel_grace_s,
                orphan_threshold_s=orphan_threshold_s,
                max_retry_rounds=max_retry_rounds,
                exp_status=exp_status,
                exp_status_path=exp_status_path,
            )
    finally:
        tracer.shutdown()


def _has_retriable_episodes(exp: Experiment) -> bool:
    """True iff any episode is FAILED/STALE/CANCELLED with retry_count below the cap."""
    statuses = FileStorage(exp.output_dir).list_episode_statuses()
    return any(s.status in RETRIABLE_STATUSES and s.retry_count < exp.max_retries for s in statuses.values())


def _run_with_retries(
    exp: Experiment,
    *,
    n_cpus: int,
    ray_poll_timeout: float,
    step_timeout_s: float,
    setup_timeout_s: float,
    cancel_grace_s: float,
    orphan_threshold_s: float,
    max_retry_rounds: int,
    exp_status: ExperimentStatus,
    exp_status_path: Path,
) -> ExpResult:
    """Run the experiment, then keep re-running retriable failures until none remain.

    Termination is guaranteed by the per-episode `max_retries` cap: each retried
    episode increments `retry_count`, so eventually every retriable episode either
    succeeds or hits the cap and `_has_retriable_episodes` returns False.
    """
    aggregated = ExpResult(tasks_num=0, config=exp.config, exp_id=f"{exp.name}_{uuid4().hex}")
    original_resume = exp.resume
    try:
        round_num = 0
        while True:
            round_result = _run_with_ray_impl(
                exp,
                n_cpus=n_cpus,
                ray_poll_timeout=ray_poll_timeout,
                step_timeout_s=step_timeout_s,
                setup_timeout_s=setup_timeout_s,
                cancel_grace_s=cancel_grace_s,
                orphan_threshold_s=orphan_threshold_s,
                exp_status=exp_status,
                exp_status_path=exp_status_path,
            )
            aggregated.tasks_num = max(aggregated.tasks_num, round_result.tasks_num)
            aggregated.trajectories.update(round_result.trajectories)
            for k, v in round_result.failures.items():
                aggregated.failures[k] = v
            for k in round_result.trajectories:
                aggregated.failures.pop(k, None)

            if not _has_retriable_episodes(exp):
                break

            if round_num >= max_retry_rounds:
                logger.info(
                    f"Stopping auto-retry after {round_num} round(s): reached max_retry_rounds={max_retry_rounds}"
                )
                break

            round_num += 1
            logger.info(f"Auto-retry round {round_num}: re-running retriable episodes")
            exp.resume = True
        return aggregated
    finally:
        exp.resume = original_resume


def _run_with_ray_impl(
    exp: Experiment,
    *,
    n_cpus: int,
    ray_poll_timeout: float,
    step_timeout_s: float,
    setup_timeout_s: float,
    cancel_grace_s: float,
    orphan_threshold_s: float,
    exp_status: ExperimentStatus,
    exp_status_path: Path,
) -> ExpResult:
    """Run a single Ray round: pre-claim, submit, poll with step-timeout, sweep stale on shutdown."""
    process_start_s = time.time()
    exp.save_config()
    output_dir = exp.output_dir
    storage = FileStorage(output_dir)

    @ray.remote
    def run_episode(episode: Episode) -> Trajectory:
        """Ray entry point: redirect logs to the episode's log file and run the episode."""
        traj_id = _trajectory_id(episode)
        log_file = get_log_path(output_dir, traj_id)
        with redirect_output_to_log(log_file, append=True, tee=False, log_format=LOG_FORMAT):
            return episode.run()

    if not ray.is_initialized():
        _warn_if_ephemeral_venv()
        ray.init(
            num_cpus=n_cpus,
            dashboard_host="0.0.0.0",
            include_dashboard=True,
            log_to_driver=True,
            runtime_env={"env_vars": get_trace_env_vars()},
        )  # TODO: Ray breaks signal handling, we cannot react to Ctrl+C here, still cannot find a workaround

    with exp.benchmark_config.make(exp.infra) as benchmark:
        try:
            episodes = exp.get_episodes_to_run(
                benchmark,
                step_timeout_s=step_timeout_s,
                cancel_grace_s=cancel_grace_s,
                orphan_threshold_s=orphan_threshold_s,
                process_start_s=process_start_s,
            )

            # Pre-claim every episode before any Ray submission so a concurrent runner
            # opening the same output_dir sees them as RUNNING (queued).
            for episode in episodes:
                _pre_claim(storage, episode)

            ref_to_traj_id: dict[ray.ObjectRef, str] = {}
            for episode in episodes:
                ref = run_episode.remote(episode)
                ref_to_traj_id[ref] = _trajectory_id(episode)
            exp_status.total_episodes = len(storage.list_episode_configs())
            exp_status.heartbeat(exp_status_path)
            logger.info(f"Start {len(episodes)} episodes in parallel using Ray with {n_cpus} workers")
            results = _poll_ray(
                exp,
                ref_to_traj_id,
                storage,
                ray_poll_timeout=ray_poll_timeout,
                step_timeout_s=step_timeout_s,
                setup_timeout_s=setup_timeout_s,
                cancel_grace_s=cancel_grace_s,
                orphan_threshold_s=orphan_threshold_s,
                exp_status=exp_status,
                exp_status_path=exp_status_path,
            )
            exp.print_stats(results)
            return results
        finally:
            ray.shutdown()
            # End-of-run STALE sweep: any RUNNING entries whose worker just got killed
            # by ray.shutdown() get marked STALE so the next round can retry them.
            sweep_stale_statuses(
                storage,
                step_timeout_s=step_timeout_s,
                cancel_grace_s=cancel_grace_s,
                orphan_threshold_s=orphan_threshold_s,
                process_start_s=process_start_s,
            )


def _poll_ray(
    exp: Experiment,
    ref_to_traj_id: dict[ray.ObjectRef, str],
    storage: FileStorage,
    *,
    ray_poll_timeout: float,
    step_timeout_s: float,
    setup_timeout_s: float,
    cancel_grace_s: float,
    orphan_threshold_s: float = DEFAULT_ORPHAN_THRESHOLD_S,
    exp_status: ExperimentStatus,
    exp_status_path: Path,
) -> ExpResult:
    """Wait on Ray refs, collecting trajectories/failures and force-killing workers with stale heartbeats."""
    results = ExpResult(tasks_num=len(ref_to_traj_id), config=exp.config, exp_id=f"{exp.name}_{uuid4().hex}")
    completed = 0
    episodes_in_progress = list(ref_to_traj_id.keys())
    last_exp_hb = time.time()
    # Timestamp since which Ray has *continuously* had a free CPU slot it isn't using
    # (reset whenever the cluster is fully busy). A QUEUED task is only an orphan if this
    # idle window persists — distinguishing "Ray can't schedule it (dead worker)" from
    # "it's waiting its turn behind busy workers."
    idle_capacity_since: float | None = None
    while len(episodes_in_progress) > 0:
        done, episodes_in_progress = ray.wait(
            episodes_in_progress,
            num_returns=len(episodes_in_progress),
            timeout=ray_poll_timeout,
        )
        completed += len(done)
        if len(done) > 0:
            logger.info(f"{completed} episodes completed, {len(episodes_in_progress)} in progress")
        for task_ref in done:
            traj_id = ref_to_traj_id[task_ref]
            try:
                traj: Trajectory = ray.get(task_ref, timeout=step_timeout_s + cancel_grace_s)
                # traj carries summary_stats but no steps (streamed to disk on the worker).
                _stats = traj.summary_stats or {}
                _n_agent, _n_env = _stats.get("n_agent_steps", 0), _stats.get("n_env_steps", 0)
                logger.info(
                    "Completed trajectory %s with %d steps (%d agent steps, %d environment steps)",
                    traj_id,
                    _n_agent + _n_env,
                    _n_agent,
                    _n_env,
                )
                results.trajectories[traj_id] = traj
            except Exception as e:
                logger.exception(f"Run failed with exception: {e}")
                results.failures[traj_id] = str(e)

        # Track sustained idle worker capacity from Ray's own accounting (robust to dead
        # workers — Ray keeps its logical CPU pool, so this reflects genuinely schedulable
        # slots, unlike counting RUNNING status files).
        has_idle_capacity = ray.available_resources().get("CPU", 0) >= 1
        idle_capacity_since = (idle_capacity_since or time.time()) if has_idle_capacity else None

        # Driver-side step timeout via filesystem read (replaces ray-dashboard list_tasks).
        _kill_stale_workers(
            episodes_in_progress,
            ref_to_traj_id,
            storage,
            results,
            step_timeout_s=step_timeout_s,
            setup_timeout_s=setup_timeout_s,
            cancel_grace_s=cancel_grace_s,
            orphan_threshold_s=orphan_threshold_s,
            idle_capacity_since=idle_capacity_since,
        )

        # Heartbeat experiment_status.json so XRay knows this driver is alive.
        now = time.time()
        if now - last_exp_hb >= _EXP_HEARTBEAT_INTERVAL_S:
            exp_status.heartbeat(
                exp_status_path,
                completed=len(results.trajectories),
                failed=len(results.failures),
            )
            last_exp_hb = now

    # Final heartbeat flushes counters before the lifecycle writes the terminal
    # status. Without this, the last batch of completions (or any work shorter
    # than `_EXP_HEARTBEAT_INTERVAL_S`) would be lost from the on-disk record.
    exp_status.heartbeat(
        exp_status_path,
        completed=len(results.trajectories),
        failed=len(results.failures),
    )
    return results


def _kill_stale_workers(
    episodes_in_progress: list[ray.ObjectRef],
    ref_to_traj_id: dict[ray.ObjectRef, str],
    storage: FileStorage,
    results: ExpResult,
    *,
    step_timeout_s: float,
    setup_timeout_s: float,
    cancel_grace_s: float,
    orphan_threshold_s: float = DEFAULT_ORPHAN_THRESHOLD_S,
    idle_capacity_since: float | None = None,
) -> None:
    """Read each active episode's status.json; force-kill stalled or orphaned workers.

    Two stall classes are handled so the poll loop can't hang on a ref forever:
    - **RUNNING** with a stale heartbeat — phase-aware budget: episodes still in setup
      (current_step == 0) get `setup_timeout_s`; episodes in the agent loop get the full
      `step_timeout_s`. Detects setup hangs (container boot, env reset) quickly without
      shortening the budget for legitimate long-running agent steps.
    - **QUEUED orphan** — detected by *idle worker capacity*, not elapsed time:
      `idle_capacity_since` is the timestamp (from the caller) since which Ray has had a
      free CPU slot it isn't using. Only when that idle window has persisted past
      `orphan_threshold_s` (not a momentary turnover dip) is a still-QUEUED task genuinely
      unschedulable (Ray could have scheduled it but didn't). A time-only bound
      (`now - started_at`) false-cancels tasks legitimately waiting their turn behind busy
      workers on a deep queue (see PR 459). `idle_capacity_since=None` ⇒ all slots busy ⇒
      QUEUED never cancelled.
    """
    now = time.time()
    to_remove: list[ray.ObjectRef] = []
    for ref in list(episodes_in_progress):
        traj_id = ref_to_traj_id[ref]
        status = storage.read_episode_status(traj_id)
        if status is None:
            continue
        # auto-fix(459)↓
        # QUEUED orphan: cancel only when Ray has had *idle* capacity it left unused,
        # sustained past orphan_threshold_s while this task waited — i.e. Ray could have
        # scheduled it but didn't. Gating on idle capacity (not elapsed time) is what stops
        # the deep-queue false-cancel the old time-only bound caused (reverted #445).
        if status.status != "RUNNING" or status.last_heartbeat_at is None:
            idle_for = (now - idle_capacity_since) if idle_capacity_since is not None else 0.0
            if (
                status.status == "QUEUED"
                and idle_capacity_since is not None
                and idle_for > orphan_threshold_s
                and now - status.started_at > orphan_threshold_s
            ):
                logger.error(
                    f"Episode {traj_id} stuck QUEUED for {now - status.started_at:.0f}s while Ray had idle "
                    f"capacity for {idle_for:.0f}s (>{orphan_threshold_s:.0f}s) — unschedulable; force killing"
                )
                try:
                    ray.cancel(ref, force=True)
                except Exception:
                    logger.exception(f"ray.cancel failed for {traj_id}")
                fresh = storage.read_episode_status(traj_id)
                if fresh is not None and fresh.status != "QUEUED":
                    to_remove.append(ref)
                    continue
                status.status = "CANCELLED"
                status.ended_at = now
                status.error_type = "OrphanedInQueue"
                status.error_message = f"unschedulable: QUEUED with idle worker capacity for >{orphan_threshold_s:.0f}s"
                storage.write_episode_status(traj_id, status)
                results.failures[traj_id] = status.error_message
                to_remove.append(ref)
            continue
        # /auto-fix(459)
        age = now - status.last_heartbeat_at
        budget = setup_timeout_s if status.current_step == 0 else step_timeout_s
        if age <= budget + cancel_grace_s:
            continue
        phase = "setup" if status.current_step == 0 else f"step {status.current_step}"
        logger.error(
            f"Episode {traj_id} {phase} stalled for {age:.0f}s (>{budget + cancel_grace_s:.0f}s) — force killing"
        )
        try:
            ray.cancel(ref, force=True)
        except Exception:
            logger.exception(f"ray.cancel failed for {traj_id}")

        # Re-read just before overwriting: between the staleness check above and
        # this point, the worker may have written a terminal status (legitimate
        # COMPLETED/FAILED). Don't clobber it — only stamp CANCELLED if we still
        # see RUNNING.
        fresh = storage.read_episode_status(traj_id)
        if fresh is not None and fresh.status != "RUNNING":
            to_remove.append(ref)
            continue
        status.status = "CANCELLED"
        status.ended_at = now
        status.error_type = "SetupTimeout" if status.current_step == 0 else "StepTimeout"
        status.error_message = f"{phase} exceeded {budget:.0f}s"
        storage.write_episode_status(traj_id, status)
        results.failures[traj_id] = status.error_message
        to_remove.append(ref)
    for ref in to_remove:
        episodes_in_progress.remove(ref)


def run_sequentially(
    exp: Experiment,
    debug_limit: int | None = None,
    *,
    step_timeout_s: float = DEFAULT_STEP_TIMEOUT_S,
    cancel_grace_s: float = DEFAULT_CANCEL_GRACE_S,
    orphan_threshold_s: float = DEFAULT_ORPHAN_THRESHOLD_S,
    max_retry_rounds: int = 3,
    otlp_endpoint: str | None = None,
    model: str | None = None,
    agent_name: str | None = None,
) -> ExpResult:
    """Run `exp` in-process (no Ray) with auto-retry — the debug-friendly path."""
    model = model or _extract_model(exp)
    tracer = get_tracer(
        exp.name,
        otlp_endpoint=otlp_endpoint,
        model=model,
        agent_name=agent_name,
    )
    try:
        with (
            tracer.benchmark(exp.name),
            _experiment_lifecycle(exp.output_dir, mode="sequential", infra=exp.infra) as (exp_status, exp_status_path),
        ):
            return _run_sequentially_with_retries(
                exp,
                debug_limit=debug_limit,
                step_timeout_s=step_timeout_s,
                cancel_grace_s=cancel_grace_s,
                orphan_threshold_s=orphan_threshold_s,
                max_retry_rounds=max_retry_rounds,
                exp_status=exp_status,
                exp_status_path=exp_status_path,
            )
    finally:
        tracer.shutdown()


def _run_sequentially_with_retries(
    exp: Experiment,
    *,
    debug_limit: int | None,
    step_timeout_s: float,
    cancel_grace_s: float,
    orphan_threshold_s: float,
    max_retry_rounds: int,
    exp_status: ExperimentStatus,
    exp_status_path: Path,
) -> ExpResult:
    """Run sequential rounds back-to-back until no retriable failures remain.

    Termination is guaranteed by the per-episode `max_retries` cap.
    """
    aggregated = ExpResult(tasks_num=0, config=exp.config, exp_id=f"{exp.name}_{uuid4().hex}")
    original_resume = exp.resume
    try:
        round_num = 0
        while True:
            round_result = _run_sequentially_impl(
                exp,
                debug_limit=debug_limit,
                step_timeout_s=step_timeout_s,
                cancel_grace_s=cancel_grace_s,
                orphan_threshold_s=orphan_threshold_s,
                exp_status=exp_status,
                exp_status_path=exp_status_path,
            )
            aggregated.tasks_num = max(aggregated.tasks_num, round_result.tasks_num)
            aggregated.trajectories.update(round_result.trajectories)
            for k, v in round_result.failures.items():
                aggregated.failures[k] = v
            for k in round_result.trajectories:
                aggregated.failures.pop(k, None)

            if not _has_retriable_episodes(exp):
                break

            if round_num >= max_retry_rounds:
                logger.info(
                    f"Stopping auto-retry after {round_num} round(s): reached max_retry_rounds={max_retry_rounds}"
                )
                break

            round_num += 1
            logger.info(f"Auto-retry round {round_num}: re-running retriable episodes")
            exp.resume = True
        return aggregated
    finally:
        exp.resume = original_resume


def _run_sequentially_impl(
    exp: Experiment,
    *,
    debug_limit: int | None,
    step_timeout_s: float,
    cancel_grace_s: float,
    orphan_threshold_s: float,
    exp_status: ExperimentStatus,
    exp_status_path: Path,
) -> ExpResult:
    """Run a single sequential round in-process, returning trajectories and failures for this round."""
    process_start_s = time.time()
    exp.save_config()
    storage = FileStorage(exp.output_dir)
    with exp.benchmark_config.make(exp.infra) as benchmark:
        all_episodes = exp.get_episodes_to_run(
            benchmark,
            step_timeout_s=step_timeout_s,
            cancel_grace_s=cancel_grace_s,
            orphan_threshold_s=orphan_threshold_s,
            process_start_s=process_start_s,
        )
        if debug_limit is not None:
            logger.info(f"Running only first {debug_limit} episodes for debugging")
        episodes = all_episodes[:debug_limit] if debug_limit is not None else all_episodes
        for episode in episodes:
            _pre_claim(storage, episode)

        exp_status.total_episodes = len(storage.list_episode_configs())
        exp_status.heartbeat(exp_status_path)

        results = ExpResult(
            tasks_num=len(episodes),
            config=exp.config,
            exp_id=f"{exp.name}_{uuid4().hex}",
        )
        for episode in episodes:
            traj_id = _trajectory_id(episode)
            log_file = get_log_path(exp.output_dir, traj_id)
            # Heartbeat before each episode.run() so XRay can detect a dead driver.
            exp_status.heartbeat(
                exp_status_path,
                completed=len(results.trajectories),
                failed=len(results.failures),
            )
            with redirect_output_to_log(log_file, append=False, tee=True, log_format=LOG_FORMAT):
                try:
                    trajectory = episode.run()
                    results.trajectories[traj_id] = trajectory
                except Exception as e:
                    logger.exception(f"Episode {traj_id} failed")
                    results.failures[traj_id] = str(e)
        # Flush final counters before the lifecycle's terminal write — the
        # in-loop heartbeat fires *before* each episode, so the last episode's
        # outcome is otherwise missing from the on-disk record.
        exp_status.heartbeat(
            exp_status_path,
            completed=len(results.trajectories),
            failed=len(results.failures),
        )
        exp.print_stats(results)
        return results


# === auto-fix notes ===  (spec: openspec/specs/auto-fix/spec.md)
# auto-fix-note(459) {class=L1 anchor=PR#459 hash=PENDING ctx=ray-runner/queued-orphan-capacity-gate/replaces-reverted-445/cube-harness@bcf746c2}
#   symptoms:  swebench-live lite-300 gold run, --n-parallel 20: 141/300 episodes
#              cancelled as OrphanedInQueue mid-run. Not orphaned — tasks >> workers +
#              slow heavy-repo evals meant the queue tail legitimately waited >1h
#              (orphan_threshold_s) behind busy workers and got killed (#445, now reverted).
#   invariant: a QUEUED episode is cancelled as an orphan ONLY when Ray has had idle
#              worker capacity it left unused (could have scheduled the task but didn't),
#              sustained past orphan_threshold_s. All slots busy ⇒ QUEUED never cancelled,
#              no matter how long it waits.
#   why:       #445 used a time-only bound (now - started_at > threshold) — can't tell
#              "Ray never scheduled this (dead worker)" from "waiting its turn." Gate on
#              ray.available_resources()['CPU'] (Ray's own schedulable-capacity accounting;
#              robust to dead workers, unlike counting RUNNING files), persisted across
#              polls (idle_capacity_since) to ignore turnover dips. Preserves the anti-hang
#              guarantee (genuine orphan with idle capacity still cancelled; RUNNING
#              stale-heartbeat path untouched). Replaces reverted auto-fix(445).
#   tested:    tests/test_experiment.py — QUEUED + all-slots-busy (idle_capacity_since None)
#              not cancelled even at 2h; QUEUED + sustained idle capacity cancelled;
#              momentary idle dip not cancelled.
#   hash=PENDING: stamped by scripts/auto_fix_lint.py (Tier-1) on first run.

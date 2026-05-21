"""Run experiments with Ray or sequentially."""

import logging
import os
import signal
import socket
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal
from uuid import uuid4

import ray
from cube.resource import InfraConfig

from cube_harness.core import Trajectory
from cube_harness.episode import Episode
from cube_harness.episode_logs import LOG_FORMAT, get_log_path, redirect_output_to_log, trajectory_log_id
from cube_harness.episode_status import RETRIABLE_STATUSES, TERMINAL_STATUSES, EpisodeStatus, next_retry_count
from cube_harness.experiment import Experiment, ExpResult, sweep_stale_statuses
from cube_harness.experiment_status import EXPERIMENT_STATUS_FILENAME, ExperimentStatus
from cube_harness.metrics.tracer import get_trace_env_vars, get_tracer
from cube_harness.storage import FileStorage, Storage

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

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

    Ctrl+C (KeyboardInterrupt) intentionally **skips** ``cleanup_stale()`` —
    the user wants out fast, and ``cleanup_stale`` makes a paginated ARM list
    call that can take seconds even with zero orphans. The next benchmark
    setup (L3) will sweep the residue.
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
    keyboard_interrupted = False
    try:
        yield exp_status, exp_status_path
        completed = True
    except KeyboardInterrupt:
        keyboard_interrupted = True
        raise
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

        if infra is not None and not keyboard_interrupted:
            try:
                deleted = infra.cleanup_stale()
                if deleted:
                    logger.info(
                        "Lifecycle exit: cleanup_stale() reclaimed %d expired resource(s)",
                        len(deleted),
                    )
            except Exception:
                logger.warning("Lifecycle exit: cleanup_stale() failed", exc_info=True)


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
    orphan_threshold_s: float = 3600.0,
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
    exp_status: ExperimentStatus,
    exp_status_path: Path,
) -> ExpResult:
    """Wait on Ray refs, collecting trajectories/failures and force-killing workers with stale heartbeats."""
    results = ExpResult(tasks_num=len(ref_to_traj_id), config=exp.config, exp_id=f"{exp.name}_{uuid4().hex}")
    completed = 0
    episodes_in_progress = list(ref_to_traj_id.keys())
    last_exp_hb = time.time()
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
                logger.info(
                    "Completed trajectory %s with %d steps (%d agent steps, %d environment steps)",
                    traj_id,
                    len(traj.steps),
                    traj.n_agent_steps,
                    traj.n_env_steps,
                )
                results.trajectories[traj_id] = traj
            except Exception as e:
                logger.exception(f"Run failed with exception: {e}")
                results.failures[traj_id] = str(e)

        # Driver-side step timeout via filesystem read (replaces ray-dashboard list_tasks).
        _kill_stale_workers(
            episodes_in_progress,
            ref_to_traj_id,
            storage,
            results,
            step_timeout_s=step_timeout_s,
            setup_timeout_s=setup_timeout_s,
            cancel_grace_s=cancel_grace_s,
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
) -> None:
    """Read each active episode's status.json; force-kill workers with stale heartbeats.

    Uses a phase-aware budget: episodes still in setup (current_step == 0) are given
    `setup_timeout_s`; episodes that have entered the agent loop get the full `step_timeout_s`.
    This lets setup hangs (container boot, env reset) be detected quickly without
    shortening the budget for legitimate long-running agent steps.
    """
    now = time.time()
    to_remove: list[ray.ObjectRef] = []
    for ref in list(episodes_in_progress):
        traj_id = ref_to_traj_id[ref]
        status = storage.read_episode_status(traj_id)
        # QUEUED → episode hasn't been picked up by a Ray worker yet (no heartbeat to check).
        # The orphan threshold path in the STALE sweep handles a never-picked-up QUEUED.
        if status is None or status.status != "RUNNING" or status.last_heartbeat_at is None:
            continue
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
    orphan_threshold_s: float = 3600.0,
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

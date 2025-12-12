"""Run experiments with Ray or sequentially."""

import logging
import os
import sys

import ray

from agentlab2.core import Trajectory
from agentlab2.experiment import Experiment, ExpResult
from agentlab2.run import AgentRun

LOG_FORMAT = "[%(levelname)s] %(asctime)s - %(name)s:%(lineno)d %(funcName)s() - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def run_with_ray(exp: Experiment, n_cpus: int = 4, ray_poll_timeout: float = 2.0) -> ExpResult:
    exp.save_config()
    ray_log_dir = os.path.join(exp.output_dir, "ray_logs")

    @ray.remote
    def run_single(agent_run: AgentRun) -> Trajectory:
        log_file = os.path.join(ray_log_dir, f"run_{agent_run.id}_task_{agent_run.task.id}.log")
        sys.stdout = open(log_file, "a", buffering=1)  # line-buffered
        sys.stderr = sys.stdout
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stdout, force=True)
        trajectory = agent_run.run()
        return trajectory

    if not ray.is_initialized():
        os.makedirs(ray_log_dir, exist_ok=True)
        ray.init(
            num_cpus=n_cpus,
            dashboard_host="0.0.0.0",
            include_dashboard=True,
            log_to_driver=True,
            runtime_env={"working_dir": None},
        )  # TODO: Ray breaks signal handling, we cannot react to Ctrl+C here, still cannot find a workaround

    exp.benchmark.setup()
    try:
        runs = exp.create_runs()
        ref_to_id = {run_single.remote(run): run.task.id for run in runs}
        logger.info(f"Start {len(runs)} runs in parallel using Ray with {n_cpus} workers")
        results = _poll_ray(exp, ref_to_id, ray_poll_timeout)
        exp.print_stats(results)
        return results
    finally:
        ray.shutdown()
        exp.benchmark.close()


def _poll_ray(exp: Experiment, ref_to_id: dict[ray.ObjectRef, str], ray_poll_timeout: float) -> ExpResult:
    results = ExpResult(tasks_num=len(ref_to_id), config=exp.config)
    completed = 0
    runs_in_progress = list(ref_to_id.keys())
    while len(runs_in_progress) > 0:
        done, runs_in_progress = ray.wait(runs_in_progress, num_returns=len(runs_in_progress), timeout=ray_poll_timeout)
        completed += len(done)
        if len(done) > 0:
            logger.info(f"{completed} runs completed, {len(runs_in_progress)} in progress")
        for task_ref in done:
            task_id = ref_to_id[task_ref]
            try:
                traj: Trajectory = ray.get(task_ref)
                logger.info(f"Completed trajectory for task {task_id} with {len(traj.steps)} steps")
                results.trajectories[task_id] = traj
            except Exception as e:
                logger.exception(f"Run failed with exception: {e}")
                results.failures[task_id] = str(e)
    return results


def run_sequentially(exp: Experiment, debug_limit: int | None = None) -> ExpResult:
    exp.save_config()
    exp.benchmark.setup()
    try:
        runs = exp.create_runs()
        if debug_limit is not None:
            logger.info(f"Running only first {debug_limit} runs")
            runs = runs[:debug_limit]
        trajectories = [run.run() for run in runs]
        results = ExpResult(
            tasks_num=len(runs),
            trajectories={traj.metadata["task_id"]: traj for traj in trajectories},
        )
        exp.print_stats(results)
        return results
    finally:
        exp.benchmark.close()

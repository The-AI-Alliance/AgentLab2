import asyncio
import logging
import time
from pathlib import Path
from typing import Self

from cube.benchmark import Benchmark, RuntimeContext
from cube.core import EnvironmentOutput, TypedBaseModel
from cube.resource import IncompatibleInfraError
from cube.task import TaskConfig
from cube.tool import Toolbox
from opentelemetry.trace import StatusCode
from termcolor import colored

from cube_harness.agent import AgentConfig
from cube_harness.core import AgentOutput, TrajectoryMetadata
from cube_harness.episode_logs import trajectory_log_id
from cube_harness.episode_status import TERMINAL_STATUSES, EpisodeStatus, next_retry_count
from cube_harness.eval_log import EpisodeRecord
from cube_harness.llm import is_permanent_llm_error
from cube_harness.metrics.tracer import get_tracer
from cube_harness.recorder import EventCounter, TurnRecorder
from cube_harness.storage import FileStorage, Storage, TrajectoryView
from cube_harness.summary import SummaryProcessor
from cube_harness.tool import Budget, BudgetExceeded, TaskDone, install_monitoring

logger = logging.getLogger(__name__)

MAX_STEPS = 1000  # System-wide upper limit on steps


class EpisodeConfig(TypedBaseModel):
    """Configuration for an episode that can be saved and reloaded."""

    id: int
    agent_config: AgentConfig
    exp_name: str
    output_dir: Path
    max_steps: int
    task_config: TaskConfig


class Episode:
    """Manages the execution of an agent on a specific task in an environment.

    RFC `agent-owns-loop` (Phase E): Episode no longer drives a per-turn
    loop. It builds the monitored toolbox + TurnRecorder, hands them to
    `agent.run(initial_obs, task, recorder)`, and finalizes regardless of
    how the agent returns or raises. The previous `_run_loop` is gone;
    every agent (legacy `step()` and new overridden `run()`) flows
    through the same Episode body.

    Public `run()` stays sync — callers (`exp_runner`, recipes, Ray
    workers) keep their existing signature. Internally `run()` wraps an
    `async _arun()` with `asyncio.run`.
    """

    def __init__(
        self,
        id: int,
        output_dir: Path,
        agent_config: AgentConfig,
        task_config: TaskConfig,
        exp_name: str,
        max_steps: int,
        storage: Storage | None,
        runtime_context: RuntimeContext | None,
    ) -> None:
        self.config = EpisodeConfig(
            id=id,
            agent_config=agent_config,
            exp_name=exp_name,
            output_dir=output_dir,
            max_steps=max_steps,
            task_config=task_config,
        )
        self._runtime_context = runtime_context
        self.storage = storage or FileStorage(output_dir)
        self.allow_overwrite = False

    @classmethod
    def load_episode_from_config(cls, config_path: Path, benchmark: Benchmark | None = None) -> Self:
        """Recreate an Episode from a persisted EpisodeConfig — used by
        the retry / resume path to rerun a previously-prepared episode."""
        # Unchanged — relies on EpisodeConfig.model_validate_json.
        with open(config_path) as f:
            episode_config = EpisodeConfig.model_validate_json(f.read())
        storage = FileStorage(episode_config.output_dir)
        runtime_context = benchmark._runtime_context if benchmark is not None else None
        return cls(
            id=episode_config.id,
            output_dir=episode_config.output_dir,
            agent_config=episode_config.agent_config,
            task_config=episode_config.task_config,
            exp_name=episode_config.exp_name,
            max_steps=episode_config.max_steps,
            storage=storage,
            runtime_context=runtime_context,
        )

    def run(self) -> TrajectoryView:
        """Sync entry point: drives the async loop via asyncio.run.

        Returns a lazy `TrajectoryView` onto the just-finalized episode dir.
        The view's metadata is loaded eagerly; events decode from disk
        on demand. Per the RFC `agent-owns-loop` scope expansion no
        full trajectory is held in memory at any point.
        """
        return asyncio.run(self._arun())

    def _open_status(self, trajectory_id: str) -> EpisodeStatus:
        """Initialise `status.json` for this attempt.

        If the prior status is terminal and this Episode opted in to overwrite
        (a legitimate retry), archive the prior directory so its terminal
        `status.json` survives. Without `allow_overwrite`, `save_metadata`
        will later raise — preserving the safety guard against accidental
        double-runs.
        """
        prior = self.storage.read_episode_status(trajectory_id)
        if prior is not None and prior.status in TERMINAL_STATUSES and self.allow_overwrite:
            ep_dir = self.storage._episode_dir(trajectory_id)
            if ep_dir.exists():
                self.storage._archive_episode(ep_dir)
        now = time.time()
        ep_status = EpisodeStatus(
            status="RUNNING",
            task_id=self.config.task_config.task_id,
            episode_id=self.config.id,
            started_at=now,
            last_heartbeat_at=now,
            current_step=0,
            retry_count=next_retry_count(prior),
        )
        self.storage.write_episode_status(trajectory_id, ep_status)
        return ep_status

    async def _arun(self) -> TrajectoryView:
        """Agent-owns-loop body. Sync `run()` wraps this with asyncio.run.

        Flow:
            1. setup (status, task, action_set, agent, trajectory, dirs).
            2. wrap task.toolbox with MonitoredTool (install_monitoring).
            3. build TurnRecorder bound to trajectory + storage + summary.
            4. record initial obs (recorder.record_reset).
            5. `await agent.run(initial.obs, task, recorder)` — the agent
               drives its own loop now.
            6. finalize:
               - terminal task.evaluate() → recorder.record_evaluation.
               - summary_stats + save_trajectory.
               - summary.on_episode_complete; EpisodeRecord.write.
               - task.close + tracer.shutdown.

        Exception handling:
            - BudgetExceeded: recorded as failure, episode marked
              MAX_STEPS_REACHED (analogous to today's max_steps exit).
            - Anything else (incl. agent-side crashes): recorded as
              failure, episode marked FAILED (or INVALID_CONFIG for
              permanent provider errors).
            - The `finally` block always runs evaluate + finalize.
        """
        task_id = self.config.task_config.task_id
        trajectory_id = trajectory_log_id(task_id, self.config.id)
        tracer = get_tracer(self.config.exp_name)

        # Heartbeat 1: covers stuck task creation / reset.
        ep_status = self._open_status(trajectory_id)
        meta: TrajectoryMetadata | None = None
        summary_proc: SummaryProcessor | None = None
        max_steps_reached = False

        try:
            with tracer.episode(task_id, experiment=self.config.exp_name) as episode_span:
                start_time = ep_status.started_at

                # 1. Build the live task and agent.
                task = self.config.task_config.make(runtime_context=self._runtime_context)
                action_set = task.action_set
                agent = self.config.agent_config.make(action_set, task_id=task_id)

                # 2. Reset the env to get the initial observation.
                obs, info = task.reset()
                initial = EnvironmentOutput(obs=obs, info=info)

                agent_name = self.config.agent_config.agent_name
                # WRITE-AT-START: persist TrajectoryMetadata with stub
                # summary fields and `end_time=None`. Makes crashed-
                # mid-run episodes loadable: the file exists on disk
                # and `TrajectoryView.is_complete` returns False until
                # finalize_episode writes the final fields below.
                meta = TrajectoryMetadata(
                    id=trajectory_id,
                    metadata={
                        "task_id": task_id,
                        "agent_name": agent_name,
                        "seed": getattr(self.config.task_config, "seed", None),
                        **initial.info,
                        "action_schemas": [a.as_dict() for a in action_set],
                    },
                    start_time=start_time,
                )
                self.storage.save_metadata(meta, allow_overwrite=self.allow_overwrite)
                ep_dir = self.storage._episode_dir(meta.id)
                (ep_dir / "episode_config.json").write_text(
                    self.config.model_dump_json(indent=2, serialize_as_any=True)
                )
                summary_proc = SummaryProcessor(ep_dir)

                # 3. Build budget + recorder + install monitoring on the
                # task's toolbox in place. Tool calls fired during the
                # run record their parent via the recorder's current
                # turn id. The shared EventCounter is what makes
                # recorder writes and monitored-tool writes land on a
                # single global event-numbering sequence on disk.
                budget = Budget(max_turns=self.config.max_steps)
                event_counter = EventCounter()
                metadata_updates: dict = {}
                recorder = TurnRecorder(
                    trajectory_id=trajectory_id,
                    storage=self.storage,
                    summary=summary_proc,
                    budget=budget,
                    event_counter=event_counter,
                    metadata_updates=metadata_updates,
                )
                install_monitoring(
                    task,
                    trajectory_id,
                    budget,
                    parent_event_id_getter=recorder.current_turn_id,
                    storage=self.storage,
                    summary=summary_proc,
                    event_counter=event_counter,
                )

                # 4. Compose the toolbox the agent will see: the task's
                # (now-monitored) tools + the agent's own non-monitored
                # tools (memory, scratchpad, …). From the agent's POV,
                # everything is just a tool — no `task` reference leaks.
                task_tool = getattr(task, "tool", None) or getattr(task, "toolbox", None)
                own_tools = [cfg.make() for cfg in self.config.agent_config.own_tool_configs]
                if isinstance(task_tool, Toolbox) and own_tools:
                    toolbox = Toolbox([*task_tool.tools, *own_tools])
                elif own_tools:
                    # Single task tool + own tools → wrap into a Toolbox.
                    toolbox = Toolbox([task_tool, *own_tools]) if task_tool is not None else Toolbox(own_tools)
                else:
                    # No own tools — pass the task's tool/toolbox through.
                    toolbox = task_tool

                # 5. Record the initial obs as a synthetic ToolCallEvent
                # whose parent is the RESET sentinel.
                recorder.record_reset(initial)
                logger.info(colored("Episode started — reset done", "blue"))

                # 6. Drive the agent. agent.run is the canonical entry.
                try:
                    await agent.run(initial.obs, toolbox, recorder)
                except BudgetExceeded as e:
                    logger.info(colored(f"Budget exceeded: {e}", "yellow"))
                    recorder.record_failure(e)
                    max_steps_reached = True
                except TaskDone:
                    # Clean episode end from the task side — agent emitted
                    # STOP_ACTION or task.finished() returned True. Not a
                    # failure; just proceed to finalization.
                    logger.info(colored("Task finished", "blue"))
                except Exception as e:
                    # Agent / env exceptions during the run. Permanent
                    # provider errors propagate after finalization so the
                    # runner stops the retry budget.
                    logger.exception(f"Error during agent.run: {e}")
                    recorder.record_failure(e)
                    raise

                # 7. Terminal evaluation. cube-standard's Task.evaluate
                # accepts obs=None — tasks track their own final state
                # internally (`self._latest_obs` set inside their own
                # `step()`). Errors propagate so callers see the real
                # exception; `finally` still finalizes the metadata.
                # is_terminal=True distinguishes this from any step-wise
                # EvaluationEvents emitted by MonitoredTool during the run.
                # If evaluate raises, record the failure as an AgentEvent
                # (so the trajectory carries the error) before re-raising —
                # the outer except below tags status and propagates to the
                # runner.
                try:
                    reward, info = task.evaluate()
                except Exception as e:
                    recorder.record_failure(e)
                    raise
                recorder.record_evaluation(reward, info, is_terminal=True)

                # Finalize: write the TrajectoryMetadata at episode end
                # with summary_stats + reward_info + end_time, then
                # update the experiment-level summary and emit the
                # eval record.
                end_time = time.time()
                final_metadata = {**meta.metadata, **metadata_updates}
                meta = meta.model_copy(
                    update={
                        "metadata": final_metadata,
                        "end_time": end_time,
                        "reward_info": {"reward": reward, "done": True, **info},
                        "summary_stats": summary_proc.summary_stats(
                            duration=end_time - start_time, final_reward=reward
                        ),
                    }
                )
                self.storage.finalize_episode(meta)
                summary_proc.on_episode_complete(meta, self.storage)
                try:
                    ep_record = EpisodeRecord.from_view(
                        self.storage.load_episode(meta.id),
                        evaluation_id=self.config.output_dir.name,
                        task_config=self.config.task_config,
                    )
                    ep_record.write(self.config.output_dir)
                except Exception:
                    logger.warning("Failed to write episode record", exc_info=True)

                logger.info(colored(f"Episode completed, reward: {reward}", "blue"))
                ep_status.reward = reward
                status = StatusCode.OK if reward > 0 else StatusCode.ERROR
                episode_span.set_status(status)

            ep_status.status = "MAX_STEPS_REACHED" if max_steps_reached else "COMPLETED"
        except Exception as e:
            logger.exception(f"Error during agent run: {e}")
            # Permanent provider errors (bad model name, bad key,
            # malformed request) and infra-incompatibility will fail
            # identically on retry — mark them terminal & non-retriable.
            permanent = is_permanent_llm_error(e) or isinstance(e, IncompatibleInfraError)
            ep_status.status = "INVALID_CONFIG" if permanent else "FAILED"
            ep_status.error_type = type(e).__name__
            ep_status.error_message = str(e)[:500]
            raise e
        finally:
            # Persist summary_stats on terminal failure paths too. With
            # it on the metadata stub, the XRay tables render correct
            # step/token/cost stats without loading any events.
            if meta is not None and summary_proc is not None and meta.summary_stats is None:
                try:
                    end = meta.end_time or time.time()
                    meta = meta.model_copy(
                        update={
                            "summary_stats": summary_proc.summary_stats(
                                duration=end - (meta.start_time or end),
                                final_reward=summary_proc.final_reward,
                            ),
                        }
                    )
                    self.storage.finalize_episode(meta)
                except Exception:
                    logger.exception("Failed to persist summary_stats on terminal path")
            ep_status.ended_at = time.time()
            ep_status.last_heartbeat_at = ep_status.ended_at
            try:
                self.storage.write_episode_status(trajectory_id, ep_status)
            except Exception:
                logger.exception("Failed to write final episode status")
            # task.close is best-effort; avoid masking the real exception.
            try:
                if "task" in locals():
                    task.close()
            except Exception:
                logger.exception("Failed to close task")
            tracer.shutdown()
        return self.storage.load_episode(trajectory_id)

    def log_agent_output(self, turns: int, agent_output: AgentOutput) -> None:
        """Legacy logger helper retained for any out-of-tree caller; the
        new flow logs via TurnRecorder + colored episode messages."""
        for llm_call in agent_output.llm_calls:
            if llm_call.output.content:
                logger.info(colored(f"Turn {turns} LLM Response: {llm_call.output.content}", "green"))
            if hasattr(llm_call.output, "reasoning_content") and llm_call.output.reasoning_content:
                logger.info(colored(f"Turn {turns} LLM Reasoning: {llm_call.output.reasoning_content}", "cyan"))
            if hasattr(llm_call.output, "thinking_blocks") and llm_call.output.thinking_blocks:
                for block in llm_call.output.thinking_blocks:
                    logger.info(colored(f"Turn {turns} LLM Thinking Block: {block}", "cyan"))
        actions_summary = [a.name for a in agent_output.actions] if agent_output.actions else []
        logger.info(colored(f"Turn {turns} Agent output: actions={actions_summary}", "magenta"))

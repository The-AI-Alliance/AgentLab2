"""GennyParallel — Genny with parallel tool-call dispatch.

Reference agent for the RFC `agent-owns-loop` Phase H.

Behaviour:
  - Same `step()` as Genny: builds a cache-friendly prompt, calls the
    LLM, parses one or more tool calls per assistant turn.
  - Overrides `Agent.run` (Phase D) to dispatch the N actions returned
    from one assistant turn as N concurrent tool calls — fans out via
    `asyncio.gather(*(task.toolbox.execute_action(a) for a in actions))`
    instead of the sequential `task.step(actions)` the default loop uses.

When this wins:
  - The LLM emits multiple INDEPENDENT tool calls in a single
    assistant message (e.g. "read file A and search for X in parallel").
    With sequential dispatch each tool call adds its latency to the
    turn; with parallel dispatch the turn pays max(latencies) instead
    of sum(latencies). On Daytona/SWE-bench tasks this is a 2-3× turn
    speed-up on tool-heavy turns.

When this does NOT win:
  - Tool calls that depend on each other's results (rare with current
    LLM tool-use patterns).
  - Cubes whose `task.step` does global bookkeeping that requires
    serial processing (e.g., enforced action ordering). For those,
    use plain Genny.

Compatibility:
  - Drop-in replacement for Genny on any cube. Set `LLMConfig.parallel_tool_calls=True`
    so the model is allowed to emit multiple tool calls per turn.
  - All monitoring + recorder + budget enforcement works unchanged —
    each parallel call still goes through `MonitoredTool.execute_action`,
    bumps the budget, and records its own `ToolCallEvent` with the
    same `turn_id` (the parent AgentEvent's id).
"""

import asyncio
import logging

from cube.core import Observation, StepError

from cube_harness.agents.genny import Genny, GennyConfig
from cube_harness.recorder import TurnRecorder

logger = logging.getLogger(__name__)


class GennyParallelConfig(GennyConfig):
    """Same as GennyConfig, but **forces** `llm_config.parallel_tool_calls=True`
    at `make()` time.

    Without this, GennyParallel silently degrades: `LLMConfig`'s default
    is `parallel_tool_calls=False`, so the LLM emits one tool call per
    turn and the `asyncio.gather` in `run()` fans out over a
    one-element list — same wall-clock as sequential dispatch.

    Caught by the agent-owns-loop reference baseline reproduction
    (gpt-5.4-mini on TerminalBench-2): the run reached parity with
    plain Genny on accuracy, but it never actually exercised parallel
    dispatch because nothing flipped the flag. The fix is to enforce
    it on the agent class that needs it, not to rely on the caller
    remembering to set it.
    """

    def make(self, action_set=None, task_id: str | None = None, **kwargs) -> "GennyParallel":
        """Instantiate GennyParallel; flip llm_config.parallel_tool_calls
        to True if the caller left it at the default False."""
        if not self.llm_config.parallel_tool_calls:
            logger.info(
                "GennyParallelConfig: forcing llm_config.parallel_tool_calls=True "
                "(was False — the LLM would otherwise emit one tool call per turn "
                "and the parallel dispatch would be a no-op)."
            )
            self.llm_config = self.llm_config.model_copy(update={"parallel_tool_calls": True})
        return GennyParallel(config=self, action_schemas=action_set or [], task_id=task_id)


class GennyParallel(Genny):
    """Genny with `asyncio.gather` tool-call dispatch.

    Inherits step() from Genny — only `run()` differs. Each AgentEvent
    spawns N concurrent `ToolCallEvent`s sharing the same `turn_id`,
    so XRay (Phase I) renders them as siblings of one turn.
    """

    async def run(self, initial_obs: Observation, toolbox, recorder: TurnRecorder) -> None:
        """Drive the agent loop with `asyncio.gather` parallel dispatch
        of the N actions returned per assistant turn.

        The toolbox is provided by Episode and contains the task's
        monitored tools + the agent's own (non-monitored) tools. The
        agent calls toolbox.execute_action(action) uniformly — no
        `task` reference; done/eval semantics are absorbed by the
        MonitoredTool wrappers."""
        obs = initial_obs
        while True:
            agent_output = await asyncio.to_thread(self.step, obs)
            recorder.record(agent_output)
            if not agent_output.actions and agent_output.error is None:
                return  # agent says "done"

            # Parallel fan-out. Each call goes through MonitoredTool's
            # execute_action (installed by Episode), which records its
            # own ToolCallEvent, enforces budget, and may raise
            # TaskDone / BudgetExceeded that propagates up through
            # asyncio.gather to Episode's outer except.
            results = await asyncio.gather(
                *(asyncio.to_thread(toolbox.execute_action, action) for action in agent_output.actions)
            )

            # Merge the parallel results into a single observation for
            # the next LLM turn. The Observation `+=` operator
            # concatenates content lists in order; we use the original
            # action order (asyncio.gather preserves it) so the LLM
            # sees results in a deterministic sequence.
            merged = self._merge_results(results)
            if merged is None:
                return  # all parallel calls errored — agent stops
            obs = merged

    @staticmethod
    def _merge_results(results: list[Observation | StepError]) -> Observation | None:
        """Combine N parallel tool results into one observation.

        StepError results are converted to a text observation so the
        LLM still sees the failure. Returns None when every result is
        an error and no useful observation was produced — the agent
        treats this as a graceful stop.
        """
        merged: Observation | None = None
        any_observation = False
        for r in results:
            if isinstance(r, Observation):
                any_observation = True
                if merged is None:
                    merged = r
                else:
                    merged += r
            else:
                # StepError: surface as text inside the merged obs so
                # the LLM can see what went wrong.
                msg = f"[tool error: {r.error_type}: {r.exception_str}]"
                if merged is None:
                    merged = Observation.from_text(msg)
                else:
                    merged += Observation.from_text(msg)
        if not any_observation and merged is None:
            return None
        return merged

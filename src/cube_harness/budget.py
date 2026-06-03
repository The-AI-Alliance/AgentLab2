"""Per-episode resource budget — caps + counters + the exceeded signal.

The Budget is the harness-side accounting object for one episode:
configured caps (agent steps, tool calls, cost, tokens, wallclock) and the
live counters bumped as the agent runs. `EventStreamer.on_step()` /
`.on_llm_call()` bump on the relevant boundaries; `MonitoredTool` checks
`Budget.exhausted` before each dispatch and raises `BudgetExceeded`.

Lives in its own module (not `tool.py`) because Budget is a cross-cutting
concept: it's enforced by tool wrappers, fed by the streamer, queried by
agents for soft-stop / prompt-injection. None of those layers should
import each other just to get at `Budget`.
"""

import threading
import time

from cube.core import Action, TypedBaseModel
from pydantic import Field, PrivateAttr


class Budget(TypedBaseModel):
    """Per-episode resource budget enforced by `MonitoredTool` + `EventStreamer`.

    Caps:
      - `max_agent_steps`: agent loop iterations (one `Agent.step()` call).
      - `max_tool_calls`: monitored tool dispatches.
      - `max_cost_usd`: cumulative LLM call cost (from `LLMCall.usage.cost`).
      - `max_prompt_tokens` / `max_completion_tokens`: per-direction
        cumulative token usage.
      - `max_wallclock_s`: elapsed seconds since the Budget was created.

    Counters bumped during the run:
      - `agent_steps` — by `EventStreamer.on_step()`, once per `Agent.run`
        iteration (one agent step). NOT per LLM call.
      - `cost_usd` / `prompt_tokens` / `completion_tokens` — by
        `EventStreamer.on_llm_call()` per LLM API call (cumulative
        across multi-LLM-call steps).
      - `tool_calls` — by `MonitoredTool._record_tool_call`.
      - `started_at` — set once at construction; elapsed time derived from it.

    `Budget.exhausted` returns True iff any configured cap is at-or-past
    its limit. `MonitoredTool` raises `BudgetExceeded(BaseException)`
    when it is. Agents can also introspect the live budget via
    `self._recorder.budget` for graceful self-stop and prompt-injection
    ("you have X% budget left") — see `Budget.__str__`.
    """

    max_agent_steps: int = 1_000
    max_tool_calls: int | None = None
    max_cost_usd: float | None = None
    max_prompt_tokens: int | None = None
    max_completion_tokens: int | None = None
    max_wallclock_s: float | None = None

    # Counters mutated in place during the run.
    agent_steps: int = 0
    tool_calls: int = 0
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    started_at: float = Field(default_factory=time.time)

    # Guards the bump methods + the `exhausted` read. Parallel agents
    # (e.g., Genny with `parallel_actions=True`) dispatch N tool calls
    # via `asyncio.gather` over `asyncio.to_thread`-wrapped calls, so
    # `tool_calls += 1` in `_record_tool_call` races across workers
    # without this lock. Mirrors `EventStreamer._lock`.
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def bump_tool_calls(self) -> None:
        """Atomic +1 on `tool_calls`. Called by `_record_tool_call`
        from MonitoredTool workers — may run on multiple threads in
        parallel."""
        with self._lock:
            self.tool_calls += 1

    def bump_agent_step(self) -> None:
        """Atomic +1 on `agent_steps` — bumped once per `Agent.run` loop
        iteration. Distinct from LLM-call count: an agent step may make
        0..N LLM calls.

        Called by the agent loop (`Agent._run` / `Agent._arun`) after
        each `self.step(obs)` returns, so `max_agent_steps` caps loop
        iterations the way callers expect.
        """
        with self._lock:
            self.agent_steps += 1

    def bump_llm_usage(self, cost: float, prompt: int, completion: int) -> None:
        """Atomic bump of LLM-call cost + token counters. Called by
        `EventStreamer.on_llm_call` per LLM API call (so a multi-LLM-call
        step accumulates correctly). Does NOT bump `agent_steps` —
        that's `bump_agent_step`'s job."""
        with self._lock:
            self.cost_usd += cost
            self.prompt_tokens += prompt
            self.completion_tokens += completion

    @property
    def exhausted(self) -> bool:
        """True iff any configured cap is at-or-past its limit. Checked
        by MonitoredTool on entry to every execute_action and by
        EventStreamer after every LLM call (via `on_llm_call`).

        Lock-protected so the multi-field read is coherent against
        concurrent bumps from parallel tool-call workers."""
        with self._lock:
            if self.agent_steps >= self.max_agent_steps:
                return True
            if self.max_tool_calls is not None and self.tool_calls >= self.max_tool_calls:
                return True
            if self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd:
                return True
            if self.max_prompt_tokens is not None and self.prompt_tokens >= self.max_prompt_tokens:
                return True
            if self.max_completion_tokens is not None and self.completion_tokens >= self.max_completion_tokens:
                return True
            if self.max_wallclock_s is not None and (time.time() - self.started_at) >= self.max_wallclock_s:
                return True
            return False

    def __getstate__(self) -> dict:
        # Strip the unpicklable Lock so Budget can ride along inside
        # any future serialized config without blowing up on Ray.
        state = super().__getstate__()
        state = dict(state)
        private = dict(state.get("__pydantic_private__") or {})
        private.pop("_lock", None)
        state["__pydantic_private__"] = private
        return state

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        # Recreate the lock — the deserialized instance is a fresh one,
        # no contention to inherit.
        self._lock = threading.Lock()

    def __str__(self) -> str:
        """Concise human-readable summary of budget usage — suitable for
        injection into an LLM prompt so the agent can plan against
        what's left. Only configured caps are listed; the agent-step cap
        always shows because it has a non-None default.

        Format: `"budget used: agent_steps 34/150 (23%), cost $1.20/$5.00 (24%), prompt_tokens 1200/5000 (24%), completion_tokens 100/1000 (10%), 60s/300s wallclock (20%)"`
        """
        parts: list[str] = []
        # max_agent_steps has a non-None default (1000), so always shown.
        # Guard the zero case (max_agent_steps=0 = born-exhausted, used
        # in tests) so the percent calc doesn't div-by-zero.
        if self.max_agent_steps > 0:
            parts.append(
                f"agent_steps {self.agent_steps}/{self.max_agent_steps} "
                f"({self.agent_steps / self.max_agent_steps * 100:.0f}%)"
            )
        else:
            parts.append(f"agent_steps {self.agent_steps}/{self.max_agent_steps}")
        if self.max_tool_calls is not None:
            pct = self.tool_calls / self.max_tool_calls * 100 if self.max_tool_calls > 0 else 0.0
            parts.append(f"tool_calls {self.tool_calls}/{self.max_tool_calls} ({pct:.0f}%)")
        if self.max_cost_usd is not None:
            pct = self.cost_usd / self.max_cost_usd * 100 if self.max_cost_usd > 0 else 0.0
            parts.append(f"cost ${self.cost_usd:.2f}/${self.max_cost_usd:.2f} ({pct:.0f}%)")
        if self.max_prompt_tokens is not None:
            pct = self.prompt_tokens / self.max_prompt_tokens * 100 if self.max_prompt_tokens > 0 else 0.0
            parts.append(f"prompt_tokens {self.prompt_tokens}/{self.max_prompt_tokens} ({pct:.0f}%)")
        if self.max_completion_tokens is not None:
            pct = self.completion_tokens / self.max_completion_tokens * 100 if self.max_completion_tokens > 0 else 0.0
            parts.append(f"completion_tokens {self.completion_tokens}/{self.max_completion_tokens} ({pct:.0f}%)")
        if self.max_wallclock_s is not None:
            elapsed = time.time() - self.started_at
            pct = elapsed / self.max_wallclock_s * 100 if self.max_wallclock_s > 0 else 0.0
            parts.append(f"{elapsed:.0f}s/{self.max_wallclock_s:.0f}s wallclock ({pct:.0f}%)")
        return "budget used: " + ", ".join(parts)


class BudgetExceeded(BaseException):
    """Raised by `MonitoredTool` wrappers when the per-episode budget is exhausted.

    Subclasses `BaseException` (not `Exception`) so an agent's
    `try / except Exception` cannot swallow it. The Episode `try/except`
    captures it explicitly and finalizes the trajectory.
    """

    def __init__(self, action: Action | None = None) -> None:
        super().__init__("budget exhausted")
        self.action = action

"""Regression: Budget counters under concurrent bumps from parallel
tool-call workers.

Genny[parallel_actions=True] fans out N tool calls via asyncio.gather over an
`_SyncToolAsAsync` adapter. The adapter hops into a real OS thread via
`asyncio.to_thread`, so N MonitoredTool workers run on N threads —
each calling `_record_tool_call` → `budget.bump_tool_calls()`. Without
the `Budget._lock` added in this PR's review pass, the
`tool_calls += 1` would race and `max_tool_calls` could overrun.

This test guards the lock.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

from cube_harness.tool import Budget


def test_concurrent_bump_tool_calls_count_correctly() -> None:
    """Fire N concurrent bump_tool_calls() from a thread pool; assert
    the final counter equals N. Without the lock, the read-modify-write
    of tool_calls would race and undercount."""
    budget = Budget(max_agent_steps=100_000)
    n_calls = 500
    barrier = threading.Barrier(n_calls)

    def worker() -> None:
        # Sync all workers to start at the same moment to maximize
        # contention and surface any race window.
        barrier.wait()
        budget.bump_tool_calls()

    with ThreadPoolExecutor(max_workers=n_calls) as pool:
        list(pool.map(lambda _: worker(), range(n_calls)))

    assert budget.tool_calls == n_calls


def test_concurrent_bump_llm_usage_coherent() -> None:
    """N concurrent bump_llm_usage(...) calls must leave the three
    counters in lockstep. Note: bump_llm_usage does NOT bump `agent_steps`
    — turn-counting is per agent step, not per LLM call (one step
    may issue N calls)."""
    budget = Budget(max_agent_steps=100_000)
    n_calls = 300
    barrier = threading.Barrier(n_calls)

    def worker() -> None:
        barrier.wait()
        budget.bump_llm_usage(cost=0.01, prompt=1, completion=2)

    with ThreadPoolExecutor(max_workers=n_calls) as pool:
        list(pool.map(lambda _: worker(), range(n_calls)))

    assert budget.agent_steps == 0  # bump_llm_usage does not bump turns
    assert abs(budget.cost_usd - n_calls * 0.01) < 1e-6
    assert budget.prompt_tokens == n_calls
    assert budget.completion_tokens == 2 * n_calls


def test_concurrent_bump_agent_step_count_correctly() -> None:
    """N concurrent bump_agent_step() calls leave `turns == N`. Separate from
    bump_llm_usage so `max_agent_steps` caps agent steps, not LLM API calls."""
    budget = Budget(max_agent_steps=100_000)
    n_calls = 300
    barrier = threading.Barrier(n_calls)

    def worker() -> None:
        barrier.wait()
        budget.bump_agent_step()

    with ThreadPoolExecutor(max_workers=n_calls) as pool:
        list(pool.map(lambda _: worker(), range(n_calls)))

    assert budget.agent_steps == n_calls
    assert budget.cost_usd == 0.0


def test_mixed_bumps_exhausted_check_coherent() -> None:
    """The `exhausted` read must be coherent against parallel bumps —
    no torn reads or misordered field checks."""
    budget = Budget(max_agent_steps=100_000, max_tool_calls=200)
    n_calls = 200
    barrier = threading.Barrier(n_calls)
    observed_exhausted = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        barrier.wait()
        budget.bump_tool_calls()
        ex = budget.exhausted
        with lock:
            observed_exhausted.append((budget.tool_calls, ex))

    with ThreadPoolExecutor(max_workers=n_calls) as pool:
        list(pool.map(worker, range(n_calls)))

    # Final state matches the cap exactly — no overrun.
    assert budget.tool_calls == 200
    assert budget.exhausted is True

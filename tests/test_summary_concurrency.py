"""Regression: EventStreamer's stats counters under concurrent emits
from parallel tool-call workers.

Genny[parallel_actions=True] dispatches N tool calls via asyncio.gather +
asyncio.to_thread. Each worker thread calls
MonitoredTool.execute_action → _record_tool_call → streamer.emit on
the SAME EventStreamer. Without the lock, the read-modify-write of
counters races: undercounted steps, dropped error_type.

Fixed by the threading.Lock guarding `_fold_stats` inside the
streamer. This test guards it.

NOTE: prior to the SummaryProcessor → EventStreamer fold, this file
tested the SummaryProcessor's jsonl writes. The jsonl was dropped
along with SummaryProcessor; what remains worth guarding is just the
counter coherence under parallel emit.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

from cube.core import Observation
from litellm import Message

from cube_harness.core import LLMCallEvent, ToolCallEvent, TrajectoryEvent
from cube_harness.llm import LLMCall, LLMConfig, Prompt, Usage
from cube_harness.streamer import EventStreamer


def _tool_call_event() -> TrajectoryEvent:
    return TrajectoryEvent(
        output=ToolCallEvent(
            parent_event_id="p",
            action_id="a-1",
            obs=Observation.from_text("ok"),
        ),
        start_time=0.0,
        end_time=0.0,
    )


def _agent_event() -> TrajectoryEvent:
    call = LLMCall(
        tag="act",
        llm_config=LLMConfig(model_name="openai/gpt-4o-mini"),
        prompt=Prompt(messages=[{"role": "user", "content": "hi"}]),
        output=Message(content="ok", role="assistant"),
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2, cost=0.0),
    )
    return TrajectoryEvent(
        output=LLMCallEvent(call=call),
        start_time=0.0,
        end_time=0.0,
    )


def test_concurrent_emit_counts_correctly() -> None:
    """Fire N emit calls from a thread pool, assert the counters add
    up to N. Without the lock, the read-modify-write of `_n_tool_calls`
    would race and undercount."""
    streamer = EventStreamer(trajectory_id="t")
    n_calls = 200
    barrier = threading.Barrier(n_calls)

    def worker() -> None:
        # Sync all workers to start at the same moment to maximize
        # contention.
        barrier.wait()
        streamer.emit(_tool_call_event())

    with ThreadPoolExecutor(max_workers=n_calls) as pool:
        list(pool.map(lambda _: worker(), range(n_calls)))

    assert streamer._n_tool_calls == n_calls


def test_concurrent_mixed_emit_keeps_counters_coherent() -> None:
    """Mixed LLM + tool emits must leave counters consistent — total
    events = sum of per-kind counts, no losses or double-counts."""
    streamer = EventStreamer(trajectory_id="t")
    n_calls = 300
    barrier = threading.Barrier(n_calls)

    def worker(i: int) -> None:
        barrier.wait()
        if i % 3 == 0:
            streamer.emit(_agent_event())
        else:
            streamer.emit(_tool_call_event())

    with ThreadPoolExecutor(max_workers=n_calls) as pool:
        list(pool.map(worker, range(n_calls)))

    n_llm = (n_calls + 2) // 3  # ceil(n_calls / 3) — i=0,3,6,…
    n_tool = n_calls - n_llm
    assert streamer._n_llm_calls == n_llm
    assert streamer._n_tool_calls == n_tool

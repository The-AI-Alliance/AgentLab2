"""Regression: SummaryProcessor under concurrent on_event from
parallel tool-call workers.

GennyParallel dispatches N tool calls via asyncio.gather +
asyncio.to_thread. Each worker thread calls MonitoredTool.execute_action
→ _record_tool_call → summary.on_event on the SAME SummaryProcessor.
Without a lock, the read-modify-write of counters races: undercounted
steps, dropped error_type, garbled jsonl rows.

Fixed by adding `threading.Lock` to SummaryProcessor in this PR's
review pass. This test guards it.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cube.core import Action, Observation

from cube_harness.core import AgentEvent, ToolCallEvent, TrajectoryEvent
from cube_harness.summary import SummaryProcessor


def _tool_call_event() -> TrajectoryEvent:
    return TrajectoryEvent(
        output=ToolCallEvent(
            parent_event_id="p",
            action_id="a-1",
            obs=Observation.from_text("ok"),
            turn_id="p",
        ),
        start_time=0.0,
        end_time=0.0,
    )


def _agent_event() -> TrajectoryEvent:
    return TrajectoryEvent(
        output=AgentEvent(actions=[Action(name="x", arguments={})]),
        start_time=0.0,
        end_time=0.0,
    )


def test_on_event_concurrent_writes_count_correctly(tmp_path: Path) -> None:
    """Fire N on_event calls from a thread pool, assert the counters
    add up to N. Without the lock, the read-modify-write of
    `_n_env_steps` would race and undercount."""
    sp = SummaryProcessor(tmp_path)
    n_calls = 200
    barrier = threading.Barrier(n_calls)

    def worker() -> None:
        # Sync all workers to start at the same moment to maximize
        # contention; without the lock the read-modify-write of
        # `_n_env_steps` would race.
        barrier.wait()
        sp.on_event(_tool_call_event())

    with ThreadPoolExecutor(max_workers=n_calls) as pool:
        list(pool.map(lambda _: worker(), range(n_calls)))

    assert sp._n_env_steps == n_calls


def test_on_event_concurrent_jsonl_lines_well_formed(tmp_path: Path) -> None:
    """The episode_summary.jsonl file must contain one valid JSON line
    per event — no interleaved / truncated rows from concurrent
    writes."""
    sp = SummaryProcessor(tmp_path)
    n_calls = 200
    barrier = threading.Barrier(n_calls)

    def worker(i: int) -> None:
        barrier.wait()
        # Mix kinds so the workers race on different code paths inside
        # on_event (agent vs tool_call vs cost-bumping branches).
        if i % 3 == 0:
            sp.on_event(_agent_event())
        else:
            sp.on_event(_tool_call_event())

    with ThreadPoolExecutor(max_workers=n_calls) as pool:
        list(pool.map(worker, range(n_calls)))

    lines = (tmp_path / "episode_summary.jsonl").read_text().splitlines()
    assert len(lines) == n_calls
    for line in lines:
        # Each line is parseable JSON — no interleaved writes.
        json.loads(line)

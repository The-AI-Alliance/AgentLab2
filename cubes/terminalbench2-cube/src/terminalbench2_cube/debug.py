"""Deterministic debug agent for testing terminalbench2-cube end-to-end without an LLM.

Public API
----------
get_debug_benchmark()         → TerminalBench2BenchmarkConfig
make_debug_agent(task_id)     → DebugAgent
"""

from __future__ import annotations

import logging
from typing import Annotated

import typer
from cube.core import Action, ActionSchema, Observation
from terminalbench2_cube.benchmark import TerminalBench2BenchmarkConfig

logger = logging.getLogger(__name__)

# Each debug task runs in oracle_mode: the ground-truth solution is uploaded
# to /solution in the container during reset(). The debug agent applies it
# and calls final_step, which triggers evaluate() → pytest → reward == 1.0.
_FINAL = Action(name="final_step", arguments={})

_TASK_ACTIONS: dict[str, list[Action]] = {
    # Relative `personal-site` works whether working_dir is /app (default) or
    # /tmp/app (after _maybe_relocate_app on non-root backends).
    # /tmp/solution is where TerminalBench2Task.reset() uploads the solution
    # for any backend — /tmp is universally writable.
    "fix-git": [
        Action(
            name="bash",
            arguments={"command": "cd personal-site && bash /tmp/solution/solve.sh 2>&1", "timeout": 600},
        ),
        _FINAL,
    ],
    "overfull-hbox": [
        # Prepend the apt-extracted python3 to PATH so solve.sh's `python3` call
        # works on minimal images (e.g. bare LaTeX) that ship without python3.
        # On images that already have python3 in PATH the prepended directory
        # either doesn't exist (harmless) or provides a compatible python3.12.
        Action(
            name="bash",
            arguments={
                "command": (
                    "export PATH=/tmp/python3_pkg/usr/bin:$PATH && "
                    "export LD_LIBRARY_PATH=/tmp/python3_pkg/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH && "
                    "bash /tmp/solution/solve.sh 2>&1"
                ),
                "timeout": 600,
            },
        ),
        _FINAL,
    ],
}


class DebugAgent:
    """Deterministic agent that replays a fixed action sequence."""

    def __init__(self, task_id: str) -> None:
        if task_id not in _TASK_ACTIONS:
            raise ValueError(f"No debug actions for {task_id!r}. Known: {list(_TASK_ACTIONS)}")
        self._task_id = task_id
        self._step = 0
        self._actions = list(_TASK_ACTIONS[task_id])

    def get_action(self, obs: Observation) -> Action:
        if self._step >= len(self._actions):
            raise StopIteration(f"All actions exhausted for task {self._task_id!r}")
        action = self._actions[self._step]
        self._step += 1
        return action

    def __call__(self, obs: Observation, action_set: list[ActionSchema]) -> Action:
        return self.get_action(obs)


def get_debug_benchmark() -> TerminalBench2BenchmarkConfig:
    """Return a ``TerminalBench2BenchmarkConfig`` scoped to the debug tasks.

    Pure factory — the harness owns ``config.install()`` and ``config.make(infra)``.
    Debug tasks run in ``oracle_mode`` so reset() uploads the gold solution.
    """
    return TerminalBench2BenchmarkConfig(oracle_mode=True).subset_from_list(list(_TASK_ACTIONS))


def make_debug_agent(task_id: str) -> DebugAgent:
    return DebugAgent(task_id)


def _cli(
    toolkit: Annotated[bool, typer.Option(help="Use EAI Toolkit infra instead of local Docker")] = False,
    eai_profile: Annotated[str, typer.Option(help="EAI profile")] = "yul101",
    eai_path: Annotated[str, typer.Option(help="Path to eai CLI")] = "eai",
    preemptable: Annotated[bool, typer.Option(help="Request preemptable resources")] = False,
) -> None:
    """Run the terminalbench2-cube oracle debug suite."""
    import terminalbench2_cube.debug as _this_module
    from cube.testing import run_debug_suite

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

    infra = None
    if toolkit:
        from cube_infra_toolkit import ToolkitInfraConfig

        # cube_data defaults to "auto" — ToolkitInfraConfig auto-provisions the
        # sidecar + uv bundle at /opt/cube/ on first launch, no flags needed.
        infra = ToolkitInfraConfig(
            profile=eai_profile,
            eai_path=eai_path,
            preemptable=preemptable,
            launch_timeout_seconds=3000,
        )

    results = run_debug_suite("terminalbench2-cube", _this_module, workers=1, infra=infra)
    failed = [r for r in results if r["error"] or not r["done"] or r["reward"] < 1.0]
    raise typer.Exit(1 if failed else 0)


if __name__ == "__main__":
    typer.run(_cli)

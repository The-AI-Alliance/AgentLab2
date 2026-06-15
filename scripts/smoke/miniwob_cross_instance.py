"""SMOKE: MiniWob cross-instance seeding — a different seed renders a different instance.

Validates the cross-instance eval mechanism end-to-end without an LLM or GPU: the
``MiniWobBenchmarkConfig.seed`` field threads into the task, ``Math.seedrandom(seed)``
renders a seed-specific instance, and the same seed is deterministic. This is the
property the cross-instance eval (``jefhinter_miniwob.py --cross-instance``) relies on.

Stands up a real MiniWob http server (free port) + headless Chromium, so run it from
the eval venv with Playwright installed:

    .venv/bin/python scripts/smoke/miniwob_cross_instance.py

Prints ``SMOKE OK/FAIL/SKIP: miniwob_cross_instance`` (exit 0/1/2).
"""

from __future__ import annotations

import sys

from cube.core import TextContent
from miniwob_cube import MINIWOB_CONFIGS

NAME = "miniwob_cross_instance"
# simple-arithmetic's instance (the arithmetic expression) is seeded, so its rendered
# observation changes with the seed — a clean signal that the seed actually took effect.
TASK_ID = "simple-arithmetic"


def _obs_text(obs: object) -> str:
    return "\n".join(c.data for c in getattr(obs, "contents", []) if isinstance(c, TextContent))


def _instance_signature(seed: int) -> str:
    """Reset TASK_ID at `seed` on a fresh free-port benchmark; return goal + obs text."""
    cfg = MINIWOB_CONFIGS["default"].model_copy(update={"seed": seed, "port": 0}).subset_from_list([TASK_ID])
    bench = cfg.make()
    with bench:
        task_config = next(cfg.get_task_configs())
        assert task_config.seed == seed, f"benchmark seed {seed} did not reach task config (got {task_config.seed})"
        task = bench.spawn(task_config)
        try:
            obs, info = task.reset()
            return f"goal={info.get('goal', '')!r}\n{_obs_text(obs)}"
        finally:
            task.close()


def main() -> int:
    try:
        first_42 = _instance_signature(42)
        second_42 = _instance_signature(42)
        other = _instance_signature(9001)
    except Exception as exc:  # browser/server not available in this env -> SKIP, not FAIL
        print(f"SMOKE SKIP: {NAME} — could not stand up MiniWob ({type(exc).__name__}: {exc})")
        return 2

    ok = True
    if first_42 != second_42:
        print("  FAIL: same seed (42) rendered two different instances -> non-deterministic")
        ok = False
    if first_42 == other:
        print("  FAIL: seeds 42 and 9001 rendered the SAME instance -> seed not applied")
        ok = False

    print(f"  seed=42   {first_42.splitlines()[0]}")
    print(f"  seed=9001 {other.splitlines()[0]}")
    if ok:
        print(f"SMOKE OK: {NAME} — seed 42 deterministic; 9001 differs (cross-instance seeding works)")
        return 0
    print(f"SMOKE FAIL: {NAME}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

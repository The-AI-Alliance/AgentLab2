"""Gold-patch oracle baseline — SWE-Gym Lite.

Applies each task's gold patch (``oracle_mode`` writes it to
``/tmp/gold_patch.diff``) and calls ``final_step``, with no LLM. Use it to
sanity-check the eval pipeline and see which tasks the environment resolves —
across all 230 tasks, not just the 2-task debug suite.

This file IS the config: edit ``bench`` for a task subset; ``run()`` provides
the generic CLI (``--experiment`` picks infra, ``--ray`` / ``--limit`` control
execution). After a run, list resolved tasks with
``gold_patch.extract_solvable(run_dir)``.

    # All 230 tasks on local Docker (default), 8 Ray workers:
    python -m swegym_lite_cube.gold_patch.recipe

    # On EAI Toolkit with 50 workers:
    python -m swegym_lite_cube.gold_patch.recipe --experiment toolkit --ray 50

    # Quick smoke — first 3 tasks, in-process:
    python -m swegym_lite_cube.gold_patch.recipe --experiment toolkit --limit 3

Requires cube-harness (not a swegym-lite-cube runtime dependency).
"""

from swegym_lite_cube.benchmark import SWEGymLiteBenchmarkConfig
from swegym_lite_cube.gold_patch.agent import GoldPatchAgentConfig

from cube_harness.experiment import Experiment
from cube_harness.infra import INFRA_CONFIGS
from cube_harness.recipe import run

# Edit for a subset, e.g. .subset_from_list(["getmoto__moto-5699", ...]).
bench = SWEGymLiteBenchmarkConfig(oracle_mode=True)


def _exp(infra: str) -> Experiment:
    return Experiment(
        name=f"gold-patch-swegym-lite-{infra}",
        agent_config=GoldPatchAgentConfig(),
        benchmark_config=bench,
        infra=INFRA_CONFIGS[infra],
        max_steps=5,
    )


if __name__ == "__main__":
    # "default" (local) is always available; cloud infras only if configured
    # in ~/.cube/infra.py. Pick one with --experiment.
    candidates = ("local", "toolkit", "daytona")
    run({("default" if name == "local" else name): _exp(name) for name in candidates if name in INFRA_CONFIGS})

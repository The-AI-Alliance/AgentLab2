"""Gold-patch oracle baseline — SWE-bench Live.

Applies each task's gold patch (``oracle_mode`` writes it to ``/tmp/gold_patch.diff``)
and calls ``final_step``, with no LLM. Use it to sanity-check the eval pipeline and
see which tasks the environment resolves — and on which infra. SWE-bench-Live images
assume ``USER root``, so the non-root EAI Toolkit scores 0 on tasks needing root;
Daytona (root) resolves them.

This file IS the config: edit ``bench`` for a different subset; ``run()`` provides the
generic CLI (``--experiment`` picks infra, ``--ray`` / ``--limit`` control execution).
After a run, list the resolved tasks post-hoc with ``solvable.extract_solvable(run_dir)``
(one run) or ``solvable.intersect_solvable(run_dirs)`` (N runs → stable/flaky) — that's
how the shipped ``lite_solvable*.json`` snapshots are regenerated.

    # lite (300 tasks) on Daytona (root, the default), 20 Ray workers:
    python -m swebench_live_cube.gold_patch.recipe --ray 20

    # quick smoke — first 3 tasks, in-process, local Docker:
    python -m swebench_live_cube.gold_patch.recipe --experiment local --limit 3

(SWE-bench-Live images are large; if Daytona launches time out, raise
``launch_timeout_seconds`` on the ``DaytonaInfraConfig`` in ``~/.cube/infra.py``.)

Requires cube-harness (not a swebench-live-cube runtime dependency).
"""

from __future__ import annotations

from swebench_live_cube.benchmark import SWEBenchLiveBenchmarkConfig
from swebench_live_cube.gold_patch.agent import GoldPatchAgentConfig

from cube_harness.experiment import Experiment
from cube_harness.infra import INFRA_CONFIGS
from cube_harness.recipe import run

# 30 diverse tasks confirmed gold-solvable, one per repo — handy for rapid iteration.
# Select with: bench = SWEBenchLiveBenchmarkConfig(oracle_mode=True).subset_from_list(sorted(_LIVE_GOLDEN_30))
_LIVE_GOLDEN_30: frozenset[str] = frozenset(
    [
        "aws-cloudformation__cfn-lint-3749",
        "beeware__briefcase-2075",
        "conan-io__conan-17102",
        "cyclotruc__gitingest-115",
        "dynaconf__dynaconf-1225",
        "flexget__flexget-4244",
        "huggingface__smolagents-285",
        "instructlab__instructlab-2526",
        "ipython__ipython-14695",
        "joke2k__faker-2142",
        "kedro-org__kedro-4387",
        "keras-team__keras-20396",
        "kozea__weasyprint-2300",
        "mikedh__trimesh-2354",
        "pdm-project__pdm-3237",
        "projectmesa__mesa-2394",
        "pvlib__pvlib-python-2249",
        "pydata__xarray-9586",
        "pylint-dev__pylint-10044",
        "pypsa__pypsa-1091",
        "python-babel__babel-1141",
        "python-control__python-control-1064",
        "pytorch__torchtune-1806",
        "shapely__shapely-2224",
        "sissbruecker__linkding-971",
        "sphinx-doc__sphinx-12975",
        "stanfordnlp__dspy-1609",
        "streamlink__streamlink-6242",
        "wireservice__csvkit-1274",
        "yt-dlp__yt-dlp-11425",
    ]
)

# Edit for a different subset, e.g. .named_subset("full") (all 1895) or
# .subset_from_list(sorted(_LIVE_GOLDEN_30)). Defaults to the lite (300) split.
bench = SWEBenchLiveBenchmarkConfig(oracle_mode=True).named_subset("lite")


def _exp(infra: str) -> Experiment:
    return Experiment(
        name=f"gold-patch-live-{infra}",
        agent_config=GoldPatchAgentConfig(),
        benchmark_config=bench,
        infra=INFRA_CONFIGS[infra],
        max_steps=5,
    )


if __name__ == "__main__":
    # SWE-bench-Live images assume USER root, so Daytona (root) is the default.
    # local Docker / EAI Toolkit stay selectable via --experiment for smokes and the
    # root-vs-non-root comparison. Each infra needs a ~/.cube/infra.py entry.
    by_key = {"default": "daytona", "local": "local", "toolkit": "toolkit"}
    run({key: _exp(infra) for key, infra in by_key.items() if infra in INFRA_CONFIGS})

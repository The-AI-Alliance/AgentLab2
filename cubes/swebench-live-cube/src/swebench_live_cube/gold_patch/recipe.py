"""Gold-patch baseline recipe — SWE-bench Live.

Runs the oracle GoldPatchAgent against a subset, identifies which tasks are
solvable (reward == 1.0), and optionally detects flaky tasks by running N times.

Outputs:
  --dump-solvable path.json          single-run solvable task IDs
  --dump-solvable path.json --n-runs 3   stable (all runs) + path_flaky.json

Requires cube-harness (not a swebench-live-cube runtime dependency).
Install with: pip install swebench-live-cube[eval]

Usage:
    # Validate pipeline on 30-task sample (no LLM, ~5 min on Toolkit):
    .venv/bin/python -m swebench_live_cube.gold_patch.recipe --subset live30 \\
        --toolkit --eai-profile yul101 --eai-path ~/bin/eai

    # Identify solvable subset across full lite (300 tasks), 3 runs for stability:
    .venv/bin/python -m swebench_live_cube.gold_patch.recipe --subset lite \\
        --toolkit --eai-profile yul101 --eai-path ~/bin/eai \\
        --n-parallel 50 --n-runs 3 --dump-solvable solvable_lite_stable.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment, ExpResult
from cube_harness.storage import FileStorage

from swebench_live_cube.gold_patch.agent import GoldPatchAgentConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Solvable-subset helpers
# ---------------------------------------------------------------------------


def extract_solvable(run_dir: Path) -> list[str]:
    """Task IDs that resolved (reward == 1.0) in a completed gold run."""
    trajs = FileStorage(run_dir).load_all_trajectory_metadata()
    return sorted(t.id.rsplit("_ep", 1)[0] for t in trajs if t.reward_info.get("reward") == 1.0)


def intersect_solvable(run_dirs: list[Path]) -> tuple[list[str], list[str]]:
    """Return (stable, flaky) across N runs.

    stable — solved in ALL runs (safe eval subset).
    flaky  — solved in SOME runs (environment instability).
    """
    sets = [set(extract_solvable(d)) for d in run_dirs]
    stable = sorted(set.intersection(*sets))
    flaky = sorted(set.union(*sets) - set(stable))
    return stable, flaky


# ---------------------------------------------------------------------------
# Benchmark / infra helpers
# ---------------------------------------------------------------------------

_LIVE_30_SAMPLE: frozenset[str] = frozenset(
    [
        "conan-io__conan-17300",
        "aws-cloudformation__cfn-lint-3768",
        "deepset-ai__haystack-8609",
        "matplotlib__matplotlib-29258",
        "reflex-dev__reflex-4717",
        "instructlab__instructlab-3118",
        "run-llama__llama_deploy-458",
        "pydata__xarray-9636",
        "pvlib__pvlib-python-2400",
        "streamlink__streamlink-6242",
        "keras-team__keras-20443",
        "pdm-project__pdm-3250",
        "sphinx-doc__sphinx-12975",
        "python-babel__babel-1141",
        "projectmesa__mesa-2394",
        "pylint-dev__pylint-10240",
        "kozea__weasyprint-2300",
        "joke2k__faker-2142",
        "wemake-services__wemake-python-styleguide-3114",
        "privacyidea__privacyidea-4223",
        "sissbruecker__linkding-989",
        "kedro-org__kedro-4387",
        "stanfordnlp__dspy-1609",
        "yt-dlp__yt-dlp-11880",
        "pypsa__pypsa-1195",
        "beeware__briefcase-2214",
        "huggingface__smolagents-843",
        "dynaconf__dynaconf-1238",
        "ipython__ipython-14822",
        "jupyterlab__jupyter-ai-1022",
    ]
)

# 30 diverse tasks confirmed solvable (reward=1.0) in gold run on lite-300 subset.
# One task per repo — covers 30 different OSS projects for broad signal.
# Use --subset live-golden-30 for rapid iteration / agent development.
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


def _make_benchmark(subset: str | None, task_ids: list[str] | None) -> object:
    from swebench_live_cube.benchmark import SWEBenchLiveBenchmarkConfig

    cfg = SWEBenchLiveBenchmarkConfig(oracle_mode=True)
    if task_ids:
        return cfg.subset_from_list(task_ids)
    if subset == "live30":
        return cfg.subset_from_list(list(_LIVE_30_SAMPLE))
    if subset == "live-golden-30":
        return cfg.subset_from_list(list(_LIVE_GOLDEN_30))
    if subset:
        return cfg.named_subset(subset)
    return cfg


def _make_infra(toolkit: bool, eai_profile: str, eai_path: str, launch_timeout: int) -> object:
    if toolkit:
        from cube_infra_toolkit import ToolkitInfraConfig

        return ToolkitInfraConfig(profile=eai_profile, eai_path=eai_path, launch_timeout_seconds=launch_timeout)
    from cube.infra_local import LocalInfraConfig

    return LocalInfraConfig()


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


def extract_cancelled(run_dir: Path) -> list[str]:
    """Task IDs whose every episode ended CANCELLED (never completed)."""
    eps = Path(run_dir) / "episodes"
    task_statuses: dict[str, set[str]] = {}
    for ep_dir in eps.iterdir():
        if not ep_dir.is_dir():
            continue
        task = ep_dir.name.rsplit("_ep", 1)[0]
        status_f = ep_dir / "status.json"
        if status_f.exists():
            s = json.loads(status_f.read_text())
            task_statuses.setdefault(task, set()).add(s.get("status", ""))
    return sorted(t for t, ss in task_statuses.items() if ss == {"CANCELLED"})


def run_once(
    *,
    subset: str | None = None,
    task_ids: list[str] | None = None,
    n_parallel: int = 50,
    toolkit: bool = False,
    eai_profile: str = "yul101",
    eai_path: str = "eai",
    launch_timeout: int = 900,
    run_label: str = "",
    debug: bool = False,
    retry_dir: Path | None = None,
) -> tuple[Path, ExpResult]:
    """Single gold pass. Returns (output_dir, ExpResult).

    If retry_dir is set, re-runs only the tasks that were CANCELLED in every
    attempt (never completed) — bypasses the max_retries gate.
    """
    if retry_dir is not None:
        cancelled = extract_cancelled(retry_dir)
        if not cancelled:
            logger.info("No CANCELLED-only tasks found in %s — nothing to retry.", retry_dir)
            return retry_dir, ExpResult(exp_id="retry-noop", tasks_num=0)
        logger.info("Retrying %d CANCELLED task(s) from %s", len(cancelled), retry_dir)
        task_ids = cancelled
    infra = _make_infra(toolkit, eai_profile, eai_path, launch_timeout)
    benchmark = _make_benchmark(subset, task_ids)
    infra_label = f"toolkit:{eai_profile}" if toolkit else "local"
    label = f" [{run_label}]" if run_label else ""
    exp = Experiment(
        name=f"gold-patch-baseline-{infra_label}",
        agent_config=GoldPatchAgentConfig(),
        benchmark_config=benchmark,
        infra=infra,
        max_steps=5,
    )
    logger.info("Gold pass%s | %s | subset=%s", label, infra_label, subset or task_ids or "debug")
    if debug:
        run_sequentially(exp)
        return exp.output_dir, ExpResult(exp_id=exp.name, tasks_num=0)
    return exp.output_dir, run_with_ray(exp, n_cpus=n_parallel)


# ---------------------------------------------------------------------------
# Multi-run with solvable output
# ---------------------------------------------------------------------------


def run_gold_baseline(
    *,
    subset: str | None = None,
    task_ids: list[str] | None = None,
    n_runs: int = 1,
    existing_run_dirs: list[Path] | None = None,
    dump_solvable: Path | None = None,
    debug: bool = False,
    **run_kwargs: object,
) -> list[Path]:
    """Run gold baseline N times; optionally write stable/flaky solvable JSON.

    existing_run_dirs — pre-completed run dirs to include in the intersection
                        (reduces the number of fresh runs needed).
    """
    run_dirs: list[Path] = list(existing_run_dirs or [])
    runs_needed = n_runs - len(run_dirs)
    total = n_runs
    last_total = 0

    for _ in range(runs_needed):
        label = f"run {len(run_dirs) + 1}/{total}" if total > 1 else ""
        run_dir, result = run_once(subset=subset, task_ids=task_ids, run_label=label, debug=debug, **run_kwargs)
        run_dirs.append(run_dir)
        last_total = result.tasks_num

    if dump_solvable is not None:
        _write_solvable(run_dirs, dump_solvable, last_total)

    return run_dirs


def _write_solvable(run_dirs: list[Path], out: Path, total: int) -> None:
    if len(run_dirs) == 1:
        solvable = extract_solvable(run_dirs[0])
        out.write_text(json.dumps(solvable, indent=2))
        pct = len(solvable) / total * 100 if total else 0
        logger.info("Solvable: %d/%d (%.1f%%) → %s", len(solvable), total, pct, out)
    else:
        stable, flaky = intersect_solvable(run_dirs)
        out.write_text(json.dumps(stable, indent=2))
        flaky_out = out.with_name(out.stem + "_flaky.json")
        flaky_out.write_text(json.dumps(flaky, indent=2))
        pct = len(stable) / total * 100 if total else 0
        logger.info(
            "%d-run gold | stable=%d/%d (%.1f%%) → %s | flaky=%d → %s",
            len(run_dirs),
            len(stable),
            total,
            pct,
            out,
            len(flaky),
            flaky_out,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s %(message)s")

    parser = argparse.ArgumentParser(description="Gold-patch baseline — SWE-bench Live")
    parser.add_argument(
        "--subset", default=None, choices=["live-golden-30", "live30", "lite", "verified", "full", "test"]
    )
    parser.add_argument("--tasks", default=None, help="Comma-separated task IDs")
    parser.add_argument("--n-runs", type=int, default=1, help="Repeat N times for flakiness detection")
    parser.add_argument("--retry", metavar="DIR", default=None, help="Re-run only CANCELLED/FAILED episodes from DIR")
    parser.add_argument("--existing-run-dir", metavar="DIR", nargs="+", default=None)
    parser.add_argument(
        "--from-runs",
        metavar="DIR",
        nargs="+",
        default=None,
        help="Skip running; intersect these completed run dirs into --dump-solvable",
    )
    parser.add_argument("--dump-solvable", metavar="PATH", default=None)
    parser.add_argument("--n-parallel", type=int, default=50)
    parser.add_argument("--launch-timeout", type=int, default=900)
    parser.add_argument("--toolkit", action="store_true")
    parser.add_argument("--eai-profile", default="yul101")
    parser.add_argument("--eai-path", default="eai")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Post-hoc intersection: skip running, just intersect existing run dirs.
    if args.from_runs:
        if not args.dump_solvable:
            parser.error("--from-runs requires --dump-solvable")
        run_dirs = [Path(d) for d in args.from_runs]
        _write_solvable(run_dirs, Path(args.dump_solvable), total=0)
        sys.exit(0)

    run_gold_baseline(
        subset=args.subset,
        task_ids=[t.strip() for t in args.tasks.split(",")] if args.tasks else None,
        n_runs=args.n_runs,
        existing_run_dirs=[Path(d) for d in args.existing_run_dir] if args.existing_run_dir else None,
        dump_solvable=Path(args.dump_solvable) if args.dump_solvable else None,
        debug=args.debug,
        n_parallel=args.n_parallel,
        launch_timeout=args.launch_timeout,
        toolkit=args.toolkit,
        eai_profile=args.eai_profile,
        eai_path=args.eai_path,
        retry_dir=Path(args.retry) if args.retry else None,
    )

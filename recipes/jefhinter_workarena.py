# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cube-harness",
#     "workarena-cube",
#     "matplotlib",
#     "wandb",
# ]
#
# [tool.uv.sources]
# cube-harness = { path = "..", editable = true }
# workarena-cube = { path = "../cubes/workarena", editable = true }
# ///
"""JefHinter on WorkArena L1. Thin recipe — the harness is ``cube_harness.jefhinter``.

Requires a vLLM endpoint with tool calling AND ``HUGGING_FACE_HUB_TOKEN`` set (the
WorkArena cube resolves a live ServiceNow instance from the gated HF pool — needs
``browsergym-workarena>=0.5.3``).

    HUGGING_FACE_HUB_TOKEN=$HF_TOKEN BROWSERGYM_HEADLESS=1 \
      .venv/bin/python recipes/jefhinter_workarena.py --n-iters 3 --n-seeds 5 --n-parallel 4
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from cube_browser_tool.bgym_tool import BgymToolConfig
from workarena_cube.benchmark import WorkArenaBenchmarkConfig

from cube_harness.jefhinter import run_jefhinter

# The 4 WorkArena L1 demo tasks from the K26 JephHinter demo.
DEMO_TASKS = [
    "workarena.servicenow.sort-incident-list",
    "workarena.servicenow.create-incident",
    "workarena.servicenow.filter-incident-list",
    "workarena.servicenow.order-standard-laptop",
]


def main(
    model: Annotated[str, typer.Option(help="served model name on the vLLM endpoint")] = "qwen3-14b",
    api_base: Annotated[str, typer.Option(help="OpenAI-compatible base url")] = "http://localhost:8002/v1",
    hinter_model: Annotated[str, typer.Option(help="miner model (defaults to --model)")] = "",
    n_iters: Annotated[int, typer.Option(help="hint iterations after baseline")] = 3,
    n_seeds: Annotated[int, typer.Option(help="seeds per L1 task")] = 5,
    max_steps: Annotated[int, typer.Option(help="max agent steps per episode")] = 12,
    n_parallel: Annotated[int, typer.Option(help="parallel episodes via Ray (1 = sequential)")] = 4,
    debug_limit: Annotated[int, typer.Option(help="cap episodes per run for a smoke (0 = no cap)")] = 0,
    tasks: Annotated[str, typer.Option(help="comma-separated task_ids (default: 4 demo tasks)")] = "",
    output_dir: Annotated[str, typer.Option(help="output directory")] = "",
    wandb_enabled: Annotated[bool, typer.Option("--wandb/--no-wandb", help="log to Weights & Biases")] = True,
    wandb_project: Annotated[str, typer.Option(help="W&B project name")] = "jeffhinter",
) -> None:
    # Built explicitly (not via the registry preset) so n_seeds reaches the seed generator,
    # which is constructed once at config-validation time. axtree-only obs for text models.
    cfg = WorkArenaBenchmarkConfig(
        tool_config=BgymToolConfig(use_html=False, use_axtree=True, use_screenshot=False),
        n_seeds_l1=n_seeds,
    ).named_subset("l1")
    task_ids = [t.strip() for t in tasks.split(",") if t.strip()] or DEMO_TASKS
    cfg = cfg.subset_from_list(task_ids)

    out = Path(output_dir) if output_dir else Path.home() / "cube_harness_results" / "jefhinter_workarena_l1"
    run_jefhinter(
        cfg,
        "workarena_l1",
        model=model,
        api_base=api_base,
        hinter_model=hinter_model,
        out_dir=out,
        n_iters=n_iters,
        max_steps=max_steps,
        max_actions=max_steps,
        n_parallel=n_parallel,
        debug_limit=debug_limit or None,
        wandb_enabled=wandb_enabled,
        wandb_project=wandb_project,
    )


if __name__ == "__main__":
    typer.run(main)

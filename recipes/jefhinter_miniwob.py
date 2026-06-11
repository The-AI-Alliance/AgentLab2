# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cube-harness",
#     "miniwob-cube",
#     "matplotlib",
#     "wandb",
# ]
#
# [tool.uv.sources]
# cube-harness = { path = "..", editable = true }
# miniwob-cube = { path = "../cubes/miniwob", editable = true }
# ///
"""JefHinter on MiniWoB. Thin recipe — the harness is ``cube_harness.jefhinter``.

    .venv/bin/python recipes/jefhinter_miniwob.py --n-iters 3 \
        --tasks click-checkboxes-large,click-tab-2-hard,email-inbox
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from miniwob_cube import MINIWOB_CONFIGS

from cube_harness.jefhinter import run_jefhinter

# The canonical "hard" MiniWoB eval set (28 tasks) used by the JeffHinter evals and the
# hint-conditioned RL thread. Picked from task_metadata.json for mid-range difficulty
# (the trivial click-* tier ceilings at ~90%): precision clicking, navigation, forms,
# and reasoning/memory tasks. Empirically ~20-50% baseline for 3B-7B models.
MINIWOB_HARD_TASKS = [
    "click-checkboxes-large",
    "click-checkboxes-transfer",
    "click-checkboxes-soft",
    "click-collapsible-2",
    "navigate-tree",
    "grid-coordinate",
    "use-autocomplete",
    "use-spinner",
    "use-slider",
    "choose-date-easy",
    "enter-date",
    "enter-time",
    "login-user",
    "search-engine",
    "simple-algebra",
    "count-shape",
    "identify-shape",
    "find-word",
    "guess-number",
    "tic-tac-toe",
    "copy-paste",
    "enter-text-dynamic",
    "click-tab-2-hard",
    "email-inbox",
    "social-media",
    "multi-layouts",
    "click-pie",
    "read-table",
]


def main(
    model: Annotated[str, typer.Option(help="served model name on the vLLM endpoint")] = "qwen2.5-7b-instruct",
    api_base: Annotated[str, typer.Option(help="OpenAI-compatible base url")] = "http://localhost:8001/v1",
    hinter_model: Annotated[str, typer.Option(help="miner model (defaults to --model)")] = "",
    hinter_api_base: Annotated[str, typer.Option(help="hinter endpoint (defaults to --api-base)")] = "",
    hinter_max_tokens: Annotated[
        int, typer.Option(help="hinter max completion tokens (raise for reasoning hinters)")
    ] = 1024,
    n_iters: Annotated[int, typer.Option(help="hint iterations after baseline")] = 3,
    repeats: Annotated[int, typer.Option(help="sampled rollouts per task per iteration (pooled)")] = 1,
    temperature: Annotated[float, typer.Option(help="agent sampling temperature (0 = greedy/deterministic)")] = 0.7,
    hinter_temperature: Annotated[float, typer.Option(help="hinter temperature (0 = deterministic mining)")] = 0.6,
    max_steps: Annotated[int, typer.Option(help="max agent steps per episode")] = 8,
    n_parallel: Annotated[int, typer.Option(help="parallel episodes via Ray (1 = sequential)")] = 1,
    miniwob_port: Annotated[
        int, typer.Option(help="local MiniWoB http server port (distinct per concurrent run)")
    ] = 8000,
    debug_limit: Annotated[int, typer.Option(help="cap episodes per run for a smoke (0 = no cap)")] = 0,
    tasks: Annotated[str, typer.Option(help="'hard' (28-task eval set), 'all' (125), or comma-separated ids")] = "hard",
    output_dir: Annotated[str, typer.Option(help="output directory")] = "",
    wandb_enabled: Annotated[bool, typer.Option("--wandb/--no-wandb", help="log to Weights & Biases")] = True,
    wandb_project: Annotated[str, typer.Option(help="W&B project name")] = "jeffhinter",
    wandb_group: Annotated[str, typer.Option(help="W&B group (cluster related runs)")] = "",
    wandb_run_name: Annotated[str, typer.Option(help="override the auto W&B run name")] = "",
) -> None:
    cfg = MINIWOB_CONFIGS["default"]
    cfg.tool_config.use_screenshot = False  # text-only served model -> html-only observation
    cfg.port = miniwob_port
    if tasks == "hard":
        task_ids = list(MINIWOB_HARD_TASKS)
    elif tasks == "all":
        task_ids = []
    else:
        task_ids = [t.strip() for t in tasks.split(",") if t.strip()]
    if task_ids:
        cfg = cfg.subset_from_list(task_ids)

    out = Path(output_dir) if output_dir else Path.home() / "cube_harness_results" / "jefhinter_miniwob"
    run_jefhinter(
        cfg,
        "miniwob",
        model=model,
        api_base=api_base,
        hinter_model=hinter_model,
        hinter_api_base=hinter_api_base,
        hinter_max_tokens=hinter_max_tokens,
        out_dir=out,
        n_iters=n_iters,
        max_steps=max_steps,
        max_actions=max_steps,
        n_parallel=n_parallel,
        debug_limit=debug_limit or None,
        wandb_enabled=wandb_enabled,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
        wandb_run_name=wandb_run_name,
        task_ids=task_ids,
        temperature=temperature,
        hinter_temperature=hinter_temperature,
        repeats=repeats,
    )


if __name__ == "__main__":
    typer.run(main)

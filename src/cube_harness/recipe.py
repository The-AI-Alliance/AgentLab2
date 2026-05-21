"""The one line every recipe ends with.

A recipe is a declarative config file — it builds `Experiment`(s) and hands
them to `run`:

    from cube_harness.recipe import run

    exp = Experiment(name="...", agent_config=..., benchmark_config=...)

    if __name__ == "__main__":
        run(exp)                              # one experiment
        run(exp_a, exp_b)                     # several, run all
        run({"default": exp_a, "swe": exp_b}) # named; --experiment picks one

`run` ships a fixed, generic CLI — identical for every recipe, not
extensible per-recipe (clone the file for anything structural):

    python recipes/foo.py                       # Ray, full benchmark
    python recipes/foo.py --limit 3             # in-process, first 3 tasks
    python recipes/foo.py --ray 32              # override Ray worker count
    python recipes/foo.py --set agent_config.max_actions=200
    python recipes/foo.py --experiment swe      # pick from a named dict
"""

import json
from typing import Annotated

import typer

from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment


def _apply_override(exp: Experiment, dotted: str) -> None:
    """Apply one ``a.b.c=value`` override. Value is JSON-parsed when possible
    (so ``200`` → int), then assigned — `ValidatedConfig` validates the type."""
    path, _, raw = dotted.partition("=")
    if not _:
        raise typer.BadParameter(f"--set expects key=value, got {dotted!r}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    *parents, leaf = path.split(".")
    target = exp
    for part in parents:
        target = getattr(target, part)
    setattr(target, leaf, value)


def run(*args: Experiment | dict[str, Experiment]) -> None:
    """Run experiment(s) through the generic recipe CLI.

    Pass one/more `Experiment`s (all run), or a single `dict[str, Experiment]`
    (the `--experiment` flag picks one; defaults to `"default"`).
    """
    if len(args) == 1 and isinstance(args[0], dict):
        named: dict[str, Experiment] = args[0]
        if not named:
            raise ValueError("run() got an empty experiment dict")
        exps: tuple[Experiment, ...] = ()
    elif args and all(isinstance(a, Experiment) for a in args):
        named = {}
        exps = args  # type: ignore[assignment]
    else:
        raise ValueError("run() takes Experiment(s) or a single dict[str, Experiment]")

    def _main(
        limit: Annotated[int | None, typer.Option(help="Run first N tasks in-process (no Ray) — debug.")] = None,
        ray: Annotated[int, typer.Option(help="Ray worker count (ignored with --limit).")] = 8,
        set_: Annotated[
            list[str] | None, typer.Option("--set", help="Override: dotted.path=value (repeatable).")
        ] = None,
        experiment: Annotated[
            str, typer.Option("--experiment", "-e", help="Named-dict recipes: which experiment to run.")
        ] = "default",
    ) -> None:
        if named:
            if experiment not in named:
                raise typer.BadParameter(f"--experiment {experiment!r} not in {sorted(named)}")
            selected: tuple[Experiment, ...] = (named[experiment],)
        else:
            selected = exps
        for exp in selected:
            for override in set_ or []:
                _apply_override(exp, override)
            if limit is not None:
                run_sequentially(exp, debug_limit=limit)
            else:
                run_with_ray(exp, n_cpus=ray)

    typer.run(_main)

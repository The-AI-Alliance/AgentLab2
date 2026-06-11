# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cube-harness",
#     "workarena-cube",
#     "miniwob-cube",
#     "matplotlib",
# ]
#
# [tool.uv.sources]
# cube-harness = { path = "..", editable = true }
# workarena-cube = { path = "../cubes/workarena", editable = true }
# miniwob-cube = { path = "../cubes/miniwob", editable = true }
# ///
"""JefHinter — in-context self-improvement loop for cube-harness agents.

Reproduces the JephHinter eval (ServiceNow K26 demo) inside cube-harness: an
agent attempts a fixed task set, an LLM *mines hints* from its own failed (and
contrastively, successful) trajectories, the hints are injected back into the
agent via ``GennyConfig.task_hints``, and the agent re-runs. Success rate is
tracked per iteration; the loop reproduces the characteristic upward curve.

What is reused vs. new:
- Reused: ``Experiment`` + runners (rollouts), ``FileStorage`` (reload),
  ``GennyConfig.task_hints`` (injection), and the investigator ``hinter``
  ``TaskHint`` schema (hint shape).
- New (this file): the outer loop, a standalone LLM ``HintMiner`` (the
  investigator's own miner is hard-wired to the Claude Code SDK, so it cannot
  drive a local vLLM/OpenAI model — we mine with ``cube_harness.llm.LLM``), the
  ``HintDB`` curator, and the metrics/curve.

This is the config — copy and edit, or drive via the CLI (``--help``).

    .venv/bin/python recipes/jefhinter.py --benchmark workarena_l1 \
        --api-base http://localhost:8001/v1 --model qwen2.5-7b-instruct
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import typer
from cube.core import Action, EnvironmentOutput
from cube_browser_tool.bgym_tool import BgymToolConfig
from miniwob_cube import MINIWOB_CONFIGS
from workarena_cube.benchmark import WorkArenaBenchmarkConfig

from cube_harness.agents.genny import GennyConfig
from cube_harness.agents.genny_configs import make_agent_config
from cube_harness.analyze.investigator.parse import extract_json_block
from cube_harness.analyze.investigator.use_cases.hinter.recipe import TaskHint
from cube_harness.core import AgentOutput, Trajectory
from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment
from cube_harness.llm import LLM, LLMConfig, Prompt
from cube_harness.storage import FileStorage

logger = logging.getLogger("jefhinter")

# The 4 WorkArena L1 demo tasks from the K26 JephHinter demo.
WORKARENA_L1_DEMO_TASKS = [
    "workarena.servicenow.sort-incident-list",
    "workarena.servicenow.create-incident",
    "workarena.servicenow.filter-incident-list",
    "workarena.servicenow.order-standard-laptop",
]

AGENT_SYSTEM_PROMPT = (
    "You are an expert web agent operating a real browser through an accessibility "
    "tree. Solve the user's task with the fewest, most deliberate actions. Read the "
    "observation carefully before acting; prefer precise element interactions over "
    "guessing. When a hint is provided, follow it."
)

MINER_SYSTEM_PROMPT = """You are a hint-harvesting expert for a web agent.

You read one or two trajectories of an agent attempting the SAME task (a failed
attempt, and when available a successful one for contrast) and extract a short,
reusable hint that would help the agent solve this task on a re-run.

Rules for a good hint:
- GENERALIZABLE: describe a reusable UI strategy or workflow. Do NOT reference
  specific record values, element ids like [123], usernames, or one-off text.
  DO name common UI elements (buttons, column headers, funnel/filter icons,
  menus) and the sequence to use them.
- ACTIONABLE: say what to DO, concretely. One or two sentences.
- GROUNDED: base it on where the failed attempt went wrong (and, if a successful
  trace is shown, on what it did differently).
- Do NOT produce a hint when the failure is clearly a harness/tool bug or the
  evaluator rejected correct behavior; return an empty list instead.

Respond with ONLY a JSON object in a ```json fence, matching:

```json
{"hints": [{"task_id": "<the task_id>", "hint_type": "task_specific",
  "text": "<one or two sentence reusable instruction>",
  "rationale": "<why this would fix the failed attempt>", "confidence": 3}]}
```

`confidence` is 0-5. Emit at most one hint. Emit `{"hints": []}` if no useful
generalizable hint exists."""


# --------------------------------------------------------------------------- #
# Configuration builders
# --------------------------------------------------------------------------- #
def build_llm(model: str, api_base: str, temperature: float, max_completion_tokens: int) -> LLMConfig:
    """LLMConfig routed at a local vLLM OpenAI-compatible endpoint."""
    return LLMConfig(
        model_name=f"openai/{model}",
        api_base=api_base,
        api_key="EMPTY",
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )


def build_benchmark(benchmark: str, task_ids: list[str] | None, n_seeds: int):
    """Build the benchmark config subset for the loop.

    Returns a ``BenchmarkConfig``. ``task_ids`` restricts to a subset; ``None``
    keeps the registry default subset.
    """
    if benchmark == "workarena_l1":
        # axtree-only (no screenshot): the served model is text-only, and axtree is the
        # canonical WorkArena observation (AgentLab's GPT-4o text config runs axtree-only).
        browser = BgymToolConfig(use_html=False, use_axtree=True, use_screenshot=False)
        cfg = WorkArenaBenchmarkConfig(tool_config=browser, n_seeds_l1=n_seeds).named_subset("l1")
        return cfg.subset_from_list(task_ids or WORKARENA_L1_DEMO_TASKS)
    if benchmark == "miniwob":
        cfg = MINIWOB_CONFIGS["default"]
        cfg.tool_config.use_screenshot = False  # text-only served model
        return cfg.subset_from_list(task_ids) if task_ids else cfg
    raise ValueError(f"unknown benchmark {benchmark!r} (expected workarena_l1 | miniwob)")


def build_agent(llm_config: LLMConfig, task_hints: dict[str, str], max_actions: int, cost_limit: float) -> GennyConfig:
    """Genny agent config with the current per-task hints injected.

    Built from the canonical ``make_agent_config``; we override the system prompt
    for the web setting, inject the per-task hints, and switch to rolling-summary
    history. Web (axtree) observations are large (~8-10k tokens each); flat history
    overflows a 32k context by ~step 3, so we use summarize mode (bounded context)
    plus an obs-size cap.
    """
    cfg = make_agent_config(
        llm_config=llm_config, template="workflow-generic", max_actions=max_actions, cost_limit=cost_limit
    )
    cfg.system_prompt = AGENT_SYSTEM_PROMPT
    cfg.task_hints = dict(task_hints)
    cfg.flat_history = False
    cfg.enable_summarize = True  # rolling summaries -> bounded context for large web obs
    cfg.max_obs_chars = 40000  # safety cap (~10k tokens) for pathologically large axtrees
    return cfg


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _reward(traj: Trajectory) -> float:
    val = (traj.reward_info or {}).get("reward")
    return float(val) if val is not None else 0.0


def score_trajectories(trajectories: list[Trajectory]) -> dict:
    """Overall + per-task success rate (reward > 0). WorkArena has no `success` key."""
    per_task: dict[str, list[float]] = defaultdict(list)
    for t in trajectories:
        per_task[t.metadata.get("task_id", "?")].append(1.0 if _reward(t) > 0 else 0.0)
    per_task_rate = {k: sum(v) / len(v) for k, v in per_task.items()}
    n = sum(len(v) for v in per_task.values())
    overall = sum(sum(v) for v in per_task.values()) / n if n else 0.0
    return {"overall": overall, "per_task": per_task_rate, "n_episodes": n}


# --------------------------------------------------------------------------- #
# Transcript rendering (for the miner)
# --------------------------------------------------------------------------- #
def _action_repr(action: Action) -> str:
    args = getattr(action, "arguments", None)
    return f"{getattr(action, 'name', '?')}({args})" if args else f"{getattr(action, 'name', '?')}()"


def render_trajectory(traj: Trajectory, max_agent_steps: int = 25) -> str:
    """Compact think/action/error/reward transcript — no raw axtree (matches JephHinter default)."""
    lines: list[str] = [f"task_id: {traj.metadata.get('task_id', '?')}", f"final_reward: {_reward(traj)}", ""]
    agent_steps = 0
    for step in traj.steps:
        out = step.output
        if isinstance(out, AgentOutput):
            agent_steps += 1
            if agent_steps > max_agent_steps:
                lines.append("... (trajectory truncated) ...")
                break
            if out.thoughts:
                lines.append(f"[step {agent_steps}] THINK: {out.thoughts.strip()[:600]}")
            acts = ", ".join(_action_repr(a) for a in out.actions) or "(no action)"
            lines.append(f"[step {agent_steps}] ACTION: {acts}")
            if out.error:
                lines.append(f"[step {agent_steps}] AGENT_ERROR: {str(out.error)[:300]}")
        elif isinstance(out, EnvironmentOutput):
            info = getattr(out, "info", None) or {}
            err = info.get("last_action_error") or info.get("error")
            if err:
                lines.append(f"          ENV_ERROR: {str(err)[:300]}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Hint miner
# --------------------------------------------------------------------------- #
class HintMiner:
    """Mine one generalizable hint per failing task from its trajectories."""

    def __init__(self, llm_config: LLMConfig, max_agent_steps: int = 25) -> None:
        self._llm: LLM = llm_config.make()
        self._max_agent_steps = max_agent_steps

    def mine(self, trajectories: list[Trajectory]) -> list[TaskHint]:
        by_task: dict[str, list[Trajectory]] = defaultdict(list)
        for t in trajectories:
            by_task[t.metadata.get("task_id", "?")].append(t)

        hints: list[TaskHint] = []
        for task_id, trajs in by_task.items():
            failed = [t for t in trajs if _reward(t) <= 0]
            passed = [t for t in trajs if _reward(t) > 0]
            if not failed:
                continue  # nothing to learn from on this task this round
            hint = self._mine_one(task_id, failed[0], passed[0] if passed else None)
            if hint is not None:
                hints.append(hint)
        return hints

    def _mine_one(self, task_id: str, failed: Trajectory, passed: Trajectory | None) -> TaskHint | None:
        parts = [
            f"Task: {task_id}",
            "",
            "--- FAILED ATTEMPT ---",
            render_trajectory(failed, self._max_agent_steps),
        ]
        if passed is not None:
            parts += ["", "--- SUCCESSFUL ATTEMPT (for contrast) ---", render_trajectory(passed, self._max_agent_steps)]
        parts += ["", "Produce the JSON hint object now."]
        prompt = Prompt(
            messages=[
                {"role": "system", "content": MINER_SYSTEM_PROMPT},
                {"role": "user", "content": "\n".join(parts)},
            ]
        )
        try:
            response = self._llm(prompt)
            obj = extract_json_block(response.message.content or "")
        except Exception as exc:  # noqa: BLE001 - mining is best-effort per task
            logger.warning("mining failed for %s: %s", task_id, exc)
            return None
        raw_hints = obj.get("hints") or []
        if not raw_hints:
            return None
        raw = raw_hints[0]
        raw.setdefault("task_id", task_id)
        raw.setdefault("hint_type", "task_specific")
        raw.setdefault("rationale", "")
        raw.setdefault("confidence", 3)
        try:
            return TaskHint.model_validate(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("hint for %s failed validation: %s", task_id, exc)
            return None


# --------------------------------------------------------------------------- #
# Hint database / curator
# --------------------------------------------------------------------------- #
class HintDB:
    """Accumulates mined hints across iterations, deduped, capped per task."""

    def __init__(self, max_per_task: int = 6) -> None:
        self._by_task: dict[str, list[str]] = defaultdict(list)
        self._max_per_task = max_per_task

    def add(self, hints: list[TaskHint]) -> int:
        added = 0
        for h in hints:
            text = h.text.strip()
            if not text:
                continue
            existing = self._by_task[h.task_id]
            if any(text.lower() == e.lower() for e in existing):
                continue  # exact-dedup (curation)
            existing.append(text)
            if len(existing) > self._max_per_task:  # keep the most recent N
                del existing[: len(existing) - self._max_per_task]
            added += 1
        return added

    def as_task_hints(self) -> dict[str, str]:
        """Render each task's hint list into the single string Genny injects."""
        out: dict[str, str] = {}
        for task_id, hints in self._by_task.items():
            if hints:
                bullets = "\n".join(f"- {h}" for h in hints)
                out[task_id] = f"Hints learned from past attempts at this task:\n{bullets}"
        return out

    @property
    def size(self) -> int:
        return sum(len(v) for v in self._by_task.values())

    def to_dict(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in self._by_task.items()}


# --------------------------------------------------------------------------- #
# Run a single experiment and reload its trajectories
# --------------------------------------------------------------------------- #
def run_one(
    label: str,
    agent_config: GennyConfig,
    benchmark_config,
    out_dir: Path,
    max_steps: int,
    n_parallel: int,
    debug_limit: int | None,
) -> list[Trajectory]:
    """Run one experiment; return its trajectories (with steps) from disk."""
    run_dir = out_dir / label
    exp = Experiment(
        name=f"jefhinter-{label}",
        agent_config=agent_config,
        benchmark_config=benchmark_config,
        output_dir=run_dir,
        max_steps=max_steps,
    )
    logger.info("[%s] running -> %s", label, run_dir)
    if n_parallel > 1 and debug_limit is None:
        run_with_ray(exp, n_cpus=n_parallel)
    else:
        run_sequentially(exp, debug_limit=debug_limit)
    trajectories = FileStorage(exp.output_dir).load_all_trajectories()
    logger.info("[%s] %d trajectories loaded", label, len(trajectories))
    return trajectories


# --------------------------------------------------------------------------- #
# Metrics + plot
# --------------------------------------------------------------------------- #
def dump_metrics(records: list[dict], db: HintDB, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"iterations": records, "hint_db": db.to_dict()}, indent=2))
    logger.info("metrics -> %s", path)


def plot_curve(records: list[dict], path: Path, title: str) -> None:
    """Per-iteration overall success-rate curve (the JephHinter signature plot)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["label"] for r in records]
    ys = [r["overall"] * 100 for r in records]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(xs)), ys, marker="o", color="#1f7a8c")
    for i, y in enumerate(ys):
        ax.annotate(f"{y:.0f}%", (i, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, rotation=0)
    ax.set_ylabel("success rate (%)")
    ax.set_xlabel("baseline / hint iteration")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    logger.info("curve -> %s", path)


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
def run_jefhinter_loop(
    benchmark_config,
    agent_llm: LLMConfig,
    hinter_llm: LLMConfig,
    out_dir: Path,
    n_iters: int,
    max_steps: int,
    max_actions: int,
    cost_limit: float,
    n_parallel: int,
    debug_limit: int | None,
    benchmark_name: str,
) -> list[dict]:
    """baseline -> (mine -> inject -> re-run) x n_iters, tracking success per iteration."""
    db = HintDB()
    miner = HintMiner(hinter_llm)
    records: list[dict] = []

    for it in range(n_iters + 1):
        label = "baseline" if it == 0 else f"iter{it}"
        task_hints = {} if it == 0 else db.as_task_hints()
        agent = build_agent(agent_llm, task_hints, max_actions, cost_limit)
        trajectories = run_one(label, agent, benchmark_config, out_dir, max_steps, n_parallel, debug_limit)

        score = score_trajectories(trajectories)
        record = {"iteration": it, "label": label, "hints_in_db": db.size, **score}
        records.append(record)
        logger.info(
            "[%s] overall=%.1f%% (%d eps, %d hints) per_task=%s",
            label,
            score["overall"] * 100,
            score["n_episodes"],
            db.size,
            score["per_task"],
        )

        if it < n_iters:  # mine for the next iteration
            new_hints = miner.mine(trajectories)
            added = db.add(new_hints)
            logger.info("[%s] mined %d hints (%d new); db now %d", label, len(new_hints), added, db.size)

        dump_metrics(records, db, out_dir / "metrics.json")  # checkpoint each iteration

    plot_curve(records, out_dir / "curve.png", f"JefHinter on {benchmark_name} — success vs hint iteration")
    return records


def main(
    benchmark: Annotated[str, typer.Option(help="workarena_l1 | miniwob")] = "workarena_l1",
    model: Annotated[str, typer.Option(help="served model name on the vLLM endpoint")] = "qwen2.5-7b-instruct",
    api_base: Annotated[str, typer.Option(help="OpenAI-compatible base url")] = "http://localhost:8001/v1",
    hinter_model: Annotated[str, typer.Option(help="miner model (defaults to --model)")] = "",
    n_iters: Annotated[int, typer.Option(help="number of hint iterations after baseline")] = 3,
    n_seeds: Annotated[int, typer.Option(help="seeds per task (WorkArena L1)")] = 5,
    max_steps: Annotated[int, typer.Option(help="max agent steps per episode")] = 15,
    max_actions: Annotated[int, typer.Option(help="agent action budget")] = 15,
    cost_limit: Annotated[float, typer.Option(help="per-episode cost limit (USD)")] = 5.0,
    n_parallel: Annotated[int, typer.Option(help="parallel episodes via Ray (1 = sequential)")] = 1,
    debug_limit: Annotated[int, typer.Option(help="cap episodes per run for a smoke (0 = no cap)")] = 0,
    tasks: Annotated[str, typer.Option(help="comma-separated task_ids subset (default: 4 L1 demo tasks)")] = "",
    output_dir: Annotated[str, typer.Option(help="output directory")] = "",
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out_dir = Path(output_dir) if output_dir else Path.home() / "cube_harness_results" / f"jefhinter_{benchmark}"
    task_ids = [t.strip() for t in tasks.split(",") if t.strip()] or None

    agent_llm = build_llm(model, api_base, temperature=0.7, max_completion_tokens=1536)
    hinter_llm = build_llm(hinter_model or model, api_base, temperature=0.6, max_completion_tokens=1024)
    benchmark_config = build_benchmark(benchmark, task_ids, n_seeds)

    logger.info("JefHinter: benchmark=%s model=%s n_iters=%d -> %s", benchmark, model, n_iters, out_dir)
    records = run_jefhinter_loop(
        benchmark_config=benchmark_config,
        agent_llm=agent_llm,
        hinter_llm=hinter_llm,
        out_dir=out_dir,
        n_iters=n_iters,
        max_steps=max_steps,
        max_actions=max_actions,
        cost_limit=cost_limit,
        n_parallel=n_parallel,
        debug_limit=debug_limit or None,
        benchmark_name=benchmark,
    )
    print("\n=== JefHinter summary ===")
    for r in records:
        print(f"  {r['label']:>9}: {r['overall'] * 100:5.1f}%  ({r['n_episodes']} eps, {r['hints_in_db']} hints)")


if __name__ == "__main__":
    typer.run(main)

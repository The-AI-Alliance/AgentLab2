"""JefHinter — in-context self-improvement eval harness for cube-harness agents.

Reproduces the JephHinter eval (ServiceNow K26 demo) natively: an agent attempts
a benchmark, an LLM *mines hints* from its own (failed, and contrastively
successful) trajectories, the hints are injected via ``GennyConfig.task_hints``,
and the agent re-runs — tracking success rate per iteration; the loop reproduces
the characteristic upward curve.

This module is the **benchmark-agnostic harness**: the miner, the curator, the
outer loop, and metrics. It takes a ``BenchmarkConfig`` object — a thin per-cube
recipe builds that config and calls :func:`run_jefhinter`. See
``recipes/jefhinter_miniwob.py`` and ``recipes/jefhinter_workarena.py``.

Reused cube-harness primitives: ``Experiment`` + runners (rollouts),
``FileStorage`` (reload), ``GennyConfig.task_hints`` (injection), and the
investigator ``hinter`` ``TaskHint`` schema (hint shape). New here: the outer
loop, a standalone LLM :class:`HintMiner` (the investigator's own miner is
hard-wired to the Claude Code SDK, so it cannot drive a local vLLM/OpenAI model —
we mine with ``cube_harness.llm.LLM``), the :class:`HintDB` curator, and the
metrics/curve.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from cube.core import Action, EnvironmentOutput

from cube_harness.agents.genny import GennyConfig
from cube_harness.agents.genny_configs import make_agent_config
from cube_harness.analyze.investigator.parse import extract_json_block
from cube_harness.analyze.investigator.use_cases.hinter.recipe import TaskHint
from cube_harness.core import AgentOutput, Trajectory
from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment
from cube_harness.llm import LLM, LLMConfig, Prompt
from cube_harness.storage import FileStorage

try:
    import wandb
except ImportError:  # optional dependency
    wandb = None

logger = logging.getLogger("jefhinter")

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


def build_genny_agent(
    llm_config: LLMConfig, task_hints: dict[str, str], max_actions: int, cost_limit: float
) -> GennyConfig:
    """Genny agent config with the current per-task hints injected.

    Built from the canonical ``make_agent_config``; we override the system prompt
    for the web setting, inject the per-task hints, and switch to rolling-summary
    history. Web (axtree/html) observations are large (~8-10k tokens each); flat
    history overflows a 32k context by ~step 3, so we use summarize mode (bounded
    context) plus an obs-size cap.
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
    repeats: int = 1,
) -> list[Trajectory]:
    """Run the experiment ``repeats`` times (sampled rollouts) and pool trajectories.

    MiniWoB tasks are fixed instances, so with temperature > 0 the repeats are i.i.d.
    draws of the agent's stochastic policy — pooling them turns the per-task success
    into a stable fraction (the instrument-noise fix; see the hint_conditioned_rl
    thread's Experiment 0 calibration).
    """
    trajectories: list[Trajectory] = []
    for rep in range(repeats):
        suffix = f"-rep{rep}" if repeats > 1 else ""
        run_dir = out_dir / label / f"rep{rep}" if repeats > 1 else out_dir / label
        exp = Experiment(
            name=f"jefhinter-{label}{suffix}",
            agent_config=agent_config,
            benchmark_config=benchmark_config,
            output_dir=run_dir,
            max_steps=max_steps,
        )
        logger.info("[%s%s] running -> %s", label, suffix, run_dir)
        if n_parallel > 1 and debug_limit is None:
            run_with_ray(exp, n_cpus=n_parallel)
        else:
            run_sequentially(exp, debug_limit=debug_limit)
        trajectories += FileStorage(exp.output_dir).load_all_trajectories()
    logger.info("[%s] %d trajectories loaded (%d repeats)", label, len(trajectories), repeats)
    return trajectories


# --------------------------------------------------------------------------- #
# Metrics + plot + optional W&B
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
    fig.savefig(str(path), dpi=120)
    logger.info("curve -> %s", path)


class WandbLogger:
    """Optional W&B logging of the per-iteration success curve + mined hints."""

    def __init__(
        self,
        enabled: bool,
        project: str,
        run_name: str,
        config: dict,
        tags: list[str] | None = None,
        group: str | None = None,
    ) -> None:
        self._run = None
        if enabled and wandb is not None:
            self._run = wandb.init(project=project, name=run_name, config=config, tags=tags, group=group, reinit=True)
        elif enabled:
            logger.warning("wandb requested but not installed; skipping W&B logging")

    def log_iteration(self, record: dict) -> None:
        if self._run is None:
            return
        row = {"success_rate": record["overall"], "n_hints": record["hints_in_db"]}
        for task_id, rate in record["per_task"].items():
            row[f"task_success/{task_id.split('.')[-1]}"] = rate
        self._run.log(row, step=record["iteration"])

    def log_final(self, curve_path: Path, db: HintDB) -> None:
        if self._run is None:
            return
        if curve_path.exists():
            self._run.log({"curve": wandb.Image(str(curve_path))})
        table = wandb.Table(columns=["task_id", "hint"])
        for task_id, hints in db.to_dict().items():
            for hint in hints:
                table.add_data(task_id, hint)
        self._run.log({"hints": table})

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()


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
    wandb_logger: WandbLogger | None = None,
    repeats: int = 1,
) -> list[dict]:
    """baseline -> (mine -> inject -> re-run) x n_iters, tracking success per iteration."""
    db = HintDB()
    miner = HintMiner(hinter_llm)
    records: list[dict] = []

    for it in range(n_iters + 1):
        label = "baseline" if it == 0 else f"iter{it}"
        task_hints = {} if it == 0 else db.as_task_hints()
        agent = build_genny_agent(agent_llm, task_hints, max_actions, cost_limit)
        trajectories = run_one(
            label, agent, benchmark_config, out_dir, max_steps, n_parallel, debug_limit, repeats=repeats
        )

        score = score_trajectories(trajectories)
        record = {"iteration": it, "label": label, "hints_in_db": db.size, **score}
        records.append(record)
        if wandb_logger is not None:
            wandb_logger.log_iteration(record)
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

    curve_path = out_dir / "curve.png"
    plot_curve(records, curve_path, f"JefHinter on {benchmark_name} — success vs hint iteration")
    if wandb_logger is not None:
        wandb_logger.log_final(curve_path, db)
        wandb_logger.finish()
    return records


def run_jefhinter(
    benchmark_config,
    benchmark_name: str,
    *,
    model: str,
    api_base: str,
    out_dir: Path,
    hinter_model: str = "",
    hinter_api_base: str = "",
    hinter_max_tokens: int = 1024,
    n_iters: int = 3,
    max_steps: int = 15,
    max_actions: int = 15,
    cost_limit: float = 5.0,
    n_parallel: int = 1,
    debug_limit: int | None = None,
    wandb_enabled: bool = True,
    wandb_project: str = "jeffhinter",
    wandb_tags: tuple[str, ...] = ("eval",),
    wandb_group: str = "",
    wandb_run_name: str = "",
    task_ids: list[str] | None = None,
    temperature: float = 0.7,
    hinter_temperature: float = 0.6,
    repeats: int = 1,
) -> list[dict]:
    """Convenience entry: build LLM configs + W&B logger, run the loop, log a summary.

    The thin per-cube recipes build ``benchmark_config`` from their registry and
    call this.
    """
    agent_llm = build_llm(model, api_base, temperature=temperature, max_completion_tokens=1536)
    hinter_llm = build_llm(
        hinter_model or model,
        hinter_api_base or api_base,
        temperature=hinter_temperature,
        max_completion_tokens=hinter_max_tokens,
    )
    hinter_name = hinter_model or model
    run_name = wandb_run_name
    if not run_name:
        run_name = f"{benchmark_name}-{model}"
        if hinter_name != model:
            run_name += f"-hinter-{hinter_name}"
        run_name += f"-iters{n_iters}"
    wandb_logger = WandbLogger(
        enabled=wandb_enabled,
        project=wandb_project,
        run_name=run_name,
        group=wandb_group or None,
        tags=list(wandb_tags),
        config={
            "benchmark": benchmark_name,
            "model": model,
            "hinter_model": hinter_name,
            "api_base": api_base,
            "hinter_api_base": hinter_api_base or api_base,
            "n_iters": n_iters,
            "max_steps": max_steps,
            "n_tasks": len(task_ids) if task_ids else None,
            "tasks": sorted(task_ids) if task_ids else "benchmark-default",
            "temperature": temperature,
            "hinter_temperature": hinter_temperature,
            "repeats": repeats,
        },
    )
    logger.info("JefHinter: benchmark=%s model=%s n_iters=%d -> %s", benchmark_name, model, n_iters, out_dir)
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
        debug_limit=debug_limit,
        benchmark_name=benchmark_name,
        wandb_logger=wandb_logger,
        repeats=repeats,
    )
    logger.info("=== JefHinter summary (%s) ===", benchmark_name)
    for r in records:
        logger.info(
            "  %9s: %5.1f%%  (%d eps, %d hints)", r["label"], r["overall"] * 100, r["n_episodes"], r["hints_in_db"]
        )
    return records

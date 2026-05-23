"""Headless launcher for the Auto-CUBE outer loop.

Runs a use case's methodology — its ``SKILL.md``, used verbatim as the agent's
system prompt — as a headless Claude Agent SDK session, mirroring how the
Investigator runs per-episode analyses (it reuses the same
``ClaudeCodeSDKDriver``). The orchestrator makes experiments by writing and
running recipes via Bash (subprocess) — Python is the configuration.

    ch-auto-cube --use-case hinter --objective "raise miniwob score on slider tasks"

Each session is rooted at ``~/auto_cube/<session-id>/`` (``experiments/`` via
``CH_EXP_DIR`` + ``journal/``). Re-run with the same ``--session-id`` to
**resume**: the agent reads the existing journal and continues.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from cube_harness.analyze.investigator.agent_driver import DriverResult

logger = logging.getLogger(__name__)

USE_CASES_DIR = Path(__file__).parent / "use_cases"

# Allowlist for the outer-loop orchestrator. Broader than the Investigator's
# read-only set: it authors exp_config.py + journal entries (Write/Edit) and
# runs experiments / git / gh / ch-investigate (Bash). Bash is intentionally
# powerful — a headless session ships PRs and stands up infra.
AUTO_CUBE_ALLOWED_TOOLS: tuple[str, ...] = ("Read", "Write", "Edit", "Bash", "Glob", "Grep")

DEFAULT_MODEL = "claude-opus-4-7"
# Safety cap on agent turns. The outer loop is long; a session that hits the
# cap is resumed (same --session-id) rather than lost — the journal is the
# checkpoint.
DEFAULT_MAX_TURNS = 500


def _auto_cube_root() -> Path:
    return Path("~/auto_cube").expanduser()


def load_system_prompt(use_case: str) -> str:
    """Return a use case's ``SKILL.md`` verbatim — the agent's system prompt."""
    skill = USE_CASES_DIR / use_case / "SKILL.md"
    if not skill.is_file():
        available = sorted(p.name for p in USE_CASES_DIR.iterdir() if (p / "SKILL.md").is_file())
        raise FileNotFoundError(f"No SKILL.md for use case {use_case!r}; available: {available}")
    return skill.read_text()


def build_kickoff(use_case: str, objective: str, session_dir: Path, resume: bool) -> str:
    """The user-turn that kicks off (or resumes) a session. The methodology
    lives in the system prompt; this only frames the objective and paths."""
    journal = session_dir / "journal"
    experiments = session_dir / "experiments"
    if resume:
        return (
            f"Resume the Auto-CUBE **{use_case}** session at `{session_dir}`.\n\n"
            f"Read `{journal}/session.md` and the latest `round_<N>/notes.md`, then continue the "
            f"methodology from where it left off toward this objective:\n\n{objective}\n\n"
            f"Experiment output goes to `{experiments}` (CH_EXP_DIR is already exported); the "
            f"journal lives at `{journal}`."
        )
    return (
        f"Run an Auto-CUBE **{use_case}** session toward this objective:\n\n{objective}\n\n"
        f"Session id: `{session_dir.name}`. Create `{journal}/session.md` from the template and "
        f"work the methodology in your system prompt. Experiment output goes to `{experiments}` "
        f"(CH_EXP_DIR is already exported); keep all journal artifacts under `{journal}`."
    )


def run_session(
    *,
    use_case: str,
    objective: str,
    session_id: str | None = None,
    model: str = DEFAULT_MODEL,
    max_turns: int = DEFAULT_MAX_TURNS,
    cwd: Path | None = None,
    verbose: bool = True,
) -> "DriverResult":
    """Launch (or resume) one headless Auto-CUBE session and return the result."""
    # Lazy import: `cube_harness.analyze.*` eagerly pulls pandas/gradio, and the
    # SDK itself is the optional `investigator` extra — keep this module import-light.
    from cube_harness.analyze.investigator.agent_driver import ClaudeCodeSDKDriver

    system_prompt = load_system_prompt(use_case)
    session_id = session_id or f"{use_case}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    session_dir = _auto_cube_root() / session_id
    journal = session_dir / "journal"
    experiments = session_dir / "experiments"
    resume = (journal / "session.md").is_file()
    journal.mkdir(parents=True, exist_ok=True)
    experiments.mkdir(parents=True, exist_ok=True)

    # Decision: experiments run as subprocesses; this exports CH_EXP_DIR so every
    # run the agent launches lands in the session instead of ~/cube_harness_results/.
    os.environ["CH_EXP_DIR"] = str(experiments)

    cwd = Path(cwd) if cwd else Path.cwd()
    kickoff = build_kickoff(use_case, objective, session_dir, resume)
    logger.info(
        "Auto-CUBE %s session %s (%s) — cwd=%s, model=%s",
        use_case,
        session_id,
        "resume" if resume else "new",
        cwd,
        model,
    )
    driver = ClaudeCodeSDKDriver()
    return asyncio.run(
        driver.run(
            system_prompt=system_prompt,
            user_prompt=kickoff,
            cwd=cwd,
            additional_dirs=[session_dir],
            model=model,
            allowed_tools=AUTO_CUBE_ALLOWED_TOOLS,
            permission_mode="bypassPermissions",
            verbose=verbose,
            max_turns=max_turns,
        )
    )


def _main(
    objective: str = typer.Option(..., "--objective", "-o", help="What this session should achieve."),
    use_case: str = typer.Option("debug", "--use-case", "-u", help="Auto-CUBE use case (debug, hinter, ...)."),
    session_id: str | None = typer.Option(None, "--session-id", help="Reuse to resume an existing session."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="Orchestrator model."),
    max_turns: int = typer.Option(DEFAULT_MAX_TURNS, "--max-turns", help="Safety cap on turns; resume to continue."),
    cwd: Path | None = typer.Option(None, "--cwd", help="Agent working dir (default: current dir)."),
) -> None:
    """Launch the Auto-CUBE outer loop headlessly via the Claude Agent SDK."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = run_session(
        use_case=use_case,
        objective=objective,
        session_id=session_id,
        model=model,
        max_turns=max_turns,
        cwd=cwd,
    )
    print(f"\nSession done — {result.duration_s:.0f}s, ${result.cost_usd:.2f}, {len(result.actions)} tool calls.")


def main() -> None:
    typer.run(_main)


if __name__ == "__main__":
    main()

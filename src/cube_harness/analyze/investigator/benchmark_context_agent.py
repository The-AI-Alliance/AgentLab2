"""Sub-agent that auto-generates `<experiment_dir>/investigation_context.md`.

When `_investigate_episode_impl` runs and `find_default_context_file(experiment_dir)`
raises `FileNotFoundError`, this agent walks `experiment_config.json`, identifies
the cube package, agent package, and `cube_harness` source, and emits a
```paths fenced block in the format `validate_context_file` already parses.

A driver is required — the previous "no driver, use a venv-walk heuristic"
fallback was speculative and never used in practice. Callers without a
driver have no business invoking this function.

A thin CLI wrapper (`ch-investigate init-context`) calls the same function for
ad-hoc bootstrap.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from cube_harness.analyze.investigator.agent_driver import AgentDriver
from cube_harness.analyze.investigator.context import _PATHS_FENCE_RE, INVESTIGATION_CONTEXT_FILENAME

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_MODEL = "claude-opus-4-7"

BENCHMARK_CONTEXT_SYSTEM_PROMPT = """You are a setup agent for the cube-harness trajectory investigator.

Your single job: produce a markdown file (`investigation_context.md`) that lists every
local source directory the investigator will need read access to in order to analyse an
agent episode.

You have read-only tools (Read / Glob / Grep / Bash). Do not write files via
Bash — your only output is the assistant message containing the markdown.

Procedure:
1. Read `experiment_config.json` in the working directory. It is JSON with
   `_type` strings naming the agent class and benchmark class (full dotted paths).
2. For each `_type`, find the on-disk directory of the package's top-level
   module. `python -c "import importlib.util as u; print(u.find_spec('PKG').origin)"`
   returns a file path; the parent directory is what you want.
3. Always include the `cube_harness` source root (parent of `cube_harness/__init__.py`)
   and the `cube` (cube-standard) source root.
4. If the experiment uses an infra package (look for `infra._type` in the config),
   include its source root too.
5. Verify each path actually exists locally before listing it. Skip anything
   missing — better to list nothing than to hallucinate.

Output format: a markdown file beginning with a one-line description, then a
fenced block of paths. Each line of the block is `name: /absolute/path`.
Example:

```paths
cube_package: /abs/path/to/cubes/swebench_verified
agent_package: /abs/path/to/cube_harness/agents
cube_harness: /abs/path/to/src/cube_harness
cube_standard: /abs/path/to/cube
```

Names are free-form labels for human readers; the investigator only uses the paths.
Pick descriptive names (`cube_package`, `agent_package`, `cube_harness`,
`cube_standard`, `infra_package`). One path per line. No trailing comments.

Reply with the markdown content only — no preamble, no closing chatter."""


def _user_prompt_for(experiment_dir: Path) -> str:
    """Build the per-experiment user prompt for the context sub-agent."""
    return f"""Experiment directory: {experiment_dir}

Read `experiment_config.json` from this directory and produce `investigation_context.md`
contents per the procedure in the system prompt. Reply with the markdown only."""


def _extract_markdown(output_text: str) -> str:
    """Extract the markdown body from the agent's response.

    The agent is instructed to reply with markdown only, but in practice it may
    wrap its answer in a ```markdown fence or add preamble. We try to find a
    fenced markdown block first; failing that, we look for a `paths` fence and
    keep everything from the start of the message up through it.
    """
    fence_md = re.search(r"```(?:markdown|md)\s*\n(.*?)```", output_text, re.DOTALL | re.IGNORECASE)
    if fence_md:
        return fence_md.group(1).strip() + "\n"
    if _PATHS_FENCE_RE.search(output_text):
        return output_text.strip() + "\n"
    raise ValueError("benchmark-context-agent did not emit a ```paths block")


async def generate_context_file(
    experiment_dir: Path,
    *,
    driver: AgentDriver,
    model: str = DEFAULT_CONTEXT_MODEL,
    verbose: bool = False,
) -> Path:
    """Invoke the sub-agent and write `<experiment_dir>/investigation_context.md`.

    The driver is required — there is no offline / no-driver fallback.
    """
    experiment_dir = Path(experiment_dir).resolve()
    out = experiment_dir / INVESTIGATION_CONTEXT_FILENAME

    user_prompt = _user_prompt_for(experiment_dir)
    result = await driver.run(
        system_prompt=BENCHMARK_CONTEXT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        cwd=experiment_dir,
        additional_dirs=[],
        model=model,
        verbose=verbose,
    )
    markdown = _extract_markdown(result.output_text)
    out.write_text(markdown)
    logger.info("benchmark-context-agent wrote %s", out)
    return out


__all__ = [
    "BENCHMARK_CONTEXT_SYSTEM_PROMPT",
    "DEFAULT_CONTEXT_MODEL",
    "generate_context_file",
]

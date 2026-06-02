"""Canonical Genny configs.

    from cube_harness.agents.genny_configs import GENNY_CONFIGS
    from cube_harness.llm import LLMConfig

    agent = GENNY_CONFIGS["swe"]
    agent.llm_config = LLMConfig(model_name="gpt-5.4-mini", temperature=1.0)
    agent.budget.cost_limit = 2.0

Every lookup returns a fresh deep copy (see `ConfigRegistry`). The building
blocks below (`INSTANCE_TEMPLATES`, `make_agent_config`) stay public for
recipes that need a non-canonical combination. There is no model registry —
a recipe constructs the `LLMConfig` it wants and assigns it.
"""

from cube.core import ConfigRegistry

from cube_harness.agents.genny import GennyConfig
from cube_harness.llm import LLMConfig

SYSTEM_PROMPT = "You are a helpful assistant that can interact with a computer shell to solve programming tasks. If an action seems to have no apparent effect, avoid retrying it."

_WORKFLOW_BLOCK = """\
Suggested approach:
1. Reproduce: confirm the current behavior — run the relevant test or command to observe the issue.
2. Explore: read the relevant source files directly with `cat`, `grep`, or `find` to locate the root cause.
3. Fix: apply the minimal change needed to resolve the issue.
4. Verify: run the relevant test or command to confirm the fix works and nothing else broke. If it does not, make a focused adjustment and try again.\
"""

_WORKFLOW_BLOCK_GENERIC = """\
Suggested approach:
1. Explore: inspect the environment — read relevant files, run diagnostic commands, or reproduce the issue before making changes.
2. Act: execute the minimal steps needed to complete the task.
3. Verify: confirm the result meets the requirement. If it does not, make a focused adjustment and try again.\
"""

INSTANCE_TEMPLATES: dict[str, str] = {
    "minimal": "{{task}}",
    "workflow": f"{{{{task}}}}\n\n{_WORKFLOW_BLOCK}",
    "workflow-generic": f"{{{{task}}}}\n\n{_WORKFLOW_BLOCK_GENERIC}",
}

# Production default: generic 3-step workflow, works across SWE-bench, TerminalBench, and similar.
DEFAULT_TEMPLATE = "workflow-generic"

# Default model: gpt-5.4-mini via OpenAI. A recipe overrides `agent.llm_config`
# with whatever it needs — there is intentionally no model lookup table.
DEFAULT_MODEL = "gpt-5.4-mini"


def make_agent_config(
    llm_config: LLMConfig | None = None,
    template: str = DEFAULT_TEMPLATE,
) -> GennyConfig:
    """Helper for the canonical ``GENNY_CONFIGS`` entries.

    When ``llm_config`` is omitted, builds a plain ``LLMConfig(model_name=DEFAULT_MODEL)``
    — no ``reasoning_effort``, no ``interleaved_thinking``. Recipes that want
    extended thinking should pass an explicit ``LLMConfig`` choosing the
    off/once/always mode from the LLMConfig docstring. (Earlier revisions of
    this helper set ``interleaved_thinking=True`` here unconditionally, but
    with ``reasoning_effort=None`` that was a silent no-op — the LLMConfig
    validator added in PR#430 now rejects that combination outright.)

    Budget caps moved to Episode/Experiment: pass `max_steps` (and
    forthcoming `max_cost_usd`) on `Experiment(...)` instead of on
    GennyConfig. Genny reads the live budget via `recorder.budget`
    (stashed by the base Agent.run) for graceful self-stop + in-prompt
    display.
    """
    return GennyConfig(
        llm_config=llm_config or LLMConfig(model_name=DEFAULT_MODEL),
        system_prompt=SYSTEM_PROMPT,
        goal_template=INSTANCE_TEMPLATES[template],
        flat_history=True,
        step_prompt="",
        max_format_errors=3,
    )


GENNY_CONFIGS: ConfigRegistry[GennyConfig] = ConfigRegistry(
    {
        "default": make_agent_config(),
        # swe: SWE-bench-style debugging rewards deliberation at turn start
        # over re-thinking after every tool call. Use the "once" cadence
        # (``reasoning_effort="medium"`` + ``interleaved_thinking=False``)
        # instead of the helper's "always" cadence — Auto-CUBE matrix
        # 2026-05-20 found no measurable accuracy gain from per-step
        # thinking on the 4-step Reproduce → Explore → Fix → Verify
        # workflow, and "once" is ~2× faster wall-clock per episode.
        "swe": make_agent_config(
            llm_config=LLMConfig(
                model_name=DEFAULT_MODEL,
                reasoning_effort="medium",
                interleaved_thinking=False,
            ),
            template="workflow",
        ),
    }
)

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

from cube_harness.agents.genny import BudgetConfig, GennyConfig
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
    max_actions: int = 150,
    cost_limit: float = 1.0,
) -> GennyConfig:
    """Genny is a multi-step tool-use agent, so the default-built LLMConfig opts
    into ``interleaved_thinking=True``: when a caller flips on Anthropic
    extended thinking via ``reasoning_effort``, the model thinks after every
    tool result rather than only on step 0 (see auto-fix(412)).

    **Recipes that override ``agent.llm_config`` with their own ``LLMConfig``
    must pass ``interleaved_thinking=True`` themselves** — the raw
    ``LLMConfig`` default is False (matches the Anthropic provider default).
    """
    return GennyConfig(
        llm_config=llm_config or LLMConfig(model_name=DEFAULT_MODEL, interleaved_thinking=True),
        system_prompt=SYSTEM_PROMPT,
        goal_template=INSTANCE_TEMPLATES[template],
        flat_history=True,
        step_prompt="",
        max_format_errors=3,
        budget=BudgetConfig(max_actions=max_actions, cost_limit=cost_limit),
    )


GENNY_CONFIGS: ConfigRegistry[GennyConfig] = ConfigRegistry(
    {
        "default": make_agent_config(),
        "swe": make_agent_config(template="workflow"),
    }
)

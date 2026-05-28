"""The Investigator defaults to the subscription `claude -p` driver.

Switching the default avoids spending Anthropic API credits on long Auto-CUBE
batches by routing through the user's Claude subscription instead. The SDK
driver remains selectable via `--driver claude-code-sdk` / by passing one
explicitly to `InvestigationConfig`.
"""

from cube_harness.analyze.investigator.agent_driver import TerminalClaudeDriver
from cube_harness.analyze.investigator.core import InvestigationConfig, _resolve_driver


def test_investigation_config_defaults_to_terminal() -> None:
    assert isinstance(InvestigationConfig().driver, TerminalClaudeDriver)


def test_resolve_driver_none_falls_back_to_terminal() -> None:
    assert isinstance(_resolve_driver(None), TerminalClaudeDriver)

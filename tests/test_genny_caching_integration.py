"""Integration tests: Genny prompt caching with the real Anthropic API.

These tests make live API calls and are NOT part of the standard CI suite.

Requirements:
  - ANTHROPIC_API_KEY must be set.

Run:
  pytest tests/test_genny_caching_integration.py -v -s
  pytest tests/test_genny_caching_integration.py -v -s -k mode_a
"""

import os

import pytest
from cube.core import ActionSchema, Observation

from cube_harness.agents.genny import Genny, GennyConfig
from cube_harness.llm import LLMConfig

# claude-haiku-4-5-20251001 does NOT support prompt caching on the Anthropic API
# (cache_creation_input_tokens is always 0 regardless of cache_control).
# Use Sonnet 4.5 which does support it.
_MODEL = "claude-sonnet-4-5-20250929"

# Observation long enough to push the cached prefix above Anthropic's 1 024-token floor.
# ~1 300 tokens of simulated web-page content.
_LONG_OBS = """\
<html>
<head><title>ServiceNow — Create Incident</title></head>
<body>
<nav id="main-nav">
  <ul>
    <li><a href="/home">Home</a></li>
    <li><a href="/incidents">Incidents</a></li>
    <li><a href="/tasks">Tasks</a></li>
    <li><a href="/knowledge">Knowledge Base</a></li>
    <li><a href="/reports">Reports</a></li>
  </ul>
</nav>

<div id="workspace">
  <h1>Create New Incident</h1>
  <form id="incident-form" method="post" action="/api/incidents">
    <div class="field-group">
      <label for="caller">Caller *</label>
      <input id="caller" name="caller" type="text" placeholder="Search users..." required />
    </div>
    <div class="field-group">
      <label for="category">Category *</label>
      <select id="category" name="category" required>
        <option value="">-- Select --</option>
        <option value="hardware">Hardware</option>
        <option value="software">Software</option>
        <option value="network">Network</option>
        <option value="security">Security</option>
      </select>
    </div>
    <div class="field-group">
      <label for="subcategory">Subcategory</label>
      <select id="subcategory" name="subcategory">
        <option value="">-- Select --</option>
      </select>
    </div>
    <div class="field-group">
      <label for="priority">Priority *</label>
      <select id="priority" name="priority" required>
        <option value="1">1 - Critical</option>
        <option value="2">2 - High</option>
        <option value="3" selected>3 - Moderate</option>
        <option value="4">4 - Low</option>
        <option value="5">5 - Planning</option>
      </select>
    </div>
    <div class="field-group">
      <label for="short-description">Short Description *</label>
      <input id="short-description" name="short_description" type="text"
             maxlength="160" placeholder="Brief description of the issue" required />
    </div>
    <div class="field-group">
      <label for="description">Description</label>
      <textarea id="description" name="description" rows="6"
                placeholder="Detailed description of the issue..."></textarea>
    </div>
    <div class="field-group">
      <label for="assignment-group">Assignment Group</label>
      <input id="assignment-group" name="assignment_group" type="text"
             placeholder="Search groups..." />
    </div>
    <div class="field-group">
      <label for="assigned-to">Assigned To</label>
      <input id="assigned-to" name="assigned_to" type="text"
             placeholder="Search agents..." />
    </div>
    <div class="field-group">
      <label for="impact">Impact</label>
      <select id="impact" name="impact">
        <option value="1">1 - High</option>
        <option value="2" selected>2 - Medium</option>
        <option value="3">3 - Low</option>
      </select>
    </div>
    <div class="field-group">
      <label for="urgency">Urgency</label>
      <select id="urgency" name="urgency">
        <option value="1">1 - High</option>
        <option value="2" selected>2 - Medium</option>
        <option value="3">3 - Low</option>
      </select>
    </div>
    <div class="actions">
      <button id="submit-btn" type="submit">Submit</button>
      <button id="save-draft-btn" type="button">Save Draft</button>
      <button id="cancel-btn" type="button">Cancel</button>
    </div>
  </form>
</div>

<div id="related-records">
  <h2>Related Records</h2>
  <table>
    <thead>
      <tr>
        <th>Number</th><th>Type</th><th>State</th><th>Opened</th><th>Short Description</th>
      </tr>
    </thead>
    <tbody id="related-tbody">
      <tr><td colspan="5">No related records found.</td></tr>
    </tbody>
  </table>
</div>
</body>
</html>
"""

_SHORT_OBS = "<result>Action completed successfully. Form is ready for the next step.</result>"


def _skip_if_no_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")


def _action_schemas() -> list[ActionSchema]:
    return [
        ActionSchema(
            name="click",
            description="Click an element on the page.",
            parameters={
                "type": "object",
                "properties": {"element_id": {"type": "string", "description": "DOM id of the element."}},
                "required": ["element_id"],
            },
        ),
        ActionSchema(
            name="type_text",
            description="Type text into a focused input field.",
            parameters={
                "type": "object",
                "properties": {
                    "element_id": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["element_id", "text"],
            },
        ),
        ActionSchema(
            name="final_answer",
            description="Submit the final answer when the task is complete.",
            parameters={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        ),
    ]


def _llm_cfg() -> LLMConfig:
    return LLMConfig(
        model_name=_MODEL,
        max_completion_tokens=256,
        set_cache_control="auto",
        # Prevent native tool-use responses so the history stays as plain
        # assistant text, avoiding the Anthropic tool_use/tool_result pairing
        # requirement in multi-step tests.  Caching behaviour is independent
        # of whether tool calls are made.
        tool_choice="none",
    )


@pytest.mark.live_api
class TestModeACaching:
    """Mode A (raw history): cache breakpoint at system + last assistant.
    Cross-step cache hits should appear from step 2 onwards.
    """

    def test_cache_read_tokens_nonzero_by_step2(self) -> None:
        _skip_if_no_key()
        config = GennyConfig(llm_config=_llm_cfg())
        agent = Genny(config=config, action_schemas=_action_schemas())

        # Step 0 — long obs becomes the goal; writes the system+goal prefix to cache.
        out0 = agent.step(Observation.from_text(_LONG_OBS))
        act0 = next(c for c in out0.llm_calls if c.tag == "act")
        print(
            f"\n[step 0] prompt={act0.usage.prompt_tokens} cached={act0.usage.cached_tokens} "
            f"cache_write={act0.usage.cache_creation_tokens} cost=${act0.usage.cost:.5f}"
        )

        # Step 1 — adds step 0's response to history; should hit the cache on system+goal.
        out1 = agent.step(Observation.from_text(_SHORT_OBS))
        act1 = next(c for c in out1.llm_calls if c.tag == "act")
        print(
            f"[step 1] prompt={act1.usage.prompt_tokens} cached={act1.usage.cached_tokens} "
            f"cache_write={act1.usage.cache_creation_tokens} cost=${act1.usage.cost:.5f}"
        )

        assert act1.usage.cached_tokens > 0, (
            f"Expected cached_tokens > 0 at step 1, got {act1.usage.cached_tokens}. "
            f"Prompt tokens: {act1.usage.prompt_tokens}. "
            "Hint: prefix may be below the 1024-token floor for Haiku."
        )

    def test_cached_tokens_grow_with_history(self) -> None:
        """Each step caches more tokens than the previous step."""
        _skip_if_no_key()
        config = GennyConfig(llm_config=_llm_cfg())
        agent = Genny(config=config, action_schemas=_action_schemas())

        cached = []
        agent.step(Observation.from_text(_LONG_OBS))
        for _ in range(3):
            out = agent.step(Observation.from_text(_SHORT_OBS))
            act = next(c for c in out.llm_calls if c.tag == "act")
            cached.append(act.usage.cached_tokens)
            print(f"  cached={act.usage.cached_tokens} prompt={act.usage.prompt_tokens}")

        # Each step should cache at least as many tokens as the previous.
        for i in range(1, len(cached)):
            assert cached[i] >= cached[i - 1], (
                f"Cached tokens went backwards: step {i - 1}={cached[i - 1]}, step {i}={cached[i]}"
            )


@pytest.mark.live_api
class TestModeBCaching:
    """Mode B (rolling summaries): separate summary + action messages ensure
    byte-stable summaries across steps. Cross-step hits extend through all prior summaries.
    """

    def test_cache_read_tokens_nonzero_by_step2(self) -> None:
        _skip_if_no_key()
        config = GennyConfig(
            llm_config=_llm_cfg(),
            enable_summarize=True,
            summarize_llm_config=_llm_cfg(),
        )
        agent = Genny(config=config, action_schemas=_action_schemas())

        out0 = agent.step(Observation.from_text(_LONG_OBS))
        act0 = next(c for c in out0.llm_calls if c.tag == "act")
        sum0 = next(c for c in out0.llm_calls if c.tag == "summary")
        print(
            f"\n[step 0] sum cached={sum0.usage.cached_tokens} | "
            f"act cached={act0.usage.cached_tokens} cost=${act0.usage.cost:.5f}"
        )

        out1 = agent.step(Observation.from_text(_SHORT_OBS))
        act1 = next(c for c in out1.llm_calls if c.tag == "act")
        sum1 = next(c for c in out1.llm_calls if c.tag == "summary")
        print(f"[step 1] sum cached={sum1.usage.cached_tokens} | act cached={act1.usage.cached_tokens}")

        # Act pass should have a cache hit from the sum pass's write on the shared prefix.
        total_cached = act1.usage.cached_tokens + sum1.usage.cached_tokens
        assert total_cached > 0, (
            f"Expected some cached tokens in step 1 (sum+act), got {total_cached}. "
            f"sum1 prompt={sum1.usage.prompt_tokens}, act1 prompt={act1.usage.prompt_tokens}."
        )


@pytest.mark.live_api
class TestFlatModeCaching:
    """Flat mode: append-only linear history; identical cache behaviour to Mode A."""

    def test_cache_read_tokens_nonzero_by_step2(self) -> None:
        _skip_if_no_key()
        config = GennyConfig(llm_config=_llm_cfg(), flat_history=True)
        agent = Genny(config=config, action_schemas=_action_schemas())

        agent.step(Observation.from_text(_LONG_OBS))
        out1 = agent.step(Observation.from_text(_SHORT_OBS))
        act1 = next(c for c in out1.llm_calls if c.tag == "act")
        print(f"\n[flat step 1] cached={act1.usage.cached_tokens} prompt={act1.usage.prompt_tokens}")

        assert act1.usage.cached_tokens > 0, f"Expected cached_tokens > 0 at step 1, got {act1.usage.cached_tokens}."

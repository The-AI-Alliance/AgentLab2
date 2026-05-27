"""
Legacy Generic Agent implementation.

This module provides a GenericAgent that reproduces the behavior of the old
agentlab GenericAgent using the new cube-harness framework abstractions.

The agent uses text-based prompting with XML-like tags for structured output,
supporting features like:
- Multi-step planning with <plan> and <step> tags
- Memory storage with <memory> tags
- Chain-of-thought reasoning with <think> tags
- Draft-then-criticise pattern with <action_draft> and <criticise> tags
- History management with shrinking to fit token limits

This implementation aims to produce identical prompts to the original agentlab
GenericAgent while using the new cube-harness Action/Observation abstractions.
"""

import logging
import re
from typing import Any, Callable, Literal

from cube.core import Action, ActionSchema, ImageContent, Observation, TypedBaseModel
from cube.task import STOP_ACTION
from PIL import Image
from pydantic import Field

from cube_harness.agent import Agent, AgentConfig
from cube_harness.core import AgentOutput, LLMCall
from cube_harness.llm import LLMConfig, Message, Prompt
from cube_harness.utils import parse_actions

logger = logging.getLogger(__name__)


# ============================================================================
# Content Type Name Mappings (used for observation parsing)
# ============================================================================

CONTENT_TYPE_NAMES: dict[str, set[str]] = {
    "html": {"pruned_html", "raw_html", "html"},
    "axtree": {"axtree_txt", "axtree", "accessibility_tree"},
    "error": {"error", "last_action_error"},
    "focused": {"focused_element", "focused_element_bid"},
    "tabs": {"tabs", "open_pages"},
}


def _build_axtree_notes(flags: "ObsFlags") -> str:
    """Build note prefix for AXTree based on observation flags."""
    notes = """\
Note: [bid] is the unique alpha-numeric identifier at the beginning of lines for each element in the AXTree. Always use bid to refer to elements in your actions.

"""
    if flags.extract_coords == "center":
        notes += """\
Note: center coordinates are provided in parenthesis and are relative to the top left corner of the page.

"""
    elif flags.extract_coords == "box":
        notes += """\
Note: bounding box of each object are provided in parenthesis and are relative to the top left corner of the page.

"""

    if flags.filter_visible_elements_only:
        notes += """\
Note: only elements that are visible in the viewport are presented. You might need to scroll the page, or open tabs or menus to see more.

"""

    if flags.extract_visible_tag:
        notes += """\
Note: You can only interact with visible elements. If the "visible" tag is not
present, the element is not visible on the page.

"""

    return notes


# ============================================================================
# HTML Tag Parsing Utilities
# ============================================================================


def parse_html_tags(text: str, keys: list[str], optional_keys: list[str] | None = None) -> dict[str, str | int]:
    """
    Parse XML-like tags from text and return their contents.

    Args:
        text: The text to parse
        keys: Required keys that must be present
        optional_keys: Optional keys that may or may not be present

    Returns:
        Dictionary mapping tag names to their contents

    Raises:
        ValueError: If a required key is missing
    """
    optional_keys = optional_keys or []
    result = {}

    all_keys = keys + optional_keys
    for key in all_keys:
        pattern = rf"<{key}>(.*?)</{key}>"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            # Join multiple matches with newlines
            result[key] = "\n".join(m.strip() for m in matches)
        elif key in keys:
            raise ValueError(f"Required tag <{key}> not found in response")

    return result


# ============================================================================
# Observation Flags
# ============================================================================


class ObsFlags(TypedBaseModel):
    """Flags controlling what observation content to include in prompts."""

    use_html: bool = True
    use_ax_tree: bool = False
    use_tabs: bool = False
    use_focused_element: bool = False
    use_error_logs: bool = False
    use_history: bool = False
    use_past_error_logs: bool = False  # Show errors in history steps
    use_action_history: bool = False
    use_think_history: bool = False
    use_memory_history: bool = True  # Note: old code didn't have this, memories shown if not None
    use_diff: bool = False  # Not implemented, for compatibility
    html_type: Literal["pruned_html", "raw_html"] = "pruned_html"
    use_screenshot: bool = True
    use_som: bool = False  # Set of Marks overlay on screenshot
    extract_visible_tag: bool = False
    extract_clickable_tag: bool = False
    extract_coords: Literal["False", "center", "box"] = "False"
    filter_visible_elements_only: bool = False
    openai_vision_detail: Literal["low", "high", "auto"] = "auto"
    filter_with_bid_only: bool = False
    filter_som_only: bool = False


# ============================================================================
# Action Flags
# ============================================================================


class ActionFlags(TypedBaseModel):
    """Flags controlling action space behavior."""

    multiaction: bool = False


# ============================================================================
# Prompt Flags
# ============================================================================


class GenericPromptFlags(TypedBaseModel):
    """
    Flags used to control prompt construction behavior.

    Attributes:
        obs: Observation rendering flags
        action: Action space formatting flags
        use_plan: Ask the LLM to provide a multi-step plan
        use_criticise: Ask the LLM to draft and criticise actions
        use_thinking: Enable chain-of-thought reasoning
        use_memory: Enable memory storage between steps
        use_concrete_example: Include a concrete example in the prompt
        use_abstract_example: Include an abstract template example
        use_hints: Add human-engineered hints to the prompt
        enable_chat: Enable chat mode vs goal mode
        max_prompt_tokens: Maximum tokens allowed in the prompt
        be_cautious: Instruct the agent to be cautious
        extra_instructions: Extra instructions for the agent
    """

    obs: ObsFlags = Field(default_factory=ObsFlags)
    action: ActionFlags = Field(default_factory=ActionFlags)
    use_plan: bool = False
    use_criticise: bool = False
    use_thinking: bool = False
    use_memory: bool = False
    use_concrete_example: bool = True
    use_abstract_example: bool = False
    use_hints: bool = False
    enable_chat: bool = False
    max_prompt_tokens: int | None = None
    be_cautious: bool = True
    extra_instructions: str | None = None
    max_trunc_itr: int = 20


# ============================================================================
# Prompt Elements
# ============================================================================


def _resolve_visibility(visible: bool | Callable[[], bool]) -> bool:
    """Resolve visibility value, handling both static bools and callables."""
    return visible() if callable(visible) else visible


class PromptElement:
    """Base class for prompt elements with visibility control."""

    def __init__(self, visible: bool | Callable[[], bool] = True):
        self._visible = visible

    @property
    def visible(self) -> bool:
        return _resolve_visibility(self._visible)

    @property
    def prompt(self) -> str:
        if not self.visible:
            return ""
        return self._prompt

    @property
    def _prompt(self) -> str:
        return ""

    @property
    def abstract_ex(self) -> str:
        if not self.visible:
            return ""
        return self._abstract_ex

    @property
    def _abstract_ex(self) -> str:
        return ""

    @property
    def concrete_ex(self) -> str:
        if not self.visible:
            return ""
        return self._concrete_ex

    @property
    def _concrete_ex(self) -> str:
        return ""

    def parse_answer(self, text: str) -> dict[str, Any]:
        """Parse the LLM response for this element's tags."""
        return {}


class TagPromptElement(PromptElement):
    """Data-driven prompt element for XML tag-based content.

    Simplifies creating prompt elements that follow the common pattern of:
    - No direct prompt (content is in examples)
    - Abstract and concrete examples with XML tags
    - Parsing optional tags from LLM response
    """

    def __init__(
        self,
        abstract_content: str,
        concrete_content: str,
        parse_keys: list[str],
        visible: bool | Callable[[], bool] = True,
    ):
        super().__init__(visible)
        self._abstract_content = abstract_content
        self._concrete_content = concrete_content
        self._parse_keys = parse_keys

    @property
    def _abstract_ex(self) -> str:
        return self._abstract_content

    @property
    def _concrete_ex(self) -> str:
        return self._concrete_content

    def parse_answer(self, text: str) -> dict[str, Any]:
        try:
            return parse_html_tags(text, keys=[], optional_keys=self._parse_keys)
        except ValueError:
            return {}


class Trunkater:
    """
    Shrinkable content that truncates from the bottom after a certain number of iterations.
    """

    def __init__(
        self,
        content: str,
        visible: bool | Callable[[], bool] = True,
        shrink_speed: float = 0.3,
        start_trunkate_iteration: int = 10,
    ):
        self._content = content
        self._visible = visible
        self.shrink_speed = shrink_speed
        self.start_trunkate_iteration = start_trunkate_iteration
        self.shrink_calls = 0
        self.deleted_lines = 0

    @property
    def visible(self) -> bool:
        return _resolve_visibility(self._visible)

    @property
    def content(self) -> str:
        if not self.visible:
            return ""
        return self._content

    def shrink(self) -> None:
        """Shrink by removing lines from the bottom after start_trunkate_iteration calls."""
        if self.visible and self.shrink_calls >= self.start_trunkate_iteration:
            lines = self._content.splitlines()
            new_line_count = int(len(lines) * (1 - self.shrink_speed))
            self.deleted_lines += len(lines) - new_line_count
            self._content = "\n".join(lines[:new_line_count])
            self._content += f"\n... Deleted {self.deleted_lines} lines to reduce prompt size."
        self.shrink_calls += 1


class ShrinkableObservation:
    """
    Observation content with shrinkable HTML and AXTree components.
    """

    def __init__(
        self,
        html_content: str | None,
        axtree_content: str | None,
        error_content: str | None,
        focused_element: str | None,
        tabs_content: str | None,
        flags: ObsFlags,
    ):
        self.flags = flags

        # Create shrinkable components with appropriate start iterations
        # HTML starts truncating earlier (iteration 5) than AXTree (iteration 10)
        self.html = Trunkater(
            content=html_content or "",
            visible=lambda: flags.use_html and bool(html_content),
            shrink_speed=0.3,
            start_trunkate_iteration=5,
        )
        self.axtree = Trunkater(
            content=axtree_content or "",
            visible=lambda: flags.use_ax_tree and bool(axtree_content),
            shrink_speed=0.3,
            start_trunkate_iteration=10,
        )

        self.error_content = error_content
        self.focused_element = focused_element
        self.tabs_content = tabs_content

    def shrink(self) -> None:
        """Shrink HTML and AXTree content."""
        self.html.shrink()
        self.axtree.shrink()

    @property
    def prompt(self) -> str:
        """Build the observation prompt with all components."""
        parts = []

        # Tabs
        if self.flags.use_tabs and self.tabs_content:
            parts.append(f"\n## Currently open tabs:\n{self.tabs_content}\n")

        # HTML
        if self.html.visible and self.html.content:
            visible_note = ""
            if self.flags.filter_visible_elements_only:
                visible_note = """\
Note: only elements that are visible in the viewport are presented. You might need to scroll the page, or open tabs or menus to see more.

"""
            parts.append(f"\n## HTML:\n{visible_note}{self.html.content}\n")

        # AXTree
        if self.axtree.visible and self.axtree.content:
            axtree_notes = _build_axtree_notes(self.flags)
            parts.append(f"\n## AXTree:\n{axtree_notes}{self.axtree.content}\n")

        # Focused element
        if self.flags.use_focused_element:
            if self.focused_element:
                parts.append(f"\n## Focused element:\nbid={repr(self.focused_element)}\n")
            else:
                parts.append("\n## Focused element:\nNone\n")

        # Error
        if self.flags.use_error_logs and self.error_content:
            parts.append(f"\n## Error from previous action:\n{self.error_content}\n")

        return "".join(parts)


class Think(PromptElement):
    """Chain-of-thought reasoning element."""

    @property
    def _prompt(self) -> str:
        return """\
Always think carefully before performing any action. You MUST write your reasoning
as text content in your response before calling any tool. Analyze the current state
of the page, consider the effect of your previous actions, and reason step by step
about what to do next. Your text reasoning will be recorded and shown in the history
of subsequent steps, so make it informative.
"""

    @property
    def _abstract_ex(self) -> str:
        return """
(Write your reasoning as text content before calling tools)
Think step by step. If you need to make calculations such as coordinates, write them here. Describe the effect
that your previous action had on the current content of the page. Then call the appropriate tool(s).
"""

    @property
    def _concrete_ex(self) -> str:
        return """
From previous action I tried to set the value of year to "2022",
using select_option, but it doesn't appear to be in the form. It may be a
dynamic dropdown, I will try using click with the bid "a324" and look at the
response from the page.
(Then call the browser_click tool with bid="a324")
"""

    def parse_answer(self, text: str) -> dict[str, Any]:
        # Try to extract <think> tags if the model produced them
        try:
            return parse_html_tags(text, keys=["think"], optional_keys=[])
        except ValueError:
            # No <think> tags — thinking comes from reasoning_content instead
            return {}


class Plan(PromptElement):
    """Multi-step planning element."""

    def __init__(self, previous_plan: str, plan_step: int, visible: bool | Callable[[], bool] = True):
        super().__init__(visible)
        self.previous_plan = previous_plan
        self.plan_step = plan_step

    @property
    def _prompt(self) -> str:
        # Note: "# Plan:" with colon for compatibility
        return f"""
# Plan:

You just executed step {self.plan_step} of the previously proposed plan:\n{self.previous_plan}\n
After reviewing the effect of your previous actions, verify if your plan is still
relevant and update it if necessary.
"""

    @property
    def _abstract_ex(self) -> str:
        # Note: preserves original typos ("befor", "each steps") for compatibility
        return """
<plan>
Provide a multi step plan that will guide you to accomplish the goal. There
should always be steps to verify if the previous action had an effect. The plan
can be revisited at each steps. Specifically, if there was something unexpected.
The plan should be cautious and favor exploring befor submitting.
</plan>

<step>Integer specifying the step of current action
</step>
"""

    @property
    def _concrete_ex(self) -> str:
        return """
<plan>
1. fill form (failed)
    * type first name
    * type last name
2. Try to activate the form
    * click on tab 2
3. fill form again
    * type first name
    * type last name
4. verify and submit
    * verify form is filled
    * submit if filled, if not, replan
</plan>

<step>2</step>
"""

    def parse_answer(self, text: str) -> dict[str, Any]:
        try:
            result = parse_html_tags(text, keys=[], optional_keys=["plan", "step"])
            if "step" in result:
                try:
                    result["step"] = int(result["step"])
                except ValueError:
                    pass
            return result
        except ValueError:
            return {}


class Memory(TagPromptElement):
    """Memory storage element for persisting information between steps."""

    def __init__(self, visible: bool | Callable[[], bool] = True):
        super().__init__(
            abstract_content="""
<memory>
Write down anything you need to remember for next steps. You will be presented
with the list of previous memories and past actions. Some tasks require to
remember hints from previous steps in order to solve it.
</memory>
""",
            concrete_content="""
<memory>
I clicked on bid "32" to activate tab 2. The accessibility tree should mention
focusable for elements of the form at next step.
</memory>
""",
            parse_keys=["memory"],
            visible=visible,
        )


class Criticise(TagPromptElement):
    """Draft-then-criticise pattern element."""

    def __init__(self, visible: bool | Callable[[], bool] = True):
        # Note: uses "had" for compatibility with original prompt
        super().__init__(
            abstract_content="""
<action_draft>
Describe the action you intend to take and why.
</action_draft>

<criticise>
Criticise action_draft. What could be wrong with it? Enumerate reasons why it
could fail. Did your past actions had the expected effect? Make sure you're not
repeating the same mistakes.
</criticise>
""",
            concrete_content="""
<action_draft>
I will click on element with bid "32" to activate the form.
</action_draft>

<criticise>
Clicking on element "32" might not work because the element is not visible yet.
I need to explore the page to find a way to activate the form first.
</criticise>
""",
            parse_keys=["action_draft", "criticise"],
            visible=visible,
        )


class BeCautious(PromptElement):
    """Caution instruction element."""

    @property
    def _prompt(self) -> str:
        return """\

Be very cautious. Avoid submitting anything before verifying the effect of your
actions. Take the time to explore the effect of safe actions first. For example
you can fill a few elements of a form, but don't click submit before verifying
that everything was filled correctly.
"""


class Hints(PromptElement):
    """Human-engineered hints element."""

    @property
    def _prompt(self) -> str:
        return """\
Note:
* Some tasks may be game like and may require to interact with the mouse position
in x, y coordinates.
* Some text field might have auto completion. To see it, you have to type a few
characters and wait until next step.
* If you have to cut and paste, don't forget to select the text first.
* Coordinate inside an SVG are relative to it's top left corner.
* Make sure to use bid to identify elements when using commands.
* Interacting with combobox, dropdowns and auto-complete fields can be tricky,
sometimes you need to use select_option, while other times you need to use fill
or click and wait for the reaction of the page.
"""


# ============================================================================
# History Management
# ============================================================================


class HistoryStep:
    """Single step in history."""

    def __init__(
        self,
        action: str | None,
        memory: str | None,
        thought: str | None,
        error: str | None,
        flags: ObsFlags,
    ):
        self.action = action
        self.memory = memory
        self.thought = thought
        self.error = error
        self.flags = flags

    @property
    def prompt(self) -> str:
        """Build step prompt with think/action history."""
        prompt = ""

        if self.flags.use_think_history:
            if self.thought:
                prompt += f"\n<think>\n{self.thought}\n</think>\n"
            else:
                prompt += "\n[No thinking on this step]\n"

        if self.flags.use_action_history:
            prompt += f"\n<action>\n{self.action}\n</action>\n"

        # Error logs (if use_past_error_logs is enabled)
        if self.flags.use_error_logs and self.error and self.flags.use_past_error_logs:
            prompt += f"\n### Error from previous action:\n{self.error}\n"

        if self.memory is not None:
            prompt += f"\n<memory>\n{self.memory}\n</memory>\n"

        return prompt


class History:
    """Manages observation and action history with shrinking capability."""

    def __init__(
        self,
        obs_history: list[str],
        actions: list[str | None],
        memories: list[str | None],
        thoughts: list[str | None],
        errors: list[str | None],
        flags: ObsFlags,
    ):
        self.obs_history = obs_history
        self.actions = actions
        self.memories = memories
        self.thoughts = thoughts
        self.errors = errors
        self.flags = flags

        # Build history steps
        self.history_steps: list[HistoryStep] = []
        for i in range(len(actions)):
            error = errors[i] if i < len(errors) else None
            self.history_steps.append(
                HistoryStep(
                    action=actions[i],
                    memory=memories[i] if i < len(memories) else None,
                    thought=thoughts[i] if i < len(thoughts) else None,
                    error=error,
                    flags=flags,
                )
            )

    @property
    def prompt(self) -> str:
        if not self.flags.use_history:
            return ""

        if not self.history_steps:
            return ""

        prompts = ["# History of interaction with the task:\n"]
        for i, step in enumerate(self.history_steps):
            prompts.append(f"## step {i}")
            prompts.append(step.prompt)

        return "\n".join(prompts) + "\n"


# ============================================================================
# Main Prompt Builder
# ============================================================================


class MainPrompt:
    """
    Main prompt builder that assembles all prompt components.

    Supports dynamic shrinking to fit within token limits.
    """

    def __init__(
        self,
        obs_history: list[str],
        actions: list[str | None],
        memories: list[str | None],
        thoughts: list[str | None],
        errors: list[str | None],
        previous_plan: str,
        step: int,
        flags: GenericPromptFlags,
        goal: str,
        # Raw observation components for current observation (enables shrinking)
        current_obs_components: dict[str, str | None] | None = None,
    ):
        self.flags = flags
        self.goal = goal

        self.history = History(obs_history, actions, memories, thoughts, errors, flags.obs)

        # Create shrinkable observation from raw components if provided
        if current_obs_components is not None:
            self.observation = ShrinkableObservation(
                html_content=current_obs_components.get("html"),
                axtree_content=current_obs_components.get("axtree"),
                error_content=current_obs_components.get("error"),
                focused_element=current_obs_components.get("focused_element"),
                tabs_content=current_obs_components.get("tabs"),
                flags=flags.obs,
            )
        else:
            # Fallback: use last obs_history entry as non-shrinkable text
            self.observation = None
            self._current_obs_text = obs_history[-1] if obs_history else ""

        # Prompt elements (actions are handled via LLM tool calling, not prompt text)
        self.be_cautious = BeCautious(visible=lambda: flags.be_cautious and flags.action.multiaction)
        self.think = Think(visible=lambda: flags.use_thinking)
        self.hints = Hints(visible=lambda: flags.use_hints)
        self.plan = Plan(previous_plan, step, lambda: flags.use_plan)
        self.criticise = Criticise(visible=lambda: flags.use_criticise)
        self.memory = Memory(visible=lambda: flags.use_memory)

    def build_prompt(self) -> str:
        """Build the full prompt text."""
        # Instructions
        if self.flags.enable_chat:
            instructions = f"""\
# Instructions

You are a UI Assistant, your goal is to help the user perform tasks using a web browser. You can
communicate with the user via a chat, in which the user gives you instructions and in which you
can send back messages. You have access to a web browser that both you and the user can see,
and with which only you can interact via specific commands.

Review the instructions from the user, the current state of the page and all other information
to find the best possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.

## Chat messages:

{self.goal}
"""
        else:
            instructions = f"""\
# Instructions
Review the current state of the page and all other information to find the best
possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.

## Goal:
{self.goal}
"""
            if self.flags.extra_instructions:
                instructions += f"""
## Extra instructions:

{self.flags.extra_instructions}
"""

        # Current observation (use shrinkable observation if available)
        if self.observation is not None:
            obs_content = self.observation.prompt
        else:
            obs_content = self._current_obs_text

        obs_section = f"""
# Observation of current step:
{obs_content}

"""

        # Assemble main prompt
        prompt_parts = [
            instructions,
            obs_section,
            self.history.prompt,
            self.hints.prompt,
            self.be_cautious.prompt,
            self.think.prompt,
            self.plan.prompt,
            self.memory.prompt,
            self.criticise.prompt,
        ]

        # Add abstract example if enabled
        if self.flags.use_abstract_example:
            abstract_parts = [
                """
# Abstract Example

Here is an abstract version of the answer with description of the content of
each tag. Make sure you follow this structure, but replace the content with your
answer. Always write your reasoning as text, then call the tool(s):
""",
                self.think.abstract_ex,
                self.plan.abstract_ex,
                self.memory.abstract_ex,
                self.criticise.abstract_ex,
            ]
            prompt_parts.extend(abstract_parts)

        # Add concrete example if enabled
        if self.flags.use_concrete_example:
            concrete_parts = [
                """
# Concrete Example

Here is a concrete example of how to format your answer.
Write your reasoning as text first, then call the appropriate tool(s):
""",
                self.think.concrete_ex,
                self.plan.concrete_ex,
                self.memory.concrete_ex,
                self.criticise.concrete_ex,
            ]
            prompt_parts.extend(concrete_parts)

        return "".join(prompt_parts)

    def shrink(self) -> None:
        """Shrink the prompt by shrinking observation content (HTML and AXTree via Trunkater)."""
        if self.observation is not None:
            self.observation.shrink()

    def parse_answer(self, text: str) -> dict[str, Any]:
        """Parse the LLM response for all expected tags (actions are parsed from tool_calls)."""
        ans_dict: dict[str, Any] = {}
        ans_dict.update(self.think.parse_answer(text))
        ans_dict.update(self.plan.parse_answer(text))
        ans_dict.update(self.memory.parse_answer(text))
        ans_dict.update(self.criticise.parse_answer(text))
        return ans_dict


# ============================================================================
# Agent Configuration
# ============================================================================


class GenericAgentConfig(AgentConfig):
    """
    Configuration for the GenericAgent.

    Attributes:
        llm_config: Configuration for the LLM
        flags: Prompt construction flags
        max_retry: Maximum retries on parse errors
        max_actions: Maximum actions before auto-stopping
    """

    llm_config: LLMConfig
    flags: GenericPromptFlags = Field(default_factory=GenericPromptFlags)
    max_retry: int = 4
    max_actions: int = 50

    @property
    def agent_name(self) -> str:
        return f"GenericAgent-{self.llm_config.model_name}".replace("/", "_")

    system_prompt: str = """\
You are an agent trying to solve a web task based on the content of the page and
user instructions. You can interact with the page and explore, and send messages to the user.
You can produce multiple actions at once. Each time you submit actions,
they will be sent to the browser and you will receive a new page.

IMPORTANT: Always include your reasoning as text content in your response BEFORE making tool calls.
Explain what you observe on the page, what you decided to do, and why. Never respond with only
tool calls and no text — always write out your thinking first, then call the tools."""

    def make(self, action_set: list[ActionSchema] | None = None, **kwargs: Any) -> "GenericAgent":
        return GenericAgent(config=self, action_set=action_set or [])


# ============================================================================
# Generic Agent Implementation
# ============================================================================


class GenericAgent(Agent):
    """
    A generic agent using text-based prompting with XML-like tags.

    This agent reproduces the behavior of the old agentlab GenericAgent,
    featuring:
    - Separate tracking of observations, actions, memories, and thoughts
    - Multi-step planning with plan and step tracking
    - Token-based prompt shrinking when exceeding limits
    - Retry mechanism for parsing errors
    """

    name: str = "generic_agent"
    description: str = "A generic agent using text-based prompting with structured tags."
    input_content_types: list[str] = ["text/plain", "image/png", "image/jpeg"]
    output_content_types: list[str] = ["text/plain"]

    def __init__(self, config: GenericAgentConfig, action_set: list[ActionSchema]):
        self.config = config
        self.llm = config.llm_config.make()
        self.token_counter = config.llm_config.make_counter()

        # Build action set with stop action
        self.action_set = list(action_set)
        self.action_set.append(STOP_ACTION)

        # Convert action schemas to tool dicts for LLM tool calling
        self.tools: list[dict] = [a.as_dict() for a in self.action_set]

        # State tracking
        self.plan = "No plan yet"
        self.plan_step = -1
        self.memories: list[str | None] = []
        self.thoughts: list[str | None] = []
        self.actions: list[str | None] = []
        self.errors: list[str | None] = []  # Track errors for history
        self.obs_history: list[str] = []
        self.goal: str = ""
        self._actions_cnt = 0

    def step(self, obs: Observation) -> AgentOutput:
        """Process observation and produce action(s)."""
        # Check if max actions reached
        if self._actions_cnt >= self.config.max_actions:
            logger.info("Max actions reached, issuing STOP action.")
            return AgentOutput(
                actions=[Action(id="stop", name=STOP_ACTION.name, arguments={})],
                action_rationale="Reached max actions, stopping.",
            )

        # Extract observation data (text for history + raw components for shrinkable observation)
        obs_text, obs_components = self._extract_obs_data(obs)

        # Extract screenshots if enabled
        screenshots = self._extract_screenshots(obs)

        # Extract error from observation (for history)
        # This error is the result of the PREVIOUS action
        error_from_obs = self._extract_error(obs)
        if self.actions:  # If there was a previous action, record its error
            self.errors.append(error_from_obs)

        # Extract goal from first observation
        if not self.goal:
            self.goal = self._extract_goal(obs)

        # Append observation to history
        self.obs_history.append(obs_text)

        # Build main prompt with raw observation components for shrinking support
        main_prompt = MainPrompt(
            obs_history=self.obs_history,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            errors=self.errors,
            previous_plan=self.plan,
            step=self.plan_step,
            flags=self.config.flags,
            goal=self.goal,
            current_obs_components=obs_components,
        )

        # Fit prompt to token limit
        prompt_text = self._fit_tokens(main_prompt)

        # Build LLM messages with optional screenshot
        messages = self._build_messages(prompt_text, screenshots)

        # Call LLM with retry
        ans_dict, llm_call = self._call_llm_with_retry(messages, main_prompt)

        # Update state from response (match old behavior: no auto-increment)
        self.plan = ans_dict.get("plan", self.plan)
        self.plan_step = ans_dict.get("step", self.plan_step)

        thoughts = ans_dict.get("thoughts")
        actions: list[Action] = ans_dict.get("actions")  # type: ignore
        action_str = str(actions) if actions else None
        self.actions.append(action_str)
        self.memories.append(ans_dict.get("memory"))
        self.thoughts.append(thoughts)
        self._actions_cnt += 1

        # Build output
        return AgentOutput(actions=actions, llm_calls=[llm_call] if llm_call else [], action_rationale=thoughts)

    def _extract_obs_data(self, obs: Observation) -> tuple[str, dict[str, str | None]]:
        """Extract both formatted text and raw components from observation in one pass.

        Returns:
            tuple: (formatted_text for history, raw_components dict for shrinkable observation)
        """
        obs_flags = self.config.flags.obs
        parts: list[str] = []
        components: dict[str, str | None] = {
            "html": None,
            "axtree": None,
            "error": None,
            "focused_element": None,
            "tabs": None,
        }

        for content in obs.contents:
            if isinstance(content.data, str):
                name_lower = (content.name or "").lower()

                # HTML content
                if any(n in name_lower for n in CONTENT_TYPE_NAMES["html"]):
                    components["html"] = content.data
                    if obs_flags.use_html:
                        parts.append(f"\n## HTML:\n{content.data}\n")
                    continue

                # AXTree content
                if any(n in name_lower for n in CONTENT_TYPE_NAMES["axtree"]):
                    components["axtree"] = content.data
                    if obs_flags.use_ax_tree:
                        axtree_notes = _build_axtree_notes(obs_flags)
                        parts.append(f"\n## AXTree:\n{axtree_notes}{content.data}\n")
                    continue

                # Error content
                if any(n in name_lower for n in CONTENT_TYPE_NAMES["error"]):
                    components["error"] = content.data if content.data else None
                    if obs_flags.use_error_logs and content.data:
                        parts.append(f"\n## Error from previous action:\n{content.data}\n")
                    continue

                # Focused element
                if any(n in name_lower for n in CONTENT_TYPE_NAMES["focused"]):
                    components["focused_element"] = content.data if content.data else None
                    if obs_flags.use_focused_element:
                        if content.data:
                            parts.append(f"\n## Focused element:\nbid={repr(content.data)}\n")
                        else:
                            parts.append("\n## Focused element:\nNone\n")
                    continue

                # Tabs content
                if any(n in name_lower for n in CONTENT_TYPE_NAMES["tabs"]):
                    components["tabs"] = content.data
                    if obs_flags.use_tabs:
                        parts.append(f"\n## Currently open tabs:\n{content.data}\n")
                    continue

                # Other text content (goal, etc.) - skip, handled separately

        return "".join(parts), components

    def _extract_screenshots(self, obs: Observation) -> list[Image.Image]:
        """Extract screenshot images from observation if use_screenshot is enabled."""
        if not self.config.flags.obs.use_screenshot:
            return []

        screenshots = []
        for content in obs.contents:
            if isinstance(content.data, Image.Image):
                screenshots.append(content.data)
        return screenshots

    def _build_messages(self, prompt_text: str, screenshots: list[Image.Image]) -> list[dict | Message]:
        """Build LLM messages with optional screenshots.

        Args:
            prompt_text: The text prompt to send
            screenshots: List of screenshot images to include

        Returns:
            List of message dicts for the LLM
        """
        messages: list[dict | Message] = [{"role": "system", "content": self.config.system_prompt}]

        if screenshots:
            # Add screenshot header text to match old dp.Observation.add_screenshot() behavior
            if self.config.flags.obs.use_som:
                screenshot_header = "\n## Screenshot:\nHere is a screenshot of the page, it is annotated with bounding boxes and corresponding bids:"
            else:
                screenshot_header = "\n## Screenshot:\nHere is a screenshot of the page:"

            # Build multimodal message with text and images
            user_content: list[dict] = [{"type": "text", "text": prompt_text + screenshot_header}]
            for screenshot in screenshots:
                # Convert PIL Image to base64
                image_content = ImageContent(data=screenshot, name="screenshot")
                image_base64 = image_content.as_base64_image_str(screenshot)
                user_content.append({"type": "image_url", "image_url": {"url": image_base64}})
            messages.append({"role": "user", "content": user_content})
        else:
            # Text-only message
            messages.append({"role": "user", "content": prompt_text})

        return messages

    def _extract_goal(self, obs: Observation) -> str:
        """Extract goal from observation contents."""
        for content in obs.contents:
            if isinstance(content.data, str):
                # Look for goal markers
                if "goal" in (content.name or "").lower():
                    return content.data
                # Check if it looks like a goal (first text content)
                if content.name is None and len(content.data) < 500:
                    return content.data
        return "Complete the task."

    def _extract_error(self, obs: Observation) -> str | None:
        """Extract error from observation contents (for history tracking)."""
        for content in obs.contents:
            if isinstance(content.data, str):
                name_lower = (content.name or "").lower()
                # Look for error markers
                if "error" in name_lower or "last_action_error" in name_lower:
                    return content.data if content.data else None
        return None

    def _fit_tokens(self, main_prompt: MainPrompt) -> str:
        """Fit prompt to token limit by shrinking observation if needed.

        Matches old fit_tokens behavior: loops up to max_trunc_itr, shrinking
        observation content (HTML/AXTree via Trunkater) each iteration.
        """
        max_tokens = self.config.flags.max_prompt_tokens
        if max_tokens is None:
            return main_prompt.build_prompt()

        for _ in range(self.config.flags.max_trunc_itr):
            prompt_text = main_prompt.build_prompt()
            tokens = self.token_counter(messages=[{"role": "user", "content": prompt_text}])
            if tokens <= max_tokens:
                return prompt_text
            main_prompt.shrink()

        # Log if still over limit after max iterations
        prompt_text = main_prompt.build_prompt()
        tokens = self.token_counter(messages=[{"role": "user", "content": prompt_text}])
        if tokens > max_tokens:
            logger.info(
                f"After {self.config.flags.max_trunc_itr} shrink iterations, prompt is still "
                f"{tokens} tokens (greater than {max_tokens}). Returning as is."
            )

        return prompt_text

    def _call_llm_with_retry(
        self, messages: list[dict | Message], main_prompt: MainPrompt
    ) -> tuple[dict[str, Any], LLMCall | None]:
        """Call LLM with retry on parse errors."""
        prompt = Prompt(messages=messages, tools=self.tools)
        last_error = None
        llm_call = None

        for retry_num in range(self.config.max_retry + 1):
            try:
                llm_response = self.llm(prompt)
                llm_call = LLMCall(
                    llm_config=self.config.llm_config,
                    prompt=prompt,
                    output=llm_response.message,
                    usage=llm_response.usage,
                )

                # Extract text response for parsing think/plan/memory/criticise tags
                text_response = llm_response.message.content or ""
                if isinstance(text_response, list):
                    text_response = " ".join(
                        item.get("text", "") if isinstance(item, dict) else str(item) for item in text_response
                    )

                # Parse response for think/plan/memory/criticise tags
                ans_dict = main_prompt.parse_answer(text_response)

                # Extract extended thinking from model's native thinking (reasoning_effort)
                reasoning_content = getattr(llm_response.message, "reasoning_content", None)
                thinking_blocks = getattr(llm_response.message, "thinking_blocks", None)

                if reasoning_content:
                    if "thoughts" not in ans_dict:
                        ans_dict["thoughts"] = reasoning_content
                    else:
                        ans_dict["thoughts"] = f"{reasoning_content}\n\n{ans_dict['thoughts']}"

                if thinking_blocks:
                    thoughts = "\n".join(thinking_blocks)
                    if "thoughts" not in ans_dict:
                        ans_dict["thoughts"] = thoughts
                    else:
                        ans_dict["thoughts"] = f"{ans_dict['thoughts']}\n\n" + thoughts

                # Fallback: use text content as implicit thinking when no other
                # thinking source was found. Models often emit reasoning text
                # alongside tool calls (e.g. "I will click on...").
                if "thoughts" not in ans_dict and text_response.strip():
                    ans_dict["thoughts"] = text_response.strip()

                # Parse actions from tool calls
                actions, _ = parse_actions(llm_response.message)
                if actions:
                    # Return first action (multiaction not supported in legacy agent)
                    ans_dict["actions"] = actions
                    logger.info(f"Successfully parsed action from tool call on attempt {retry_num + 1}")
                    return ans_dict, llm_call

                # No tool calls found, retry
                last_error = "No tool calls found in response"
                if retry_num < self.config.max_retry:
                    logger.warning(f"Parse attempt {retry_num + 1} failed: {last_error}. Retrying...")
                    # Add error message for retry
                    messages.append({"role": "assistant", "content": text_response})
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Error: {last_error}. Please call one of the available tools to perform an action.",
                        }
                    )
                    prompt = Prompt(messages=messages, tools=self.tools)

            except Exception as e:
                last_error = str(e)
                logger.warning(f"LLM call attempt {retry_num + 1} failed: {e}")

        # All retries exhausted
        logger.error(f"All retry attempts exhausted. Last error: {last_error}")
        return {"actions": None}, llm_call

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        self.plan = "No plan yet"
        self.plan_step = -1
        self.memories = []
        self.thoughts = []
        self.actions = []
        self.errors = []
        self.obs_history = []
        self.goal = ""
        self._actions_cnt = 0

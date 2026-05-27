import logging
from typing import TYPE_CHECKING

from cube.core import Action, ActionSchema, Observation
from cube.task import STOP_ACTION
from litellm import Message
from termcolor import colored

from cube_harness.agent import Agent, AgentConfig, apply_description_overrides
from cube_harness.core import AgentOutput
from cube_harness.llm import LLMConfig, Prompt
from cube_harness.utils import parse_actions

if TYPE_CHECKING:
    from cube_harness.streamer import EventStreamer

logger = logging.getLogger(__name__)

# How many times to re-prompt the model within a single step when every tool
# call it emitted had malformed JSON arguments. After this many retries the step
# returns empty actions and the episode ends cleanly (never an infinite loop).
MAX_PARSE_RETRIES = 2


class ReactAgentConfig(AgentConfig):
    llm_config: LLMConfig
    can_finish: bool = True
    max_actions: int = 10
    max_obs_chars: int = 100000  # truncate long observations to M chars
    max_history_tokens: int = 120000  # compact history if it exceeds N tokens
    render_last_n_steps: int = -1  # include last N steps in prompt, if -1 - include all. For tasks with long obs.
    system_prompt: str = """
You are an expert AI Agent trained to assist users with complex web tasks.
Your role is to understand the goal, perform actions until the goal is accomplished and respond in a helpful and accurate manner.
Keep your replies brief, concise, direct and on topic. Prioritize clarity and avoid over-elaboration.
Do not express emotions or opinions."""
    react_prompt: str = """
Think along the following lines:
1. Summarize the last observation and describe the visible changes in the state.
2. Evaluate action success, explain impact on task and next steps.
3. If you see any errors in the last observation, think about it. If there is no error, just move on.
4. List next steps to move towards the goal and propose next immediate action.
Then produce the single function call that performs the proposed action. If the task is complete, produce the final step."""
    summarize_system_prompt: str = """
You are a helpful assistant that summarizes agent interaction history. Following messages is the history to summarize:"""
    summarize_prompt: str = """
Summarize the presented agent interaction history concisely.
Focus on:
- The original goal
- Key actions taken and their outcomes
- Important errors or obstacles encountered
- Current progress toward the goal
Provide a concise summary that preserves all information needed to continue the task."""

    @property
    def agent_name(self) -> str:
        return f"ReactAgent-{self.llm_config.model_name}".replace("/", "_")

    def make(self, action_set: list[ActionSchema] | None = None, **kwargs) -> "ReactAgent":
        return ReactAgent(config=self, tools=action_set or [])


class ReactAgent(Agent):
    name: str = "react_agent"
    description: str = "An agent implementing the ReAct framework for web tasks."
    input_content_types: list[str] = ["image/png", "image/jpeg", "text/plain", "application/json"]
    output_content_types: list[str] = ["application/json"]

    def __init__(self, config: ReactAgentConfig, tools: list[ActionSchema]):
        super().__init__(config)
        self.llm = config.llm_config.make()
        self.token_counter = config.llm_config.make_counter()
        # STOP (`final_step`) is always part of the task's `action_set` — it's a universal
        # `@tool_action` on the Tool base, already Anthropic-safe
        # (`{"type": "object", "properties": {}}`). We never append it manually.
        # `can_finish=False` opts the agent out of offering STOP to the LLM.
        self.tools: list[dict] = [tool.as_dict() for tool in tools]
        if not config.can_finish:
            self.tools = [t for t in self.tools if t["function"]["name"] != STOP_ACTION.name]
        apply_description_overrides(self.tools, config.description_overrides)

        self.history: list[dict | Message] = []
        self._actions_cnt = 0

    def attach_recorder(self, recorder: "EventStreamer") -> None:
        super().attach_recorder(recorder)
        self.llm.attach_recorder(recorder)

    def step(self, obs: Observation) -> AgentOutput:
        if self.max_actions_reached():
            logger.info("Max actions reached, issuing STOP action.")
            return AgentOutput(actions=[Action(id="stop", name=STOP_ACTION.name, arguments={})])
        self.history += obs.to_llm_messages()
        self.maybe_compact_history()
        self._actions_cnt += 1

        # A model can emit a tool call whose arguments are not valid JSON. Rather
        # than crash the episode, reply with a corrective `role="tool"` message —
        # this keeps the tool_call/tool_result pairing valid (strict providers
        # reject an orphaned tool_call on the next turn) and gives the model the
        # feedback it needs to self-correct. Bounded by MAX_PARSE_RETRIES so a
        # persistently-broken model degrades to a clean episode end (empty
        # actions) instead of looping forever.
        actions: list[Action] = []
        for _ in range(MAX_PARSE_RETRIES + 1):
            llm_output = self._call_llm()
            actions, malformed = parse_actions(llm_output)
            for tc in malformed:
                self.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "Your previous tool call arguments were not valid JSON. "
                        "Please retry the call with a valid JSON object.",
                    }
                )
            if actions or not malformed:
                break
            logger.warning("All tool calls had malformed JSON arguments; re-prompting the model.")
        return AgentOutput(actions=actions)

    def _call_llm(self) -> Message:
        """Render the current history, call the LLM once, append the assistant
        message to history, and return it."""
        messages = self.choose_steps_to_render(self.history)
        prompt = Prompt(messages=messages, tools=self.tools)
        prompt_tokens = self.token_counter(messages=messages)
        logger.info(f"Prompt tokens (estimated): {prompt_tokens}")
        try:
            logger.debug(f"Prompt: {prompt}")
            call = self.llm.call(prompt, tag="act")
            logger.debug(f"LLM Response: {call.output}")
        except Exception as e:
            logger.exception(colored(f"Error getting LLM response: {e}. Prompt: {prompt}", "red"))
            raise e
        usage = call.usage
        logger.info(
            f"LLM usage - prompt: {usage.prompt_tokens}, completion: {usage.completion_tokens}, "
            f"cached: {usage.cached_tokens}, cache_created: {usage.cache_creation_tokens}, cost: ${usage.cost:.4f}"
        )
        llm_output = call.output
        self.history.append(llm_output)
        return llm_output

    def choose_steps_to_render(self, history: list[dict | Message]) -> list[dict | Message]:
        """Select which parts of history to include in the prompt based on length."""
        # goal + last N messages
        return [
            dict(role="system", content=self.config.system_prompt),
            self.history[0],  # goal
            *self.history[-self.config.render_last_n_steps :],
            dict(role="user", content=self.config.react_prompt),
        ]

    def max_actions_reached(self) -> bool:
        return self._actions_cnt >= self.config.max_actions

    def maybe_compact_history(self):
        tokens = self.token_counter(messages=self.history)
        if tokens > self.config.max_history_tokens:
            logger.info("Compacting history due to length.")
            self.compact_history()
            short_tokens = self.token_counter(messages=self.history)
            logger.info(f"Compacted history from {tokens} to {short_tokens} tokens.")

    def _get_role(self, msg: dict | Message) -> str:
        if isinstance(msg, dict):
            return msg.get("role", "")
        return getattr(msg, "role", "")

    def compact_history(self):
        """
        Compact the history by summarizing the first half of messages with the LLM.
        Updates self.history in place by replacing the first half with the summary message.
        """
        midpoint = len(self.history) // 2
        # Advance past any tool messages to avoid splitting tool_call/tool_result pairs
        while midpoint < len(self.history) and self._get_role(self.history[midpoint]) == "tool":
            midpoint += 1
        if midpoint >= len(self.history):
            logger.warning("compact_history: could not find a clean split point, skipping compaction.")
            return
        first_half = self.history[:midpoint]
        second_half = self.history[midpoint:]
        messages = [
            dict(role="system", content=self.config.summarize_system_prompt),
            *first_half,
            dict(role="user", content=self.config.summarize_prompt),
        ]
        prompt = Prompt(messages=messages)
        try:
            call = self.llm.call(prompt, tag="compact")
        except Exception as e:
            logger.exception(f"Error compacting history: {e}")
            raise

        summary = call.output.content
        logger.info(f"Compacted {midpoint} messages into summary:\n{summary}")
        # Rebuild history: system + summary + remaining messages
        summary_message = dict(role="assistant", content=f"## Previous Interactions summary:\n{summary}")
        self.history = [summary_message, *second_half]

    def get_training_pairs(self) -> list[tuple[list[dict | AgentOutput], AgentOutput]]:
        input_output_pairs = []
        prev_history = []
        for msg in self.history:
            if isinstance(msg, AgentOutput):
                input_output_pairs.append((prev_history, msg))
            prev_history.append(msg)
        return input_output_pairs

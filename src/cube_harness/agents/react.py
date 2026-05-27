import logging

from cube.core import Action, ActionSchema, Observation
from cube.task import STOP_ACTION
from litellm import Message
from termcolor import colored

from cube_harness.agent import Agent, AgentConfig
from cube_harness.core import AgentOutput, LLMCall
from cube_harness.llm import LLMConfig, Prompt
from cube_harness.utils import parse_actions

logger = logging.getLogger(__name__)


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
        self.config = config
        self.llm = config.llm_config.make()
        self.token_counter = config.llm_config.make_counter()
        self.tools: list[dict] = [tool.as_dict() for tool in tools]
        if config.can_finish:
            stop_tool = STOP_ACTION.as_dict()
            # Ensure parameters has "type": "object" — some LLM APIs (e.g. Anthropic)
            # require it even for tools with no parameters.
            if not stop_tool["function"]["parameters"].get("type"):
                stop_tool["function"]["parameters"] = {"type": "object", "properties": {}}
            self.tools.append(stop_tool)

        self.history: list[dict | Message] = []
        self._actions_cnt = 0

    def step(self, obs: Observation) -> AgentOutput:
        if self.max_actions_reached():
            logger.info("Max actions reached, issuing STOP action.")
            return AgentOutput(actions=[Action(id="stop", name=STOP_ACTION.name, arguments={})])
        self.history += obs.to_llm_messages()
        self.maybe_compact_history()
        messages = self.choose_steps_to_render(self.history)
        prompt = Prompt(messages=messages, tools=self.tools)
        prompt_tokens = self.token_counter(messages=messages)
        logger.info(f"Prompt tokens (estimated): {prompt_tokens}")
        try:
            logger.debug(f"Prompt: {prompt}")
            llm_response = self.llm(prompt)
            logger.debug(f"LLM Response: {llm_response}")
        except Exception as e:
            logger.exception(colored(f"Error getting LLM response: {e}. Prompt: {prompt}", "red"))
            raise e
        usage = llm_response.usage
        logger.info(
            f"LLM usage - prompt: {usage.prompt_tokens}, completion: {usage.completion_tokens}, "
            f"cached: {usage.cached_tokens}, cache_created: {usage.cache_creation_tokens}, cost: ${usage.cost:.4f}"
        )
        llm_output = llm_response.message
        self.history.append(llm_output)
        self._actions_cnt += 1
        llm_call = LLMCall(llm_config=self.config.llm_config, prompt=prompt, output=llm_output, usage=usage)
        actions, decode_error = parse_actions(llm_output)
        return AgentOutput(actions=actions, llm_calls=[llm_call], error=decode_error)

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
            llm_response = self.llm(prompt)
        except Exception as e:
            logger.exception(f"Error compacting history: {e}")
            raise

        summary = llm_response.message.content
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

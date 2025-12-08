import logging

from termcolor import colored

from agentlab2.agent import Agent, AgentConfig
from agentlab2.core import ActionSchema, AgentOutput, Observation
from agentlab2.llm import LLM, Prompt, obs_to_messages

logger = logging.getLogger(__name__)


class ReactAgentConfig(AgentConfig):
    llm: LLM
    use_html: bool = True
    use_axtree: bool = False
    use_screenshot: bool = True
    max_actions: int = 10
    max_obs_chars: int = 100000  # truncate long observations to M chars
    max_history_tokens: int = 120000  # compact history if it exceeds N tokens
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

    def make(self, actions: list[ActionSchema]) -> "ReactAgent":
        return ReactAgent(config=self, actions=actions)


class ReactAgent(Agent):
    metadata = {
        "name": "react_agent",
        "description": "An agent implementing the ReAct framework for web tasks.",
        "input_content_types": ["image/png", "image/jpeg", "text/plain", "application/json"],
        "output_content_types": ["application/json"],
    }

    def __init__(self, config: ReactAgentConfig, actions: list[ActionSchema]):
        self.config = config
        self.llm = config.llm
        self.tools: list[dict] = [action.schema() for action in actions]
        self.history: list[dict | AgentOutput] = []

    def reset(self) -> None:
        self.history = []

    def obs_preprocess(self, obs: Observation) -> Observation:
        """
        Filter observation contents based on agent config.
        """
        obs = obs.model_copy(deep=True)
        if not self.config.use_html:
            obs.contents.pop("pruned_html", None)
            obs.contents.pop("html", None)
        if not self.config.use_axtree:
            obs.contents.pop("axtree_txt", None)
        if not self.config.use_screenshot:
            obs.contents.pop("screenshot", None)
        return obs

    def step(self, obs: Observation) -> AgentOutput:
        if self.max_actions_reached():
            logger.warning("Max actions reached, stopping agent.")
            return AgentOutput(content="Max actions reached, stopping agent.")

        obs = self.obs_preprocess(obs)
        self.history += obs_to_messages(obs)
        self.maybe_compact_history()
        messages = [
            dict(role="system", content=self.config.system_prompt),
            *self.history,
            dict(role="user", content=self.config.react_prompt),
        ]
        prompt = Prompt(messages=messages, tools=self.tools)
        try:
            logger.debug(f"Prompt: {prompt}")
            output = self.llm(prompt)
            logger.debug(f"LLM Response: {output}")
        except Exception as e:
            logger.exception(colored(f"Error getting LLM response: {e}. Prompt: {prompt}", "red"))
            raise e

        self.history.append(output)
        return output

    def max_actions_reached(self) -> bool:
        prev_actions = [msg for msg in self.history if isinstance(msg, AgentOutput) and msg.tool_calls]
        return len(prev_actions) >= self.config.max_actions

    def maybe_compact_history(self):
        tokens = self.llm.counter(messages=self.history)
        if tokens > self.config.max_history_tokens:
            logger.info("Compacting history due to length.")
            self.compact_history()
            short_tokens = self.llm.counter(messages=self.history)
            logger.info(f"Compacted history from {tokens} to {short_tokens} tokens.")

    def compact_history(self):
        """
        Compact the history by summarizing the first half of messages with the LLM.
        Updates self.history in place by replacing the first half with the summary message.
        """
        midpoint = len(self.history) // 2
        first_half = self.history[:midpoint]
        second_half = self.history[midpoint:]
        messages = [
            dict(role="system", content=self.config.summarize_system_prompt),
            *first_half,
            dict(role="user", content=self.config.summarize_prompt),
        ]
        prompt = Prompt(messages=messages)
        try:
            llm_message = self.llm(prompt)
        except Exception as e:
            logger.exception(f"Error compacting history: {e}")
            raise

        summary = llm_message.content
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

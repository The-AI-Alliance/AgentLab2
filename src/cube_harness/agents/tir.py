import json
import logging

from cube.core import Action, ActionSchema, Observation
from litellm import Message

from cube_harness.agent import Agent, AgentConfig
from cube_harness.core import AgentOutput, LLMCall
from cube_harness.llm import LLMConfig, Prompt
from cube_harness.utils import parse_actions

logger = logging.getLogger(__name__)

def tir_parse_actions_tolerant(llm_output) -> list[Action]:
    """Mirror native TIR tolerance for malformed tool-call arguments.

    Native TIR does not fail the rollout when tool-call arguments are malformed
    or not an object; it recovers to empty/default arguments.
    """
    try:
        return parse_actions(llm_output)
    except Exception:
        actions: list[Action] = []
        for tc in getattr(llm_output, "tool_calls", []) or []:
            name = getattr(getattr(tc, "function", None), "name", None)
            if not name:
                continue
            raw_args = getattr(getattr(tc, "function", None), "arguments", {})
            if isinstance(raw_args, dict):
                args = raw_args
            else:
                try:
                    parsed = json.loads(raw_args) if isinstance(raw_args, str) else {}
                    args = parsed if isinstance(parsed, dict) else {}
                except Exception:
                    args = {}
            actions.append(Action(id=getattr(tc, "id", None), name=name, arguments=args))
        return actions

class TirAgentConfig(AgentConfig):
    """TIR-style tool-calling agent.

    Prompt shape per step:
    - optional system prompt
    - accumulated conversation history:
      - initial user task observation
      - assistant tool-call messages
      - tool result messages from the environment

    No extra React instruction message is injected each turn.
    """

    llm_config: LLMConfig
    system_prompt: str = ""
    max_actions: int = 3
    min_generation_tokens: int = 256  # minimum tokens required for generation after accounting for prompt length;

    def make(self, action_set: list[ActionSchema]) -> "TirAgent":
        return TirAgent(config=self, tools=action_set)


class TirAgent(Agent):
    name: str = "tir_agent"
    description: str = "A minimal TIR-style agent that iterates over tool calls."
    input_content_types: list[str] = ["image/png", "image/jpeg", "text/plain", "application/json"]
    output_content_types: list[str] = ["application/json"]

    def __init__(self, config: TirAgentConfig, tools: list[ActionSchema]):
        self.config = config
        self.llm = config.llm_config.make()
        self.tools: list[dict] = [tool.as_dict() for tool in tools]
        self.tool_names = set([tool.name for tool in tools])
        self.token_counter = config.llm_config.make_counter()
        self.max_completion_tokens = config.llm_config.max_completion_tokens
        self.max_model_len = config.llm_config.max_model_len

        self.history: list[dict | Message] = []
        self._actions_cnt = 0

    def max_actions_reached(self) -> bool:
        return self._actions_cnt >= self.config.max_actions

    def _build_prompt_messages(self) -> list[dict | Message]:
        messages: list[dict | Message] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.extend(self.history)
        return messages

    def step(self, obs: Observation) -> AgentOutput:
        if self.max_actions_reached():
            return AgentOutput(actions=[])

        self.history += obs.to_llm_messages()
        messages = self._build_prompt_messages()
        prompt = Prompt(messages=messages, tools=self.tools)
        prompt_tokens = self.token_counter(messages=messages, tools=self.tools)
        
        remaining = self.max_model_len - prompt_tokens
        if remaining < self.config.min_generation_tokens:
            logger.warning(
                "Prompt length %d leaves only %d tokens for generation (max_model_len=%d), stopping loop",
                prompt_tokens, remaining, self.max_model_len,
            )
            return AgentOutput(actions=[])

        # FIX THIS
        max_tokens_this_turn = min(self.max_completion_tokens, remaining)
        if max_tokens_this_turn < self.max_completion_tokens:
            logger.warning(
                "capping max_tokens from %d to %d (prompt_len=%d, max_model_len=%d)",
                self.max_completion_tokens, max_tokens_this_turn, prompt_tokens, self.max_model_len,
            )
            self.llm.config.max_completion_tokens = max_tokens_this_turn

        llm_response = self.llm(prompt)
        llm_output = llm_response.message
        self.history.append(llm_output)
        self._actions_cnt += 1
        llm_call = LLMCall(
            llm_config=self.config.llm_config,
            prompt=prompt,
            prompt_tokens=llm_response.usage.prompt_tokens,
            output_tokens=llm_response.usage.completion_tokens,
            output=llm_output,
            usage=llm_response.usage,
            logprobs=llm_response.logprobs,
            completion_token_ids=llm_response.completion_token_ids,
            finish_reason=llm_response.finish_reason,
            metadata=llm_response.metadata,
        )
        actions = tir_parse_actions_tolerant(llm_output)
        for i in range(len(actions)):
            if actions[i].name not in self.tool_names:
                logger.warning(f"LLM called unknown tool '{actions[i].name}'")
                actions[i] = Action(id=actions[i].id, name="_unknown_tool", 
                                arguments={"name": actions[i].name, "arguments": actions[i].arguments})

        return AgentOutput(actions=actions, llm_calls=[llm_call])

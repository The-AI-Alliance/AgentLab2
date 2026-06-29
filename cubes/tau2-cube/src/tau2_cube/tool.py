"""Tool layer for tau2_cube.

Domain-agnostic adapter over a tau2 ``Environment``. Exposes:

- the domain's tools dynamically (``action_set`` built from ``env.get_tools()``,
  dispatched via ``env.get_response``), and
- a synthetic ``send_message_to_user`` action that drives tau2's ``UserSimulator``
  (the env-side LLM playing the customer) and returns its reply as the observation.

``self._env`` is the live tau2 Environment and the shared seam: ``CubeTask`` reads
state off it (e.g. ``self.tool._env.get_db_hash()``) to score. The user simulator
is wired in per-episode by ``CubeTask.reset()`` via ``start_user_session()``.
"""

from cube.container import Container
from cube.core import Action, ActionSchema, Content, Observation, StepError
from cube.tool import Tool, ToolConfig
from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.data_model.tasks import UserScenario
from tau2.registry import registry
from tau2.user.user_simulator import UserSimulator

SEND_MESSAGE_TO_USER = "send_message_to_user"


class CubeToolConfig(ToolConfig):
    """Selects the tau2 domain and the user-simulator LLM."""

    domain: str = "mock"
    user_llm: str = "hosted_vllm/qwen-32b"  # LiteLLM model id for the user simulator
    user_llm_args: dict = {}  # e.g. {"temperature": 0.0}

    def make(self, container: Container | None = None) -> "CubeTool":
        return CubeTool(self)


class CubeTool(Tool):
    """Agent-facing tool over a tau2 domain Environment (dynamic action surface)."""

    def __init__(self, config: CubeToolConfig) -> None:
        self._config = config
        self._user: UserSimulator | None = None
        self._user_state = None
        self.reset()

    def reset(self) -> None:
        """(Re)build the tau2 Environment for this domain. User session is wired
        separately by CubeTask.reset() once the task's scenario is known."""
        self._env = registry.get_env_constructor(self._config.domain)()
        self._user = None
        self._user_state = None
        self._messages: list = []  # tau2-format conversation log for evaluate_simulation

    def start_user_session(self, user_scenario: UserScenario) -> None:
        """Create the user simulator for this episode (LLM-free until first message)."""
        self._user = UserSimulator(
            llm=self._config.user_llm,
            instructions=user_scenario,  # type: ignore[arg-type]  # tau2 declares str but accepts UserScenario
            llm_args=dict(self._config.user_llm_args),
            tools=None,  # TODO: pass env.get_user_tools() for telecom dual-control
        )
        self._user_state = self._user.get_init_state()

    @property
    def action_set(self) -> list[ActionSchema]:
        """Domain tools (from tau2) + the synthetic send_message_to_user action."""
        schemas = []
        for tool in self._env.get_tools():
            fn = tool.openai_schema["function"]
            schemas.append(
                ActionSchema(
                    name=fn["name"], description=fn.get("description") or fn["name"], parameters=fn["parameters"]
                )
            )
        schemas.append(
            ActionSchema(
                name=SEND_MESSAGE_TO_USER,
                description="Send a natural-language message to the customer and receive their reply.",
                parameters={
                    "type": "object",
                    "properties": {"message": {"type": "string", "description": "The message to say to the customer."}},
                    "required": ["message"],
                },
            )
        )
        return schemas

    def execute_action(self, action: Action) -> Observation | StepError:
        """Route send_message_to_user to the user simulator; everything else to the env.
        Records the turn into self._messages (tau2 format) for evaluate_simulation.

        Tool errors come back as normal observations (tau2 semantics: the agent sees
        the error and can recover) — not StepError, which would end the episode.
        """
        if action.name == SEND_MESSAGE_TO_USER:
            return self._talk_to_user(action)
        # Domain tool: record AssistantMessage(tool_call) + the canonical ToolMessage from
        # env.get_response. get_response executes once and serializes via to_json_str — the
        # exact form set_state() reproduces when the evaluator replays the trajectory.
        tc = ToolCall(id=action.id or action.name, name=action.name, arguments=action.arguments, requestor="assistant")
        self._messages.append(AssistantMessage(role="assistant", content=None, tool_calls=[tc]))
        tool_msg = self._env.get_response(tc)
        self._messages.append(tool_msg)
        return Observation(contents=[Content.from_data(tool_msg.content or "", tool_call_id=action.id)])

    def _talk_to_user(self, action: Action) -> Observation | StepError:
        """Feed the agent's message to the user simulator, return the user's reply."""
        if self._user is None:
            return StepError.from_exception(
                RuntimeError("No user session — call start_user_session() in Task.reset().")
            )
        agent_msg = AssistantMessage(role="assistant", content=action.arguments.get("message", ""))
        self._messages.append(agent_msg)
        user_msg, self._user_state = self._user.generate_next_message(agent_msg, self._user_state)
        self._messages.append(user_msg)
        return Observation(contents=[Content.from_data(user_msg.content or "", tool_call_id=action.id)])


if __name__ == "__main__":
    for dom in ("mock", "airline", "retail", "telecom"):
        tool = CubeToolConfig(domain=dom).make()
        print(f"\n{dom}: {len(tool.action_set)} actions")
        for a in tool.action_set:
            print(f"  - {a.name}")

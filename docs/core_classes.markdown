# Core Classes Schema

![Core Classes Diagram](assets/images/classes_schema.png)

All objects are derived from the Pydantic `BaseModel` class, which provides data validation and serialization capabilities.

## Tools/Actions abstractions
```
class Tool(BaseModel):
    """
    Tool is a model that represents a tool specification with a type, name, description and arguments.
    Compatible with OAI, Anthropic and VLLM definitions.

    Attributes:
        type (Literal["function"]): The type of the tool, which is always "function".
        name (str): The name of the function.
        description (str): A brief description of the function.
        parameters (dict): A dictionary containing the parameters of the function.
    """

    type: Literal["function"] = "function"
    name: str
    description: str
    parameters: dict

    @classmethod
    """create tool object from python function"""
    def from_function(cls, func: Callable) -> "Tool":
        pass

    @property
    def schema(self) -> Dict[str, Any]:
    """produce dict that could be passed as tool schema into LLM api"""
        return self.json_dump()

class ToolCall(BaseModel):
    """
    A class representing a function call.

    Attributes:
        id (str): The identifier for the tool call.
        name (str): The name of the function being called.
        arguments (Any): The arguments to be passed to the function.
    """
    id: str
    name: str
    arguments: Dict[str, Any]
```

## LLM interaction abstractions, LiteLLM based:
[https://docs.litellm.ai/docs/completion/input](https://docs.litellm.ai/docs/completion/input)
[https://docs.litellm.ai/docs/completion/output](https://docs.litellm.ai/docs/completion/output)
```
class LLMMessage(BaseModel):
    """ LLMMessage represents a message for the LLM input"""
    role: Literal["system", "user", "assistant", "function", "tool"]
    content: str | list[Dict[str, Any]] | None #  may be null for assistant messages with function calls.
    name: Optional[str] = None  #  required if the role is "function". The name should match the name of the function represented in the content
    function_call: Optional[ToolCall] = None  #  may be present if the assistant is calling a function
    tool_call_id: Optional[str] = None  #  may be present if the message is from a tool result

class Prompt(BaseModel):
    """ Prompt represents the input prompt to chat completion api of LLM"""
    messages: List[LLMMessage]
    tools: list[dict]

    def to_list_of_dicts(self) -> List[Dict[str, Any]]:
        return [message.model_dump() for message in self.messages]

class LLMOutput(BaseModel):
    """ LLMOutput represents the output from LLM"""
    text: str
    tool_calls: List[ToolCall] | None = None
    thoughts: Optional[str] = None
    tokens: list[int] | None = None
    logprobs: list[float] | None = None

    metadata: dict = Field(default_factory=dict)

    @classmethod
    def from_litellm_response(cls, response: Response) -> "LLMOutput":
        pass
```

## Environment, Benchmark and Task abstractions
```
class Content(BaseModel):
    type: str  # e.g., "text/plain", "image/png"
    data: Any  # The actual content data

class Observation(BaseModel):
    tool_call_id: str | None = None # first observation may not be linked to any tool call
    contents: list[Content]
    metadata: dict = Field(default_factory=dict)
    reward_info: dict = Field(default_factory=dict)

    def to_messages(self) -> List[LLMMessage]:
    """Convert observation to a list of message suitable for sending to LLM""'
        pass

class Environment(BaseModel):
    def reset(self):
        pass

    @property
    def actions(self) -> List[Tool]:
    """ Returns list of actions supported by that environment"""
        pass

    def step(self, tool_call: ToolCall) -> Observation:
        pass

    def multi_step(self, tool_calls: List[ToolCall]) -> List[Observation]:
        """ Execute multiple function calls in parallel"""
        pass

    def close(self):
        pass

class Task(BaseModel):
    id: str
    evaluate_per_step: bool = False

    def setup(self, environment: Environment) -> list[Observation], dict:
    """ Returns goal obsevation, initial observation and a dict with additional task info"""
        pass

    def teardown(self, environment: Environment):
        pass

    def validate(self, environment: Environment, trace: Trace) -> dict:
    """validates the whole trace and state of the env in the end of the run"""
        pass

    def validate_step(environment: Environment, tool_calls: list[ToolCall],  observations: list[Observation]):
    """if evaluate_per_step=True this will be called to produce reward for each step"""
        pass

class Benchmark(BaseModel):
    name: str
    tasks: List[Task]
    environment: Environment

    def initialize(self):
    """loads data for tasks from the storage, prepares the environment"""
        pass
```

## Agent abstraction
```
class Trace(BaseModel):
    steps: List[Observation | LLMOutput]
    metadata: dict = Field(default_factory=dict)
    reward_info: dict = Field(default_factory=dict)

class Agent(BaseModel):
    """ ACP based definition, https://agentcommunicationprotocol.dev/core-concepts/agent-manifest"""
    name: str
    description: str | None = None
    input_content_types: List[str] # values ["image/png", "image/jpeg", "text/plain", "application/json"]
    output_content_types: List[str]
    actions: list[Tool]
    metadata: dict = Field(default_factory=dict)

    def reset(self):
        pass

    def step(self, trace: Trace) -> LLMOutput:
        """ Agent given not the last observation, but the full trace of all previous steps """

    def finished(self) -> bool:
        pass
```

## Single run abstraction
```
class AgentRun(BaseModel):
    agent: Agent
    task: Task
    environment: Environment

    @classmethod
    def from_config(cls, task: Task, config: Config) -> "AgentRun":
        environment = Environment(**config.environment)
        actions = task.filter_actions(environment.actions)
        agent = Agent(**config.agent, actions=actions)
        return AgentRun(agent, task, environment)

    def run(self) -> Dict[str, Any] -> Trace:
        self.agent.reset()
        self.environment.reset()
        observations, task_info = self.task.setup(self.environment)
        trace = Trace(
            steps=observations,
            metadata=dict(
                agent_info=self.agent.metadata,
                env_info=self.environment.metadata,
                task_info=task_info,
            )
        )
        while not self.task.finished() and not self.agent.finished():
            llm_output = self.agent.step(trace)
            trace.steps.append(llm_output)
            if llm_output.tool_calls:
                observations = self.environment.multi_step(llm_output.tool_calls)
                if self.task.validate_per_step:
                    # validator will update obs.reward_info in-place
                    self.task.validate_step(self.environment, llm_output.tool_calls, observations)
                trace.steps.extend(observations)

        self.task.teardown(self.environment)         
        trace.reward_info = self.task.validate(self.environment, trace)
        return trace
```

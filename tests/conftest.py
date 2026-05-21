"""Common fixtures for cube_harness tests."""

import tempfile
from pathlib import Path
from uuid import uuid4

import pytest
from cube.benchmark import Benchmark as CubeBenchmark
from cube.benchmark import (  # noqa: F401 — needed for Pydantic to resolve Task's TYPE_CHECKING import
    BenchmarkConfig as CubeBenchmarkConfig,
)
from cube.benchmark import (  # noqa: F401
    BenchmarkMetadata,
)
from cube.core import Action, ActionSchema, Content, EnvironmentOutput, Observation
from cube.task import Task as CubeTask
from cube.task import TaskConfig as CubeTaskConfig
from cube.task import TaskMetadata
from cube.tool import Tool, ToolConfig, tool_action
from PIL import Image

from cube_harness.agent import Agent, AgentConfig
from cube_harness.core import (
    AgentOutput,
    Trajectory,
    TrajectoryStep,
)
from cube_harness.episode import Episode
from cube_harness.llm import LLMConfig, Prompt

# --- Core fixtures ---


@pytest.fixture
def tmp_dir():
    """Temporary directory fixture for tests that need file I/O.

    Yields a path ending in _{uuid8} so Experiment's uniqueness validator
    recognises it as already unique and does not mutate the path.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / f"exp_{uuid4().hex[:8]}"
        path.mkdir()
        yield path


@pytest.fixture
def sample_action_schema() -> ActionSchema:
    """Sample action schema for testing."""
    return ActionSchema(
        name="click",
        description="Click on an element",
        parameters={
            "type": "object",
            "properties": {"element_id": {"type": "string", "description": "The element to click"}},
            "required": ["element_id"],
        },
    )


@pytest.fixture
def sample_action() -> Action:
    """Sample action for testing."""
    return Action(id="action_1", name="click", arguments={"element_id": "button_1"})


@pytest.fixture
def sample_content() -> Content:
    """Sample text content."""
    return Content.from_data("Hello, world!", name="greeting")


@pytest.fixture
def sample_image_content() -> Content:
    """Sample image content."""
    img = Image.new("RGB", (100, 100), color="red")
    return Content.from_data(img, name="screenshot")


@pytest.fixture
def sample_observation() -> Observation:
    """Sample observation with text content."""
    return Observation(contents=[Content.from_data("Task: Click the button")])


@pytest.fixture
def sample_env_output(sample_observation) -> EnvironmentOutput:
    """Sample environment output."""
    return EnvironmentOutput(obs=sample_observation, reward=0.5, done=False, info={"step": 1})


@pytest.fixture
def sample_agent_output(sample_action) -> AgentOutput:
    """Sample agent output."""
    return AgentOutput(actions=[sample_action])


@pytest.fixture
def empty_trajectory() -> Trajectory:
    """Empty trajectory for testing."""
    return Trajectory(id="test_traj")


@pytest.fixture
def sample_trajectory(sample_env_output, sample_agent_output) -> Trajectory:
    """Sample trajectory with steps."""
    traj = Trajectory(id="test_traj", metadata={"task_id": "test_task"})
    traj.steps.append(TrajectoryStep(output=sample_env_output, start_time=0.0, end_time=0.1))
    traj.steps.append(TrajectoryStep(output=sample_agent_output, start_time=0.1, end_time=0.2))
    return traj


# --- LLM fixtures ---


@pytest.fixture
def sample_llm_config() -> LLMConfig:
    """Sample LLM configuration."""
    return LLMConfig(model_name="gpt-5-nano", temperature=0.7, max_tokens=4096)


@pytest.fixture
def sample_prompt() -> Prompt:
    """Sample prompt for LLM."""
    return Prompt(
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}],
        tools=[],
    )


# --- Tool fixtures ---


class MockTool(Tool):
    """Mock tool implementation for testing."""

    def __init__(self):
        self.click_count = 0
        self.typed_texts = []

    @tool_action
    def click(self, element_id: str) -> str:
        """Click on an element.

        Args:
            element_id: The element to click.

        Returns:
            Click confirmation message.
        """
        self.click_count += 1
        return f"Clicked on {element_id}"

    @tool_action
    def type_text(self, element_id: str, text: str) -> str:
        """Type text into an element.

        Args:
            element_id: The element to type into.
            text: The text to type.

        Returns:
            Type confirmation message.
        """
        self.typed_texts.append((element_id, text))
        return f"Typed '{text}' into {element_id}"

    def reset(self):
        self.click_count = 0
        self.typed_texts = []


@pytest.fixture
def mock_tool() -> MockTool:
    """Mock tool for testing."""
    return MockTool()


class MockToolConfig(ToolConfig):
    """Mock tool configuration for testing."""

    def make(self, container=None) -> MockTool:
        _ = container
        return MockTool()


@pytest.fixture
def mock_tool_config() -> MockToolConfig:
    """Mock tool config for testing."""
    return MockToolConfig()


# --- Agent fixtures ---


class MockAgentConfig(AgentConfig):
    """Mock agent configuration."""

    name: str = "mock_agent"

    def make(self, action_set=None, **kwargs) -> "MockAgent":
        _ = action_set, kwargs
        return MockAgent(config=self)


class MockAgent(Agent):
    """Mock agent implementation for testing."""

    name = "MockAgent"
    description = "A mock agent for testing"
    input_content_types = ["text"]
    output_content_types = ["action"]

    def __init__(self, config: MockAgentConfig):
        super().__init__(config)
        self.step_count = 0
        self.actions_to_return: list[Action] = []

    def step(self, obs: Observation) -> AgentOutput:
        _ = obs
        self.step_count += 1
        if self.actions_to_return:
            actions = self.actions_to_return
        else:
            actions = [Action(name="final_step", arguments={})]
        return AgentOutput(actions=actions)


@pytest.fixture
def mock_agent_config() -> MockAgentConfig:
    """Mock agent config for testing."""
    return MockAgentConfig()


@pytest.fixture
def mock_agent(mock_agent_config) -> MockAgent:
    """Mock agent for testing."""
    return MockAgent(config=mock_agent_config)


# --- Cube mock classes ---


class MockCubeTask(CubeTask):
    """Minimal cube Task for testing — no external dependencies."""

    def reset(self) -> tuple[Observation, dict]:
        return Observation.from_text("Cube task goal"), {}

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict]:
        _ = obs
        return 1.0, {"success": True}


class MockCubeTaskConfig(CubeTaskConfig):
    """Cube TaskConfig that instantiates a MockCubeTask."""

    def make(self, runtime_context=None) -> MockCubeTask:
        _ = runtime_context
        return MockCubeTask(
            metadata=TaskMetadata(id=self.task_id),
            tool_config=self.tool_config or MockToolConfig(),
        )


class MockCubeBenchmark(CubeBenchmark):
    """Live runtime pair for ``MockCubeBenchmarkConfig`` — no shared infrastructure."""

    def _setup(self) -> None:
        pass

    def close(self) -> None:
        pass


class MockCubeBenchmarkConfig(CubeBenchmarkConfig):
    """Cube BenchmarkConfig with two inline tasks for testing."""

    benchmark_metadata = BenchmarkMetadata(
        name="mock-cube",
        version="0.1.0",
        description="Mock cube benchmark for testing",
    )
    task_metadata = {
        "mock_cube_task_1": TaskMetadata(id="mock_cube_task_1"),
        "mock_cube_task_2": TaskMetadata(id="mock_cube_task_2"),
    }
    task_config_class = MockCubeTaskConfig
    benchmark_class = MockCubeBenchmark


@pytest.fixture
def mock_cube_task_config() -> MockCubeTaskConfig:
    """Cube task config for mock_cube_task_1."""
    return MockCubeTaskConfig(metadata=TaskMetadata(id="mock_cube_task_1"))


@pytest.fixture
def mock_cube_benchmark_config() -> MockCubeBenchmarkConfig:
    """Cube benchmark config with two mock tasks."""
    return MockCubeBenchmarkConfig()


@pytest.fixture
def mock_episode(tmp_dir, mock_agent_config, mock_cube_task_config) -> Episode:
    """Sample episode for testing."""
    return Episode(
        id=0,
        output_dir=tmp_dir,
        agent_config=mock_agent_config,
        task_config=mock_cube_task_config,
        exp_name="mock-episode",
        max_steps=5,
        runtime_context=None,
        storage=None,
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--xray-screenshots",
        action="store_true",
        default=False,
        help="Save xray e2e screenshots to /tmp/xray_screenshots/ for visual inspection.",
    )

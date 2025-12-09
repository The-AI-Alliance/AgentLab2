from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from agentlab2.environment import EnvironmentConfig, Task


class Benchmark(BaseModel, ABC):
    """Represents a benchmark consisting of multiple tasks and an environment."""

    metadata: dict = Field(default_factory=dict)
    env_config: EnvironmentConfig

    @abstractmethod
    def setup(self):
        """
        Perform common steps necessary to prepare the environment for all tasks,
        like running web server, launching containers, etc.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up resources after all tasks are done.
        Called automatically by Experiment
        """
        pass

    @abstractmethod
    def tasks(self) -> list[Task]:
        """Return the list of tasks in this benchmark."""
        pass

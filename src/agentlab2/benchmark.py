from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from agentlab2.environment import Task


class Benchmark(BaseModel, ABC):
    """Represents a benchmark consisting of multiple tasks and an environment."""

    metadata: dict = Field(default_factory=dict)
    tasks: list[Task] = Field(default_factory=list)

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

from agentlab2.core import Action, ActionSchema
from agentlab2.environment import Environment, EnvironmentConfig
from agentlab2.tools.playwright import SyncPlaywrightTool


class BrowserEnvConfig(EnvironmentConfig):
    """Configuration for BrowserEnv."""

    headless: bool = True
    timeout: int = 30000  # in milliseconds
    pw_kwargs: dict = {}

    def make(self) -> "BrowserEnv":
        return BrowserEnv(self)


def final_step():
    """Stop the task execution."""
    pass


class BrowserEnv(Environment):
    """Environment that uses just a single tool, playwright browser, to interact with web pages."""

    metadata: dict = {"tools": ["SyncPlaywrightTool"]}

    def __init__(self, config: BrowserEnvConfig):
        super().__init__()
        self.config = config
        self.browser_tool = SyncPlaywrightTool(
            headless=self.config.headless,
            timeout=self.config.timeout,
            **self.config.pw_kwargs,
        )
        self._finished: bool = False

    @property
    def actions(self):
        return self.browser_tool.actions + [ActionSchema.from_function(final_step)]

    def step(self, action: Action):
        if action.name == "final_step":
            self._finished = True
            action_result = None
        else:
            action_result = self.browser_tool.execute_action(action)
        return self.browser_tool.page_obs(action.id, action_result)

    def goto(self, url: str):
        self.browser_tool.goto(url)

    def finished(self) -> bool:
        return self._finished

    def evaluate_js(self, script: str):
        return self.browser_tool.evaluate_js(script)

    def reset(self):
        self.browser_tool.reset()
        self._finished = False

    def close(self):
        self.browser_tool.close()

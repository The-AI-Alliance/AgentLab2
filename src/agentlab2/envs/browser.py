from agentlab2.environment import Environment, EnvironmentConfig
from agentlab2.tools.playwright import SyncPlaywrightTool


class BrowserEnvConfig(EnvironmentConfig):
    """Configuration for BrowserEnv."""

    headless: bool = True
    timeout: int = 30000  # in milliseconds
    pw_kwargs: dict = {}

    def make(self) -> "BrowserEnv":
        return BrowserEnv(self)


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

    @property
    def actions(self):
        return self.browser_tool.actions

    def step(self, actions):
        if not actions:
            raise ValueError("No actions provided for BrowserEnv step.")
        action = actions[0]
        observation = self.browser_tool.step(action)
        return observation

    def goto(self, url: str):
        self.browser_tool.goto(url)

    def evaluate_js(self, script: str):
        return self.browser_tool.evaluate_js(script)

    def reset(self):
        self.browser_tool.reset()

    def close(self):
        self.browser_tool.close()

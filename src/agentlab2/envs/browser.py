from agentlab2.core import Action, Content, EnvironmentOutput
from agentlab2.environment import Environment, EnvironmentConfig, Task
from agentlab2.tools.playwright import SyncPlaywrightTool


class BrowserEnvConfig(EnvironmentConfig):
    """Configuration for BrowserEnv."""

    headless: bool = True
    timeout: int = 30000  # in milliseconds
    use_html: bool = True
    use_axtree: bool = False
    use_screenshot: bool = True
    prune_html: bool = True
    pw_kwargs: dict = {}

    def make(self, task: Task) -> "BrowserEnv":
        """Create a BrowserEnv instance from the configuration for specified task."""
        return BrowserEnv(self, task)


class BrowserEnv(Environment):
    """Environment that uses just a single tool, playwright browser, to interact with web pages."""

    metadata: dict = {"tools": ["SyncPlaywrightTool"]}

    def __init__(self, config: BrowserEnvConfig, task: Task):
        self.task = task
        self.config = config
        self.browser_tool = SyncPlaywrightTool(
            headless=self.config.headless,
            timeout=self.config.timeout,
            use_html=self.config.use_html,
            use_axtree=self.config.use_axtree,
            use_screenshot=self.config.use_screenshot,
            prune_html=self.config.prune_html,
            **self.config.pw_kwargs,
        )

    def actions(self):
        return self.task.filter_actions(self.browser_tool.actions)

    def setup(self) -> EnvironmentOutput:
        self.browser_tool.reset()
        goal, info = self.task.setup(self)
        obs = self.browser_tool.page_obs()
        obs.contents["goal"] = Content(data=goal)
        obs = self.task.obs_postprocess(obs)
        return EnvironmentOutput(observation=obs, info=info)

    def step(self, action: Action) -> EnvironmentOutput:
        action_result = self.browser_tool.execute_action(action)
        obs = self.browser_tool.page_obs(action.id, action_result)
        done = self.task.finished()
        if self.task.validate_per_step:
            reward, info = self.task.validate(obs, action)
        elif done:
            reward, info = self.task.validate(obs, action)
        else:
            reward, info = 0.0, {}
        return EnvironmentOutput(observation=obs, reward=reward, info=info, done=done)

    def goto(self, url: str):
        self.browser_tool.goto(url)

    def evaluate_js(self, script: str):
        return self.browser_tool.evaluate_js(script)

    def close(self):
        self.task.teardown()
        self.browser_tool.close()

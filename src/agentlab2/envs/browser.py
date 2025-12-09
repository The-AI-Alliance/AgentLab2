from agentlab2.core import Action, Content, EnvironmentOutput, ToolSchema
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

    def make(self) -> "BrowserEnv":
        """Create a BrowserEnv instance from the configuration for specified task."""
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
            use_html=self.config.use_html,
            use_axtree=self.config.use_axtree,
            use_screenshot=self.config.use_screenshot,
            prune_html=self.config.prune_html,
            **self.config.pw_kwargs,
        )

    @property
    def actions(self):
        final_step_action = ToolSchema(name="final_step", description="Stop the task execution.")
        return self.browser_tool.actions + [final_step_action]

    def setup(self, task: Task) -> EnvironmentOutput:
        self.task = task
        self.browser_tool.reset()
        goal, info = self.task.setup(self)
        obs = self.browser_tool.page_obs()
        obs.contents["goal"] = Content(data=goal)
        obs = self.task.obs_postprocess(obs)
        return EnvironmentOutput(observation=obs, info=info)

    def step(self, action: Action) -> EnvironmentOutput:
        assert self.task is not None, "Environment step called before setup."
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
        if self.task is not None:
            self.task.teardown()
        self.browser_tool.close()

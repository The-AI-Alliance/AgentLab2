from agentlab2.core import Action, EnvironmentOutput
from agentlab2.environment import EnvironmentConfig, Task, ToolboxEnv
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
        tool_config = self.model_dump()
        tool_config.pop("pw_kwargs", None)
        browser_tool = SyncPlaywrightTool(**tool_config, **self.pw_kwargs)
        return BrowserEnv(task, browser_tool)


class BrowserEnv(ToolboxEnv):
    """Environment that uses only one browser tool for interaction."""

    def __init__(self, task: Task, browser_tool: SyncPlaywrightTool):
        super().__init__(task, [browser_tool])
        self.browser_tool = browser_tool

    def setup(self) -> EnvironmentOutput:
        output = super().setup()
        page_obs = self.browser_tool.page_obs()
        output.obs.contents += self.task.obs_postprocess(page_obs).contents
        return output

    def step(self, action: Action) -> EnvironmentOutput:
        output = super().step(action)
        page_obs = self.browser_tool.page_obs()
        output.obs.contents += self.task.obs_postprocess(page_obs).contents
        return output

    def goto(self, url: str):
        self.browser_tool.goto(url)

    def evaluate_js(self, script: str):
        return self.browser_tool.evaluate_js(script)

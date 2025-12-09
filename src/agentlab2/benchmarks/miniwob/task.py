import logging
from typing import Any, ClassVar

from PIL import Image

from agentlab2.core import Content, Observation, ToolSchema
from agentlab2.environment import Task
from agentlab2.envs.browser import BrowserEnv

logger = logging.getLogger(__name__)


class MiniWobTask(Task[BrowserEnv]):
    desc: str
    subdomain: str
    base_url: str
    remove_human_display: bool = True
    episode_max_time: int = 1000000
    max_turns: int = 10
    validate_per_step: bool = True
    actions_whitelist: ClassVar[list[str]] = [
        "browser_press_key",
        "browser_type",
        "browser_click",
        "browser_drag",
        "browser_hover",
        "browser_select_option",
        "browser_mouse_click_xy",
    ]
    _env: BrowserEnv = None  # type: ignore

    def model_post_init(self, __context: Any):
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]

    @property
    def url(self) -> str:
        return f"{self.base_url}/{self.subdomain}.html"

    def setup(self, env: BrowserEnv) -> tuple[str, dict]:
        """
        Set up everything needed to execute the task.

        Args:
            page: the active playwright page.

        Returns:
            goal: str, goal of the task.
            info: dict, custom information from the task.
        """
        logger.info(f"Setting up MiniWob task {self.id} at {self.url}")
        env.goto(self.url)
        setup_js = self._get_setup_js()
        setup_result = env.evaluate_js(setup_js)
        goal, info = self._parse_setup_result(setup_result)
        self._env = env
        return goal, info

    def teardown(self) -> None:
        """
        Tear down the task, clean up resources if needed.

        Args:
            page: the active playwright page.
        """
        teardown_js = self._get_teardown_js()
        if teardown_js:
            self._env.evaluate_js(teardown_js)

    def validate(self, *args, **kwargs) -> tuple[float, dict]:
        """
        Validate the task, either per step or at the end.

        Returns:
            reward: float, the reward obtained.
            info: dict, custom information from the validation.
        """
        validate_js = self._get_step_validate_js()
        validate_result = self._env.evaluate_js(validate_js)
        reward, info = self._parse_validation_result(validate_result)
        return reward, info

    def _get_setup_js(self) -> str:
        if self.remove_human_display:
            js = r"""
let __display_ids = ['reward-display', 'click-canvas', 'sync-task-cover'];
let __display_divs = {};
let __query_div_hidden_copy = null;

removeDisplay = function() {
  core.clearTimer();
  document.body.removeEventListener('click', core.canvasDrawClick);

  __query_div_hidden_copy = document.getElementById('query').cloneNode(true);
  document.getElementById('query').innerHTML = '';

  for (i in __display_ids) {
    elem_id = __display_ids[i];
    elem = document.getElementById(elem_id);
    // remove elem from the document
    elem.remove();
    // but keep it stored somewhere to bring back later
    __display_divs[elem_id] = elem;
  }
};

bringBackDisplay = function() {
  document.getElementById('query').innerHTML = __query_div_hidden_copy.innerHTML;
  for (var elem_id in __display_divs){
    document.body.appendChild(__display_divs[elem_id]);
  }
  core.createDisplay();
};

core.endEpisode_legacy = core.endEpisode;
core.startEpisodeReal_legacy = core.startEpisodeReal;
core.getUtterance_legacy = core.getUtterance;

core.getUtterance = function () {
  bringBackDisplay();
  utterance = core.getUtterance_legacy();
  removeDisplay();
  return utterance;
};

core.endEpisode = function(reward, time_proportional, reason){
  bringBackDisplay();
  core.endEpisode_legacy(reward, time_proportional, reason);
  removeDisplay();
};

core.startEpisodeReal = function() {
  bringBackDisplay();
  core.startEpisodeReal_legacy();
  removeDisplay();
};

removeDisplay();
"""
        else:
            js = ""
        js += f"""
Math.seedrandom(42);
core.EPISODE_MAX_TIME = {self.episode_max_time};
core.startEpisodeReal();
while (!WOB_TASK_READY) {{
  await new Promise(resolve => setTimeout(resolve, 100));
}}
return core.getUtterance();
    """
        return f"async () => {{{js}}}"

    def _parse_setup_result(self, setup_result: str | dict | list) -> tuple[str, dict]:
        if isinstance(setup_result, dict):
            return setup_result["utterance"], {}
        elif isinstance(setup_result, str):
            return setup_result, {}
        else:
            raise ValueError(f"Unexpected setup_result type: {type(setup_result)}")

    def _get_teardown_js(self) -> str:
        return ""

    def _get_step_validate_js(self) -> str:
        return """() => {
return [WOB_REWARD_GLOBAL, WOB_RAW_REWARD_GLOBAL, WOB_REWARD_REASON, WOB_DONE_GLOBAL, WOB_EPISODE_ID, WOB_TASK_READY];
}"""

    def _parse_validation_result(self, validation_result: str | dict | list) -> tuple[float, dict]:
        if isinstance(validation_result, list):
            chunks = validation_result
            done = chunks[3]
        elif isinstance(validation_result, dict):
            raise ValueError("Validation result as dict is not supported")
        else:
            chunks = [c.strip() for c in validation_result.split(",")]
            done = chunks[3].strip().lower() == "true"
        raw_reward = float(chunks[1])
        reward = float(raw_reward > 0)
        return reward, {
            "raw_reward": raw_reward,
            "reward_reason": chunks[2],
            "done": done,
        }

    def obs_postprocess(self, obs: Observation) -> Observation:
        if screenshot := obs.contents.get("screenshot", None):
            # crop to 332x214 because this is the viewport size for MiniWob
            if isinstance(screenshot.data, Image.Image):
                obs.contents["screenshot"] = Content(data=screenshot.data.crop((0, 0, 332, 214)))
        return obs

    def filter_actions(self, actions: list[ToolSchema]) -> list[ToolSchema]:
        filtered = [a for a in actions if a.name in self.actions_whitelist]
        logger.info(f"Chosen {len(filtered)} out of {len(actions)} actions for MiniWob task.")
        return filtered

    def finished(self) -> bool:
        _, info = self.validate()
        return info.get("done", False)

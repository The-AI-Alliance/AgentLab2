from io import TextIOWrapper
import logging
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from cube.benchmark import RuntimeContext
from cube.core import Content, Observation
from cube.task import Task, TaskConfig, TaskMetadata  # noqa: F401 — TaskMetadata kept for typing
from cube.tools.browser import BrowserTool
from PIL import Image
from pydantic import PrivateAttr


class MiniWobTaskMetadata(TaskMetadata):
    """TaskMetadata subclass for MiniWob++ tasks.
    Adds cube-specific public fields that are safe to ship in task_metadata.json.
    """

    nondeterministic: bool = False


logger = logging.getLogger(__name__)


class MiniWobTask(Task):
    validate_per_step: bool = True
    base_url: str = "http://localhost:8000/miniwob"
    remove_human_display: bool = True
    episode_max_time: int = 1000000

    _server_process: subprocess.Popen | None = PrivateAttr(default=None)
    _stdout_file: TextIOWrapper | None = PrivateAttr(default=None)
    _stderr_file: TextIOWrapper | None = PrivateAttr(default=None)

    @property
    def tool(self) -> BrowserTool:  # type: ignore[override]
        return self._tool  # type: ignore[return-value]

    @property
    def url(self) -> str:
        return f"{self.base_url}/{self.metadata.id}.html"

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        self.tool.reset()
        self.tool.goto(self.url)
        setup_result = self.tool.evaluate_js(_build_setup_js(self.remove_human_display, self.episode_max_time))
        goal, info = _parse_setup_result(setup_result)
        obs = Observation.from_text(goal) + self.obs_postprocess(self.tool.page_obs())
        return obs, {**info, "task_id": self.id, "task_url": self.url, "goal": goal}

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict[str, Any]]:
        result = self.tool.evaluate_js("""() => {
return [WOB_REWARD_GLOBAL, WOB_RAW_REWARD_GLOBAL, WOB_REWARD_REASON, WOB_DONE_GLOBAL, WOB_EPISODE_ID, WOB_TASK_READY];}""")
        return _parse_validation_result(result)

    def finished(self, obs: Observation | None = None) -> bool:
        return self.tool.evaluate_js("() => {return WOB_DONE_GLOBAL;}")

    def close(self) -> None:
        try:
            super().close()
        finally:
            _stop_miniwob_server(
                process=self._server_process,
                stdout_file=self._stdout_file,
                stderr_file=self._stderr_file,
            )
            self._server_process = None
            self._stdout_file = None
            self._stderr_file = None

    def obs_postprocess(self, obs: Observation) -> Observation:
        contents = []
        for content in obs.contents:
            if content.name == "screenshot" and isinstance(content.data, Image.Image):
                # crop to 332x214 because this is the viewport size for MiniWob
                contents.append(Content.from_data(content.data.crop((0, 0, 332, 214)), name=content.name))
            else:
                contents.append(content)
        obs.contents = contents
        return obs


class MiniWobTaskConfig(TaskConfig[MiniWobTaskMetadata]):
    html_path: str
    port: int | None = None
    remove_human_display: bool = True
    episode_max_time: int = 1000000
    server_start_timeout: float = 10.0
    server_start_poll_interval: float = 0.1

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
    ) -> MiniWobTask:
        _ = runtime_context
        assert self.tool_config is not None, "tool_config must be set"
        server = _start_miniwob_server(
            html_path=Path(self.html_path),
            port=self.port,
            task_id=self.metadata.id,
            startup_timeout=self.server_start_timeout,
            startup_poll_interval=self.server_start_poll_interval,
        )
        try:
            task = MiniWobTask(
                metadata=self.metadata,
                tool_config=self.tool_config,
                base_url=server.base_url,
                remove_human_display=self.remove_human_display,
                episode_max_time=self.episode_max_time,
            )
        except Exception:
            _stop_miniwob_server(
                process=server.process,
                stdout_file=server.stdout_file,
                stderr_file=server.stderr_file,
            )
            raise
        task._server_process = server.process
        task._stdout_file = server.stdout_file
        task._stderr_file = server.stderr_file
        return task


class _MiniWobServer:
    def __init__(
        self,
        *,
        port: int,
        base_url: str,
        process: subprocess.Popen,
        stdout_file: TextIOWrapper,
        stderr_file: TextIOWrapper,
    ) -> None:
        self.port = port
        self.base_url = base_url
        self.process = process
        self.stdout_file = stdout_file
        self.stderr_file = stderr_file


def _allocate_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_miniwob_server(
    *,
    html_path: Path,
    port: int | None,
    task_id: str,
    startup_timeout: float,
    startup_poll_interval: float,
) -> _MiniWobServer:
    selected_port = port or _allocate_free_port()
    base_url = f"http://localhost:{selected_port}/miniwob"
    tmp_dir = Path(tempfile.gettempdir())
    safe_task_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in task_id)
    stdout_file = open(tmp_dir / f"miniwob_server_{safe_task_id}_{selected_port}_stdout.log", "w")
    stderr_file = open(tmp_dir / f"miniwob_server_{safe_task_id}_{selected_port}_stderr.log", "w")
    logger.info("Starting MiniWob server at port %s serving from %s...", selected_port, html_path)
    process = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(selected_port), "--bind", "127.0.0.1"],
        cwd=html_path,
        stdout=stdout_file,
        stderr=stderr_file,
    )

    server = _MiniWobServer(
        port=selected_port,
        base_url=base_url,
        process=process,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
    )
    startup_deadline = time.monotonic() + startup_timeout
    last_response_error: Exception | None = None

    while time.monotonic() < startup_deadline:
        if process.poll() is not None:
            stderr_file.flush()
            stderr_path = Path(stderr_file.name)
            stderr_content = stderr_path.read_text() if stderr_path.exists() else "No stderr available"
            returncode = process.returncode
            _stop_miniwob_server(process=process, stdout_file=stdout_file, stderr_file=stderr_file)
            raise RuntimeError(f"MiniWob server failed to start (exit code {returncode}): {stderr_content}")

        try:
            urllib.request.urlopen(base_url, timeout=1).close()
            logger.info("MiniWob server responding at %s", base_url)
            return server
        except Exception as exc:
            last_response_error = exc
            time.sleep(startup_poll_interval)

    _stop_miniwob_server(process=process, stdout_file=stdout_file, stderr_file=stderr_file)
    raise RuntimeError(
        f"MiniWob server failed to respond at {base_url} within {startup_timeout:.1f}s"
    ) from last_response_error


def _stop_miniwob_server(
    *,
    process: subprocess.Popen | None,
    stdout_file: TextIOWrapper | None,
    stderr_file: TextIOWrapper | None,
) -> None:
    if process is not None and process.poll() is None:
        logger.info("Shutting down MiniWob server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Server did not terminate gracefully, killing...")
            process.kill()
            process.wait(timeout=5)

    if stdout_file is not None and not stdout_file.closed:
        stdout_file.close()
    if stderr_file is not None and not stderr_file.closed:
        stderr_file.close()


def _build_setup_js(remove_human_display: bool, episode_max_time: int) -> str:
    if remove_human_display:
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
core.EPISODE_MAX_TIME = {episode_max_time};
core.startEpisodeReal();
while (!WOB_TASK_READY) {{
  await new Promise(resolve => setTimeout(resolve, 100));
}}
return core.getUtterance();
    """
    return f"async () => {{{js}}}"


def _parse_setup_result(setup_result: str | dict) -> tuple[str, dict]:
    if isinstance(setup_result, dict):
        return setup_result["utterance"], {}
    elif isinstance(setup_result, str):
        return setup_result, {}
    else:
        raise ValueError(f"Unexpected setup_result type: {type(setup_result)}")


def _parse_validation_result(validation_result: str | dict | list) -> tuple[float, dict]:
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

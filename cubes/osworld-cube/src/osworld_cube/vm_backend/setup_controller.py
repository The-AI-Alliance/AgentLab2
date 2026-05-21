"""Task setup controller: ported from desktop_env.controllers.setup.SetupController.

Uses GuestAgent for all VM communication instead of direct vm_ip/server_port params.
Proxy support (proxy pool, tinyproxy, Google Drive OAuth) is preserved but inactive
unless explicitly enabled.
"""

import json
import logging
import os
import platform
import shutil
import sqlite3
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Union

import requests
from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright
from requests_toolbelt.multipart.encoder import MultipartEncoder

from cube_computer_tool.guest_agent import GuestAgent

logger = logging.getLogger(__name__)

MAX_RETRIES = 20
CHROMIUM_REMOTE_DEBUGGING_PORT = 9222


class SetupController:
    """Orchestrates task-setup steps in the OSWorld VM.

    Ported from desktop_env.controllers.setup.SetupController.
    The key difference is that the constructor takes a :class:`GuestAgent`
    instead of ``vm_ip`` + ``server_port``, and the proxy pool initialization
    is removed from the module level.

    Parameters
    ----------
    guest : GuestAgent
        HTTP client connected to the running VM.
    chromium_port : int
        Host port forwarded to the VM's Chromium DevTools port.
    vlc_port : int
        Host port forwarded to the VM's VLC HTTP port.
    cache_dir : str
        Directory for caching downloaded setup files.
    client_password : str
        Password for ``sudo`` operations inside the VM.
    screen_width : int
        VM screen width in pixels.
    screen_height : int
        VM screen height in pixels.
    """

    def __init__(
        self,
        guest: GuestAgent,
        chromium_port: int = 9222,
        vlc_port: int = 8080,
        cache_dir: str = "cache",
        client_password: str = "password",
        screen_width: int = 1920,
        screen_height: int = 1080,
        guest_chromium_port: int = CHROMIUM_REMOTE_DEBUGGING_PORT,
    ) -> None:
        self.guest = guest
        self.chromium_port = chromium_port
        self.guest_chromium_port = guest_chromium_port
        self.vlc_port = vlc_port
        self.http_server = guest._base_url
        self.http_server_setup_root = guest._base_url + "/setup"
        self.cache_dir = cache_dir
        self.client_password = client_password
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.use_proxy = False

    def reset_cache_dir(self, cache_dir: str) -> None:
        """Update the per-task cache directory."""
        self.cache_dir = cache_dir

    def setup(self, config: list[dict[str, Any]], use_proxy: bool = False) -> bool:
        """Run all setup steps from the task config.

        Parameters
        ----------
        config : list[dict]
            List of setup step dicts with ``type`` and ``parameters`` keys.
        use_proxy : bool
            Whether this task requires a proxy.

        Returns
        -------
        bool
            True if all steps completed, False if the VM was unreachable.
        """
        self.use_proxy = use_proxy

        # Wait for VM connectivity
        for retry in range(MAX_RETRIES):
            try:
                requests.get(self.http_server + "/terminal", timeout=5)
                break
            except Exception:
                time.sleep(5)
                logger.info("Waiting for VM connectivity: %d/%d", retry + 1, MAX_RETRIES)
        else:
            logger.error("VM unreachable after %d retries", MAX_RETRIES)
            return False

        for i, cfg in enumerate(config):
            config_type: str = cfg["type"]
            parameters: dict[str, Any] = cfg["parameters"]
            setup_fn_name = "_{:}_setup".format(config_type)

            if not hasattr(self, setup_fn_name):
                raise AttributeError(f"SetupController has no method {setup_fn_name!r}")

            try:
                logger.info("Setup step %d/%d: %s", i + 1, len(config), setup_fn_name)
                getattr(self, setup_fn_name)(**parameters)
                if setup_fn_name == "_launch_setup":
                    time.sleep(2)
                logger.info("Setup completed: %s", setup_fn_name)
            except Exception as exc:
                logger.error("Setup failed at step %d/%d: %s — %s", i + 1, len(config), setup_fn_name, exc)
                logger.error(traceback.format_exc())
                raise RuntimeError(f"Setup step {i + 1} failed: {setup_fn_name} — {exc}") from exc

        return True

    # ------------------------------------------------------------------
    # Setup step handlers (alphabetical)
    # ------------------------------------------------------------------

    def _connect_to_chrome(self, playwright: Any, remote_debugging_url: str, attempts: int = 30) -> Any:
        last_exc: Exception | None = None
        for attempt in range(attempts):
            if attempt > 0:
                time.sleep(5)

            try:
                return playwright.chromium.connect_over_cdp(remote_debugging_url)
            except Exception as exc:
                last_exc = exc
                logger.warning("Chrome CDP attempt %d/%d failed: %s", attempt + 1, attempts, exc)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Could not connect to Chrome CDP at {remote_debugging_url}")

    def _activate_window_setup(self, window_name: str, strict: bool = False, by_class: bool = False) -> None:
        payload = json.dumps({"window_name": window_name, "strict": strict, "by_class": by_class})
        try:
            resp = requests.post(
                self.http_server_setup_root + "/activate_window",
                headers={"Content-Type": "application/json"},
                data=payload,
            )
            if resp.status_code == 200:
                logger.info("Window activated: %s", window_name)
            else:
                logger.error("Failed to activate window %s: %s", window_name, resp.text)
        except requests.RequestException as exc:
            logger.error("activate_window error: %s", exc)

    def _change_wallpaper_setup(self, path: str) -> None:
        if not path:
            raise ValueError(f"Invalid wallpaper path: {path!r}")
        payload = json.dumps({"path": path})
        try:
            resp = requests.post(
                self.http_server_setup_root + "/change_wallpaper",
                headers={"Content-Type": "application/json"},
                data=payload,
            )
            if resp.status_code != 200:
                logger.error("Failed to change wallpaper: %s", resp.text)
        except requests.RequestException as exc:
            logger.error("change_wallpaper error: %s", exc)

    def _chrome_close_tabs_setup(self, urls_to_close: list[str]) -> None:
        from osworld_cube.vm_backend.metrics.utils import compare_urls

        time.sleep(5)
        remote_debugging_url = f"http://{self.guest.host}:{self.chromium_port}"

        with sync_playwright() as p:
            browser = self._connect_to_chrome(p, remote_debugging_url)
            context = browser.contexts[0]
            for i, url in enumerate(urls_to_close):
                for page in context.pages:
                    if compare_urls(page.url, url):
                        page.close()
                        logger.info("Closed tab %d: %s", i + 1, url)
                        break

    def _chrome_open_tabs_setup(self, urls_to_open: list[str]) -> None:
        remote_debugging_url = f"http://{self.guest.host}:{self.chromium_port}"
        logger.info("Connecting to Chrome @ %s", remote_debugging_url)

        with sync_playwright() as p:
            browser = self._connect_to_chrome(p, remote_debugging_url)
            for i, url in enumerate(urls_to_open):
                context = browser.contexts[0]
                page = context.new_page()
                try:
                    page.goto(url, timeout=60000)
                except Exception:
                    logger.warning("Opening %s exceeded time limit", url)
                logger.info("Opened tab %d: %s", i + 1, url)

                if i == 0:
                    context.pages[0].close()

    def _close_window_setup(self, window_name: str, strict: bool = False, by_class: bool = False) -> None:
        payload = json.dumps({"window_name": window_name, "strict": strict, "by_class": by_class})
        try:
            resp = requests.post(
                self.http_server_setup_root + "/close_window",
                headers={"Content-Type": "application/json"},
                data=payload,
            )
            if resp.status_code != 200:
                logger.error("Failed to close window %s: %s", window_name, resp.text)
        except requests.RequestException as exc:
            logger.error("close_window error: %s", exc)

    def _command_setup(self, command: list[str], **kwargs: Any) -> None:
        self._execute_setup(command, **kwargs)

    def _download_setup(self, files: list[dict[str, str]]) -> None:
        for f in files:
            url: str = f["url"]
            path: str = f["path"]
            if not url or not path:
                raise ValueError(f"Invalid download url={url!r} or path={path!r}")

            cache_path = os.path.join(
                self.cache_dir,
                "{:}_{:}".format(uuid.uuid5(uuid.NAMESPACE_URL, url), os.path.basename(path)),
            )

            if not os.path.exists(cache_path):
                logger.info("Downloading %s → %s", url, cache_path)
                for attempt in range(3):
                    try:
                        resp = requests.get(url, stream=True, timeout=300)
                        resp.raise_for_status()
                        with open(cache_path, "wb") as fp:
                            for chunk in resp.iter_content(chunk_size=8192):
                                if chunk:
                                    fp.write(chunk)
                        break
                    except requests.RequestException as exc:
                        logger.error("Download attempt %d failed: %s", attempt + 1, exc)
                        if os.path.exists(cache_path):
                            os.remove(cache_path)
                else:
                    raise requests.RequestException(f"Failed to download {url} after 3 attempts")

            form = MultipartEncoder({"file_path": path, "file_data": (os.path.basename(path), open(cache_path, "rb"))})
            resp = requests.post(
                self.http_server_setup_root + "/upload",
                headers={"Content-Type": form.content_type},
                data=form,
                timeout=600,
            )
            if resp.status_code != 200:
                raise requests.RequestException(f"Upload failed ({resp.status_code}): {resp.text}")
            logger.info("Uploaded %s to VM at %s", os.path.basename(path), path)

    def _execute_setup(
        self,
        command: Union[str, list[str]],
        stdout: str = "",
        stderr: str = "",
        shell: bool = False,
        until: dict[str, Any] | None = None,
    ) -> None:
        if not command:
            raise ValueError("Empty execute command")

        until = until or {}
        command = self._replace_screen_env(command)
        payload = json.dumps({"command": command, "shell": shell})
        headers = {"Content-Type": "application/json"}

        terminates = False
        nb_failings = 0
        while not terminates:
            try:
                resp = requests.post(self.http_server_setup_root + "/execute", headers=headers, data=payload)
                if resp.status_code == 200:
                    results = resp.json()
                    if stdout:
                        Path(self.cache_dir, stdout).write_text(results.get("output", ""))
                    if stderr:
                        Path(self.cache_dir, stderr).write_text(results.get("error", ""))
                else:
                    results = None
                    nb_failings += 1
            except requests.RequestException as exc:
                logger.error("execute error: %s", exc)
                results = None
                nb_failings += 1

            if not until:
                terminates = True
            elif results is not None:
                terminates = (
                    ("returncode" in until and results.get("returncode") == until["returncode"])
                    or ("stdout" in until and until["stdout"] in results.get("output", ""))
                    or ("stderr" in until and until["stderr"] in results.get("error", ""))
                )
            terminates = terminates or nb_failings >= 5
            if not terminates:
                time.sleep(0.3)

    def _execute_with_verification_setup(
        self,
        command: list[str],
        verification: dict[str, Any] | None = None,
        max_wait_time: int = 10,
        check_interval: float = 1.0,
        shell: bool = False,
    ) -> dict[str, Any]:
        if not command:
            raise ValueError("Empty command")
        payload = json.dumps(
            {
                "command": command,
                "shell": shell,
                "verification": verification or {},
                "max_wait_time": max_wait_time,
                "check_interval": check_interval,
            }
        )
        try:
            resp = requests.post(
                self.http_server_setup_root + "/execute_with_verification",
                headers={"Content-Type": "application/json"},
                data=payload,
                timeout=max_wait_time + 10,
            )
            if resp.status_code == 200:
                return resp.json()
            raise RuntimeError(f"Verification failed: {resp.text}")
        except requests.RequestException as exc:
            raise RuntimeError(f"Request failed: {exc}") from exc

    def _googledrive_setup(self, **config: Any) -> None:
        """Manage Google Drive files (delete/upload/mkdirs).

        Requires ``pydrive`` and a valid settings YAML.
        """
        from pydrive.auth import GoogleAuth
        from pydrive.drive import GoogleDrive

        settings_file = config.get("settings_file", "evaluation_examples/settings/googledrive/settings.yml")
        gauth = GoogleAuth(settings_file=settings_file)
        drive = GoogleDrive(gauth)

        def mkdir_in_googledrive(paths: list[str]) -> str:
            paths = [paths] if not isinstance(paths, list) else paths
            parent_id = "root"
            for p in paths:
                q = f'"{parent_id}" in parents and title = "{p}" and mimeType = "application/vnd.google-apps.folder" and trashed = false'
                folders = drive.ListFile({"q": q}).GetList()
                if not folders:
                    parents_meta: dict = {} if parent_id == "root" else {"parents": [{"id": parent_id}]}
                    folder_file = drive.CreateFile(
                        {"title": p, "mimeType": "application/vnd.google-apps.folder", **parents_meta}
                    )
                    folder_file.Upload()
                    parent_id = folder_file["id"]
                else:
                    parent_id = folders[0]["id"]
            return parent_id

        for oid, operation in enumerate(config["operation"]):
            params = config["args"][oid]
            if operation == "delete":
                q = params.get("query", "")
                trash = params.get("trash", False)
                for item in drive.ListFile(
                    {
                        "q": f"( {q} ) and mimeType != 'application/vnd.google-apps.folder'"
                        if q
                        else "mimeType != 'application/vnd.google-apps.folder'"
                    }
                ).GetList():
                    item.Trash() if trash else item.Delete()
                for item in drive.ListFile(
                    {
                        "q": f"( {q} ) and mimeType = 'application/vnd.google-apps.folder'"
                        if q
                        else "mimeType = 'application/vnd.google-apps.folder'"
                    }
                ).GetList():
                    item.Trash() if trash else item.Delete()
            elif operation == "mkdirs":
                mkdir_in_googledrive(params["path"])
            elif operation == "upload":
                with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmpf:
                    resp = requests.get(params["url"], stream=True)
                    resp.raise_for_status()
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            tmpf.write(chunk)
                paths = params["path"] if isinstance(params["path"], list) else [params["path"]]
                parent_id = mkdir_in_googledrive(paths[:-1])
                parents_meta = {} if parent_id == "root" else {"parents": [{"id": parent_id}]}
                gfile = drive.CreateFile({"title": paths[-1], **parents_meta})
                gfile.SetContentFile(tmpf.name)
                gfile.Upload()
            else:
                raise ValueError(f"Unknown googledrive operation: {operation!r}")

    def _launch_setup(self, command: Union[str, list[str]], shell: bool = False) -> None:
        if not command:
            raise ValueError("Empty launch command")
        if not shell and isinstance(command, str) and len(command.split()) > 1:
            command = command.split()
        if isinstance(command, list) and command[0] == "google-chrome" and self.use_proxy:
            command.append("--proxy-server=http://127.0.0.1:18888")
        payload = json.dumps({"command": command, "shell": shell})
        try:
            resp = requests.post(
                self.http_server_setup_root + "/launch",
                headers={"Content-Type": "application/json"},
                data=payload,
            )
            if resp.status_code != 200:
                logger.error("Failed to launch: %s", resp.text)
        except requests.RequestException as exc:
            logger.error("launch error: %s", exc)

    def _login_setup(self, **config: Any) -> None:
        """Login to a website (currently only Google Drive supported)."""
        remote_debugging_url = f"http://{self.guest.host}:{self.chromium_port}"
        with sync_playwright() as p:
            browser = self._connect_to_chrome(p, remote_debugging_url)
            context = browser.contexts[0]
            plat = config["platform"]

            if plat == "googledrive":
                page = context.new_page()
                try:
                    page.goto("https://drive.google.com/drive/my-drive", timeout=60000)
                except Exception:
                    logger.warning("Google Drive page load timed out")
                settings = json.load(open(config["settings_file"]))
                email, password = settings["email"], settings["password"]
                try:
                    page.wait_for_selector('input[type="email"]', state="visible", timeout=3000)
                    page.fill('input[type="email"]', email)
                    page.click("#identifierNext > div > button")
                    page.wait_for_selector('input[type="password"]', state="visible", timeout=5000)
                    page.fill('input[type="password"]', password)
                    page.click("#passwordNext > div > button")
                    page.wait_for_load_state("load", timeout=5000)
                except PlaywrightTimeout:
                    logger.error("Timeout during Google Drive login")
            else:
                raise NotImplementedError(f"Login platform {plat!r} not supported")

    def _open_setup(self, path: str) -> None:
        if not path:
            raise ValueError(f"Invalid open path: {path!r}")
        payload = json.dumps({"path": path})
        try:
            resp = requests.post(
                self.http_server_setup_root + "/open_file",
                headers={"Content-Type": "application/json"},
                data=payload,
                timeout=1810,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to open {path!r}: {exc}") from exc

    def _proxy_setup(self, client_password: str = "") -> None:
        """Configure a system-wide proxy via tinyproxy inside the VM.

        Requires a configured proxy pool (PROXY_CONFIG_FILE env var).
        """
        raise NotImplementedError(
            "Proxy setup requires the desktop_env AWS proxy pool which is not available in the "
            "bare QEMU backend. If you need proxy support, configure the VM's network settings "
            "directly or set up a proxy outside the QEMU manager."
        )

    def _sleep_setup(self, seconds: float) -> None:
        time.sleep(seconds)

    def _upload_file_setup(self, files: list[dict[str, str]]) -> None:
        for f in files:
            local_path = f["local_path"]
            path = f["path"]
            if not os.path.exists(local_path):
                raise ValueError(f"Local file not found: {local_path!r}")
            for attempt in range(3):
                try:
                    with open(local_path, "rb") as fp:
                        form = MultipartEncoder({"file_path": path, "file_data": (os.path.basename(path), fp)})
                        resp = requests.post(
                            self.http_server_setup_root + "/upload",
                            headers={"Content-Type": form.content_type},
                            data=form,
                            timeout=(10, 600),
                        )
                        if resp.status_code == 200:
                            break
                        last_err: Exception = requests.RequestException(
                            f"Upload failed ({resp.status_code}): {resp.text}"
                        )
                except requests.RequestException as exc:
                    last_err = exc
                    time.sleep(2**attempt)
            else:
                raise last_err  # type: ignore[possibly-undefined]

    def _update_browse_history_setup(self, **config: Any) -> None:
        cache_path = os.path.join(self.cache_dir, "history_new.sqlite")
        db_url = "https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/chrome/44ee5668-ecd5-4366-a6ce-c1c9b8d4e938/history_empty.sqlite?download=true"

        if not os.path.exists(cache_path):
            for attempt in range(3):
                try:
                    resp = requests.get(db_url, stream=True)
                    resp.raise_for_status()
                    with open(cache_path, "wb") as fp:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                fp.write(chunk)
                    break
                except requests.RequestException as exc:
                    logger.error("History DB download attempt %d failed: %s", attempt + 1, exc)
            else:
                raise requests.RequestException(f"Failed to download history DB from {db_url}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "history_empty.sqlite")
            shutil.copy(cache_path, db_path)

            for item in config["history"]:
                url = item["url"]
                title = item["title"]
                visit_time = datetime.now() - timedelta(seconds=item["visit_time_from_now_in_seconds"])
                chrome_ts = int((visit_time - datetime(1601, 1, 1)).total_seconds() * 1_000_000)

                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO urls (url, title, visit_count, typed_count, last_visit_time, hidden) VALUES (?, ?, ?, ?, ?, ?)",
                    (url, title, 1, 0, chrome_ts, 0),
                )
                url_id = cursor.lastrowid
                cursor.execute(
                    "INSERT INTO visits (url, visit_time, from_visit, transition, segment_id, visit_duration) VALUES (?, ?, ?, ?, ?, ?)",
                    (url_id, chrome_ts, 0, 805306368, 0, 0),
                )
                conn.commit()
                conn.close()

            os_type = self.guest.get_vm_platform()
            if os_type == "Windows":
                result = self.guest.execute_python_command(
                    r"""import os; print(os.path.join(os.getenv('USERPROFILE'), 'AppData', 'Local', 'Google', 'Chrome', 'User Data', 'Default', 'History'))"""
                )
            elif os_type == "Linux":
                if "arm" in platform.machine():
                    result = self.guest.execute_python_command(
                        "import os; print(os.path.join(os.getenv('HOME'), 'snap', 'chromium', 'common', 'chromium', 'Default', 'History'))"
                    )
                else:
                    result = self.guest.execute_python_command(
                        "import os; print(os.path.join(os.getenv('HOME'), '.config', 'google-chrome', 'Default', 'History'))"
                    )
            elif os_type == "Darwin":
                result = self.guest.execute_python_command(
                    r"""import os; print(os.path.join(os.getenv('HOME'), 'Library', 'Application Support', 'Google', 'Chrome', 'Default', 'History'))"""
                )
            else:
                raise RuntimeError(f"Unsupported OS type: {os_type!r}")

            chrome_history_path = result["output"].strip() if result else ""

            form = MultipartEncoder(
                {
                    "file_path": chrome_history_path,
                    "file_data": (os.path.basename(chrome_history_path), open(db_path, "rb")),
                }
            )
            resp = requests.post(
                self.http_server_setup_root + "/upload",
                headers={"Content-Type": form.content_type},
                data=form,
            )
            if resp.status_code != 200:
                logger.error("Failed to upload history DB: %s", resp.text)

            self._execute_setup(
                ["sudo chown -R user:user /home/user/.config/google-chrome/Default/History"], shell=True
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _replace_screen_env(self, command: Union[str, list[str]]) -> Union[str, list[str]]:
        replacements = {
            "{CLIENT_PASSWORD}": self.client_password,
            "{SCREEN_WIDTH}": str(self.screen_width),
            "{SCREEN_HEIGHT}": str(self.screen_height),
            "{SCREEN_WIDTH_HALF}": str(self.screen_width // 2),
            "{SCREEN_HEIGHT_HALF}": str(self.screen_height // 2),
        }
        if isinstance(command, str):
            for k, v in replacements.items():
                command = command.replace(k, v)
            return command
        return [
            token.replace("{CLIENT_PASSWORD}", replacements["{CLIENT_PASSWORD}"])
            .replace("{SCREEN_WIDTH_HALF}", replacements["{SCREEN_WIDTH_HALF}"])
            .replace("{SCREEN_HEIGHT_HALF}", replacements["{SCREEN_HEIGHT_HALF}"])
            .replace("{SCREEN_WIDTH}", replacements["{SCREEN_WIDTH}"])
            .replace("{SCREEN_HEIGHT}", replacements["{SCREEN_HEIGHT}"])
            for token in command
        ]

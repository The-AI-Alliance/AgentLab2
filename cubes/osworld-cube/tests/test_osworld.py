"""Tests for osworld_cube — verifies compliance with the CUBE protocol ABCs."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import importlib.util
import json

from PIL import Image

from cube import LocalInfraConfig
from cube.core import Action, Observation, TextContent
from cube.resource import InfraConfig, ResourceHandle

# ---------------------------------------------------------------------------
# Patch targets — pointing at cube_computer_tool and osworld_cube.task
# ---------------------------------------------------------------------------

PATCH_GUEST_AGENT = "cube_computer_tool.computer.GuestAgent"
PATCH_EVALUATOR = "osworld_cube.task.Evaluator"
PATCH_SETUP_CTRL = "osworld_cube.task.SetupController"
PATCH_SLEEP = "osworld_cube.task.time.sleep"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_screenshot_bytes(w: int = 100, h: int = 100) -> bytes:
    """Return a minimal PNG screenshot as bytes."""
    img = Image.new("RGB", (w, h), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_mock_guest(screenshot: bytes | None = None, axtree: str = "<root/>") -> MagicMock:
    """Return a Mock that looks like GuestAgent."""
    guest = MagicMock()
    guest.get_screenshot.return_value = screenshot or _make_screenshot_bytes()
    guest.get_accessibility_tree.return_value = axtree
    guest.get_terminal_output.return_value = ""
    guest.execute_action.return_value = None
    guest.execute_python_command.return_value = {"returncode": 0, "output": ""}
    return guest


# ---------------------------------------------------------------------------
# ComputerConfig
# ---------------------------------------------------------------------------


class TestComputerConfig:
    def test_defaults(self) -> None:
        from osworld_cube.computer import ComputerConfig

        cfg = ComputerConfig()
        assert cfg.require_a11y_tree is True
        assert cfg.observe_after_action is True

    def test_action_space_default(self) -> None:
        from osworld_cube.computer import ActionSpace, ComputerConfig

        cfg = ComputerConfig()
        assert cfg.action_space == ActionSpace.COMPUTER_13


class TestDebugBenchmark:
    def test_get_debug_benchmark_returns_config(self) -> None:
        """``get_debug_benchmark`` returns a ``DebugOSWorldBenchmarkConfig`` per the
        cube-standard rc7 protocol — infra is no longer a config field; it is
        passed via ``config.make(infra)``."""
        from osworld_cube.debug import DebugOSWorldBenchmarkConfig, get_debug_benchmark

        cfg = get_debug_benchmark()
        assert isinstance(cfg, DebugOSWorldBenchmarkConfig)
        assert cfg.num_tasks == 2

    def test_default_infra_resolves_to_local(self) -> None:
        """The ``__main__`` path calls ``_get_default_infra`` which falls back to
        ``LocalInfraConfig`` when no env-var override is set."""
        import os

        from osworld_cube.debug import _get_default_infra

        os.environ.pop("OSWORLD_CUBE_TEST_INFRA_CONFIG_FILE", None)
        infra = _get_default_infra()
        assert isinstance(infra, LocalInfraConfig)


# ---------------------------------------------------------------------------
# Computer
# ---------------------------------------------------------------------------


class TestComputer:
    def test_make_without_vm_succeeds(self) -> None:
        """ComputerConfig.make() with no vm creates a tool in unattached state."""
        from osworld_cube.computer import ComputerConfig

        cfg = ComputerConfig()
        computer = cfg.make()
        assert computer.config is not None
        assert computer._guest is None

    def test_action_set_computer13(self) -> None:
        from osworld_cube.computer import ComputerConfig

        computer = ComputerConfig(action_space="computer_13").make()
        names = {a.name for a in computer.action_set}
        for expected in (
            "click",
            "double_click",
            "right_click",
            "drag_to",
            "scroll",
            "typing",
            "press",
            "hotkey",
            "wait",
            "done",
            "fail",
        ):
            assert expected in names, f"Missing action: {expected}"
        assert "run_pyautogui" not in names

    def test_action_set_pyautogui(self) -> None:
        from osworld_cube.computer import ComputerConfig

        computer = ComputerConfig(action_space="pyautogui").make()
        names = {a.name for a in computer.action_set}
        assert "run_pyautogui" in names
        for terminal in ("wait", "done", "fail"):
            assert terminal in names, f"Missing action: {terminal}"
        assert "click" not in names

    def test_attach_endpoint_connects_guest(self) -> None:
        from osworld_cube.computer import ComputerConfig

        computer = ComputerConfig().make()

        with patch(PATCH_GUEST_AGENT) as mock_ga_cls:
            mock_ga_cls.return_value = MagicMock()
            computer.attach_endpoint("http://host:5000")

        mock_ga_cls.assert_called_once_with(host="host", port=5000)
        assert computer._guest is not None

    def test_click_dispatches_to_guest(self) -> None:
        from osworld_cube.computer import ComputerConfig

        computer = ComputerConfig(observe_after_action=False).make()
        mock_guest = _make_mock_guest()
        computer._guest = mock_guest

        result = computer.click(x=100, y=200)
        assert result == "Success"
        mock_guest.execute_action.assert_called_once()
        call_args = mock_guest.execute_action.call_args[0][0]
        assert call_args["action_type"] == "CLICK"
        assert call_args["parameters"]["x"] == 100
        assert call_args["parameters"]["y"] == 200

    def test_done_sets_is_done(self) -> None:
        from osworld_cube.computer import ComputerConfig

        computer = ComputerConfig(observe_after_action=False).make()
        assert computer._is_done is False
        computer.done()
        assert computer._is_done is True

    def test_fail_sets_is_done(self) -> None:
        from osworld_cube.computer import ComputerConfig

        computer = ComputerConfig(observe_after_action=False).make()
        computer.fail()
        assert computer._is_done is True

    def test_update_marks_and_run_pyautogui(self) -> None:
        from osworld_cube.computer import ComputerConfig

        computer = ComputerConfig(action_space="pyautogui", observe_after_action=False).make()
        mock_guest = _make_mock_guest()
        computer._guest = mock_guest

        computer.update_marks([[10, 20, 30, 40], [50, 60, 10, 10]])
        computer.run_pyautogui("pyautogui.click(*tag_1)")

        # tag_1 center: (10 + 30//2, 20 + 40//2) = (25, 40)
        # tag_2 center: (50 + 10//2, 60 + 10//2) = (55, 65)
        call_code = mock_guest.execute_python_command.call_args[0][0]
        assert "tag_1 = (25, 40)" in call_code
        assert "tag_2 = (55, 65)" in call_code
        assert "pyautogui.click(*tag_1)" in call_code

    def test_execute_action_dispatches_via_cube(self) -> None:
        """cube.tool.Tool.execute_action routes by action name to the correct @tool_action."""
        from osworld_cube.computer import ComputerConfig

        computer = ComputerConfig(observe_after_action=False).make()
        mock_guest = _make_mock_guest()
        computer._guest = mock_guest

        result = computer.execute_action(Action(name="typing", arguments={"text": "hello"}))
        assert isinstance(result, Observation)
        mock_guest.execute_action.assert_called_once()

    def test_close_does_not_stop_vm(self) -> None:
        """ComputerBase.close() must NOT tear down the VM -- caller owns VM lifecycle.

        The tool no longer holds a VM reference (provisioning is owned by the
        InfraConfig/ResourceHandle path); close() must be a pure no-op that does
        not touch the attached guest connection.
        """
        from osworld_cube.computer import ComputerConfig

        computer = ComputerConfig().make()
        mock_guest = _make_mock_guest()
        computer._guest = mock_guest
        computer.close()
        assert computer._guest is mock_guest
        mock_guest.stop.assert_not_called()


# ---------------------------------------------------------------------------
# OSWorldTask helpers
# ---------------------------------------------------------------------------


def _make_task_metadata(task_id: str = "t1", instruction: str = "Do something"):
    from osworld_cube.task import OSWorldTaskMetadata

    return OSWorldTaskMetadata(
        id=task_id,
        abstract_description="",
        instruction=instruction,
        domain="os",
        test_sets=["test_all"],
        snapshot="init_state",
        os_type="ubuntu",
        related_apps=[],
    )


def _make_execution_info(evaluator: dict | None = None):
    """Default execution info for OSWorldTask tests — empty config, simple evaluator."""
    from osworld_cube.task import OSWorldExecutionInfo

    return OSWorldExecutionInfo(
        config=[],
        evaluator=evaluator if evaluator is not None else {"func": "check_file", "expected": {}},
    )


def _make_mock_handle(server_port: int = 15000) -> MagicMock:
    """Return a Mock that looks like a ResourceHandle."""
    handle = MagicMock(spec=ResourceHandle)
    handle.endpoint = f"http://localhost:{server_port}"
    handle.run_id = "test-run-id-1234"
    return handle


# ---------------------------------------------------------------------------
# OSWorldTask
# ---------------------------------------------------------------------------


class TestOSWorldTask:
    def test_model_post_init_creates_tool_without_vm(self) -> None:
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldTask

        task = OSWorldTask(
            metadata=_make_task_metadata(),
            execution_info=_make_execution_info(),
            tool_config=ComputerConfig(),
        )
        assert task._computer is not None
        assert task._computer._guest is None
        assert task._handle is None

    def test_reset_with_infra_launches_vm(self) -> None:
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldTask

        mock_handle = _make_mock_handle()
        mock_infra = MagicMock(spec=InfraConfig)
        mock_infra.launch.return_value = mock_handle

        with (
            patch(PATCH_GUEST_AGENT) as mock_ga_cls,
            patch(PATCH_SETUP_CTRL),
            patch(PATCH_SLEEP),
        ):
            mock_guest = _make_mock_guest()
            mock_ga_cls.return_value = mock_guest

            task = OSWorldTask(
                metadata=_make_task_metadata(),
                execution_info=_make_execution_info(),
                tool_config=ComputerConfig(),
                runtime_context={"infra": mock_infra},
            )
            obs, info = task.reset()

        mock_infra.launch.assert_called_once()
        assert task._handle is mock_handle
        assert isinstance(obs, Observation)

    def test_reset_returns_obs_and_info(self) -> None:
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldTask

        task = OSWorldTask(
            metadata=_make_task_metadata(),
            execution_info=_make_execution_info(),
            tool_config=ComputerConfig(),
        )
        mock_handle = _make_mock_handle()
        task._handle = mock_handle

        with (
            patch(PATCH_GUEST_AGENT) as mock_ga_cls,
            patch(PATCH_SETUP_CTRL),
            patch(PATCH_SLEEP),
        ):
            mock_guest = _make_mock_guest()
            mock_ga_cls.return_value = mock_guest
            task._computer.attach_endpoint(mock_handle.endpoint)

            obs, info = task.reset()

        assert isinstance(obs, Observation)
        texts = [c.data for c in obs.contents if isinstance(c, TextContent)]
        assert any("Do something" in t for t in texts)
        assert info["task_id"] == "t1"
        assert info["task_domain"] == "os"

    def test_reset_resets_is_done(self) -> None:
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldTask

        task = OSWorldTask(
            metadata=_make_task_metadata(),
            execution_info=_make_execution_info(),
            tool_config=ComputerConfig(),
        )
        mock_handle = _make_mock_handle()
        task._handle = mock_handle

        with (
            patch(PATCH_GUEST_AGENT) as mock_ga_cls,
            patch(PATCH_SETUP_CTRL),
            patch(PATCH_SLEEP),
        ):
            mock_ga_cls.return_value = _make_mock_guest()
            task._computer.attach_endpoint(mock_handle.endpoint)
            task._computer._is_done = True
            task.reset()
            assert task._computer._is_done is False

    def test_evaluate_calls_evaluator(self) -> None:
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldTask

        task = OSWorldTask(
            metadata=_make_task_metadata(),
            execution_info=_make_execution_info({"func": "check_file"}),
            tool_config=ComputerConfig(),
        )
        task._handle = _make_mock_handle()
        task._computer._guest = _make_mock_guest()

        with patch(PATCH_EVALUATOR) as mock_eval_cls:
            mock_eval_cls.return_value.evaluate.return_value = 0.5
            reward, info = task.evaluate(Observation.from_text("state"))

        assert reward == 0.5
        assert "evaluator" in info

    def test_evaluate_no_evaluator_returns_zero(self) -> None:
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldExecutionInfo, OSWorldTask, OSWorldTaskMetadata

        task = OSWorldTask(
            metadata=OSWorldTaskMetadata(
                id="no-eval",
                abstract_description="",
                instruction="",
                domain="os",
                test_sets=[],
                snapshot="init_state",
                os_type="ubuntu",
                related_apps=[],
            ),
            execution_info=OSWorldExecutionInfo(config=[], evaluator={}),
            tool_config=ComputerConfig(),
        )
        reward, info = task.evaluate(Observation())
        assert reward == 0.0
        assert info.get("error") == "no_evaluator"

    def test_finished_reflects_computer_is_done(self) -> None:
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldTask

        task = OSWorldTask(
            metadata=_make_task_metadata(),
            execution_info=_make_execution_info(),
            tool_config=ComputerConfig(),
        )
        assert task.finished(Observation()) is False
        task._computer._is_done = True
        assert task.finished(Observation()) is True

    def test_step_done_action_triggers_evaluate(self) -> None:
        """Full step loop: agent calls done() -> task.step() -> EnvironmentOutput.done is True."""
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldTask

        task = OSWorldTask(
            metadata=_make_task_metadata(),
            execution_info=_make_execution_info(),
            tool_config=ComputerConfig(observe_after_action=False),
        )
        mock_handle = _make_mock_handle()
        task._handle = mock_handle

        with (
            patch(PATCH_GUEST_AGENT) as mock_ga_cls,
            patch(PATCH_SETUP_CTRL),
            patch(PATCH_SLEEP),
            patch(PATCH_EVALUATOR) as mock_eval_cls,
        ):
            mock_ga_cls.return_value = _make_mock_guest()
            mock_eval_cls.return_value.evaluate.return_value = 1.0
            task._computer.attach_endpoint(mock_handle.endpoint)
            task.reset()

            env_out = task.step(Action(name="done", arguments={}))

        assert env_out.done is True
        assert env_out.reward == 1.0

    def test_step_click_not_done(self) -> None:
        """A regular action does not set done."""
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldTask

        task = OSWorldTask(
            metadata=_make_task_metadata(),
            execution_info=_make_execution_info(),
            tool_config=ComputerConfig(observe_after_action=False),
        )
        mock_handle = _make_mock_handle()
        task._handle = mock_handle

        with (
            patch(PATCH_GUEST_AGENT) as mock_ga_cls,
            patch(PATCH_SETUP_CTRL),
            patch(PATCH_SLEEP),
        ):
            mock_ga_cls.return_value = _make_mock_guest()
            task._computer.attach_endpoint(mock_handle.endpoint)
            task.reset()

            env_out = task.step(Action(name="click", arguments={"x": 10, "y": 20}))
        assert env_out.done is False

    def test_close_closes_handle(self) -> None:
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldTask

        task = OSWorldTask(
            metadata=_make_task_metadata(),
            execution_info=_make_execution_info(),
            tool_config=ComputerConfig(),
        )
        mock_handle = _make_mock_handle()
        task._handle = mock_handle
        task._computer._guest = _make_mock_guest()

        task.close()
        mock_handle.close.assert_called_once()
        assert task._handle is None


# ---------------------------------------------------------------------------
# OSWorldTestSet
# ---------------------------------------------------------------------------


class TestOSWorldTestSet:
    def test_enum_values_are_filenames(self) -> None:
        from osworld_cube.benchmark import OSWorldTestSet

        assert OSWorldTestSet.TEST_ALL.value == "test_all.json"
        assert OSWorldTestSet.TEST_INFEASIBLE.value == "test_infeasible.json"
        assert OSWorldTestSet.TEST_NOGDRIVE.value == "test_nogdrive.json"
        assert OSWorldTestSet.TEST_SMALL.value == "test_small.json"

    def test_enum_is_str_subclass(self) -> None:
        from osworld_cube.benchmark import OSWorldTestSet

        assert isinstance(OSWorldTestSet.TEST_ALL, str)
        assert OSWorldTestSet.TEST_ALL == "test_all.json"

    def test_enum_from_string(self) -> None:
        from osworld_cube.benchmark import OSWorldTestSet

        assert OSWorldTestSet("test_small.json") is OSWorldTestSet.TEST_SMALL


# ---------------------------------------------------------------------------
# OSWorldBenchmark
# ---------------------------------------------------------------------------


class TestOSWorldBenchmark:
    def test_benchmark_metadata(self) -> None:
        from osworld_cube.benchmark import OSWorldBenchmarkConfig

        assert OSWorldBenchmarkConfig.benchmark_metadata.name == "osworld-cube"
        assert OSWorldBenchmarkConfig.task_config_class.__name__ == "OSWorldTaskConfig"
        assert OSWorldBenchmarkConfig.benchmark_class.__name__ == "OSWorldBenchmark"

    def test_load_all_tasks_from_shipped_metadata(self) -> None:
        from osworld_cube.benchmark import OSWorldBenchmarkConfig

        assert len(OSWorldBenchmarkConfig.task_metadata) > 0
        assert all(tm.test_sets for tm in OSWorldBenchmarkConfig.task_metadata.values())

    def test_domain_filter_via_subset_from_glob(self) -> None:
        from osworld_cube.benchmark import OSWorldBenchmarkConfig

        cfg = OSWorldBenchmarkConfig()
        chrome_cfg = cfg.subset_from_glob("domain", "chrome")
        assert isinstance(chrome_cfg, OSWorldBenchmarkConfig)
        assert len(chrome_cfg.tasks()) > 0
        assert len(chrome_cfg.tasks()) < len(cfg.tasks())
        assert all(tm.domain == "chrome" for tm in chrome_cfg.tasks().values())

    def test_named_subset_test_small(self) -> None:
        from osworld_cube.benchmark import OSWorldBenchmarkConfig

        cfg = OSWorldBenchmarkConfig()
        small_cfg = cfg.named_subset("test_small")
        assert isinstance(small_cfg, OSWorldBenchmarkConfig)
        assert len(small_cfg.tasks()) > 0
        assert len(small_cfg.tasks()) < len(cfg.tasks())
        assert all("test_small" in tm.test_sets for tm in small_cfg.tasks().values())

    def test_get_task_configs_carries_metadata(self) -> None:
        from osworld_cube.benchmark import OSWorldBenchmarkConfig, OSWorldTaskConfig

        cfg = OSWorldBenchmarkConfig()
        configs = list(cfg.get_task_configs())
        assert len(configs) == len(cfg.tasks()) == 368

        for tc in configs:
            assert isinstance(tc, OSWorldTaskConfig)
            assert tc.task_id in cfg.tasks()
            # metadata travels with the config — no driver-side lookup required
            assert tc.metadata.id == tc.task_id

    def test_task_config_make_produces_osworld_task(self) -> None:
        from osworld_cube.benchmark import OSWorldBenchmarkConfig, OSWorldTaskConfig
        from osworld_cube.task import OSWorldExecutionInfo, OSWorldTask

        cfg = OSWorldBenchmarkConfig()
        tc = next(cfg.get_task_configs())
        fake_exec_info = {"config": [], "evaluator": {"func": "check_file"}}
        with (
            patch.object(OSWorldTaskConfig, "verify_installed"),
            patch.object(OSWorldTaskConfig, "load_task_execution_info", return_value=fake_exec_info),
        ):
            task = tc.make()

        assert isinstance(task, OSWorldTask)
        assert task.metadata.id == tc.task_id
        assert isinstance(task.execution_info, OSWorldExecutionInfo)
        assert task.execution_info.config == []
        assert task.execution_info.evaluator == {"func": "check_file"}

    def test_test_all_has_all_tasks(self) -> None:
        from osworld_cube.benchmark import OSWorldBenchmarkConfig

        cfg = OSWorldBenchmarkConfig().named_subset("test_all")
        assert len(cfg.tasks()) == len(OSWorldBenchmarkConfig.task_metadata)

    def test_use_som_propagates_to_task_configs(self) -> None:
        from osworld_cube.benchmark import OSWorldBenchmarkConfig, OSWorldTaskConfig

        cfg = OSWorldBenchmarkConfig(use_som=True)
        tc = next(cfg.get_task_configs())
        assert isinstance(tc, OSWorldTaskConfig)
        assert tc.use_som is True

    def test_make_with_local_infra_publishes_runtime_context(self) -> None:
        """``OSWorldBenchmarkConfig.make(infra)`` constructs the runtime pair and
        publishes infra into ``runtime_context['infra']`` so per-task VM launches
        flow naturally through ``Task.runtime_context``."""
        from osworld_cube.benchmark import OSWorldBenchmarkConfig

        mock_infra = MagicMock(spec=InfraConfig)
        mock_infra.provision_status.return_value = "ready"
        mock_infra.fingerprint.return_value = "mock-fp"

        cfg = OSWorldBenchmarkConfig()
        bench = cfg.make(infra=mock_infra)
        try:
            assert bench._runtime_context.get("infra") is mock_infra
        finally:
            bench.close()

    def test_setup_invokes_cleanup_stale(self) -> None:
        """Benchmark setup must sweep stale resources from previous runs before
        launching new VMs. Skipping this lets crashed-task orphans accumulate and
        bill until manual cleanup."""
        from osworld_cube.benchmark import OSWorldBenchmarkConfig

        mock_infra = MagicMock(spec=InfraConfig)
        mock_infra.provision_status.return_value = "ready"
        mock_infra.cleanup_stale.return_value = []

        cfg = OSWorldBenchmarkConfig()
        bench = cfg.make(infra=mock_infra)
        try:
            mock_infra.cleanup_stale.assert_called_once()
        finally:
            bench.close()

    def test_setup_without_infra_does_not_call_cleanup_stale(self) -> None:
        """When no infra is configured (local-only runs), ``_setup`` must not
        attempt to sweep — there is nothing to sweep against."""
        from osworld_cube.benchmark import OSWorldBenchmark, OSWorldBenchmarkConfig

        bench = OSWorldBenchmark(config=OSWorldBenchmarkConfig())
        bench.setup()
        try:
            assert bench._infra is None
        finally:
            bench.close()

    def test_close_does_not_raise(self) -> None:
        """Closing a fresh runtime pair must succeed without errors."""
        from osworld_cube.benchmark import OSWorldBenchmark, OSWorldBenchmarkConfig

        bench = OSWorldBenchmark(config=OSWorldBenchmarkConfig())
        bench.setup()
        bench.close()

    def test_round_trip_serialization(self) -> None:
        """``OSWorldBenchmarkConfig`` serializes and deserializes with all instance
        fields preserved (rc7 cross-process transport contract)."""
        from osworld_cube.benchmark import OSWorldBenchmarkConfig

        original = OSWorldBenchmarkConfig(use_som=True)
        dumped = original.model_dump_json()
        restored = OSWorldBenchmarkConfig.model_validate_json(dumped)
        assert restored.use_som is True
        assert len(restored.tasks()) == len(original.tasks())


# ---------------------------------------------------------------------------
# scripts/create_task_metadata.py
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "create_task_metadata.py"


def _load_script():
    """Import create_task_metadata.py as a module (path-based, no package needed)."""
    spec = importlib.util.spec_from_file_location("create_task_metadata", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _make_fake_repo(root: Path) -> Path:
    """Create a minimal OSWorld repo structure under *root* with 2 tasks."""
    repo = root / "OSWorld"
    eval_dir = repo / "evaluation_examples"
    examples_dir = eval_dir / "examples" / "os"
    examples_dir.mkdir(parents=True)

    # Two tasks both in test_all; task-b also in test_small
    (eval_dir / "test_all.json").write_text(json.dumps({"os": ["task-a", "task-b"]}))
    (eval_dir / "test_small.json").write_text(json.dumps({"os": ["task-b"]}))
    (eval_dir / "test_nogdrive.json").write_text(json.dumps({}))
    (eval_dir / "test_infeasible.json").write_text(json.dumps({}))

    for task_id, instr in [("task-a", "Do A"), ("task-b", "Do B")]:
        (examples_dir / f"{task_id}.json").write_text(
            json.dumps(
                {
                    "id": task_id,
                    "instruction": instr,
                    "snapshot": "init_state",
                    "os_type": "ubuntu",
                    "related_apps": ["os"],
                    "config": [],
                    "evaluator": {"func": "check_include_exclude"},
                }
            )
        )
    return repo


class TestCreateTaskMetadata:
    def test_generates_file_from_repo(self, tmp_path: Path) -> None:
        repo = _make_fake_repo(tmp_path)
        out = tmp_path / "task_metadata.json"

        mod = _load_script()
        count = mod.generate_task_metadata(repo_dir=repo, output_path=out)

        assert count == 2
        assert out.exists()
        tasks = json.loads(out.read_text())
        assert len(tasks) == 2
        ids = {t["id"] for t in tasks}
        assert ids == {"task-a", "task-b"}

    def test_correct_typed_fields(self, tmp_path: Path) -> None:
        repo = _make_fake_repo(tmp_path)
        out = tmp_path / "task_metadata.json"

        mod = _load_script()
        mod.generate_task_metadata(repo_dir=repo, output_path=out)

        tasks = {t["id"]: t for t in json.loads(out.read_text())}
        assert tasks["task-a"]["domain"] == "os"
        assert tasks["task-a"]["instruction"] == "Do A"
        assert tasks["task-a"]["snapshot"] == "init_state"
        assert tasks["task-a"]["os_type"] == "ubuntu"
        # ``extra_info`` was removed in cube-standard rc7 — heavy data lives on
        # OSWorldExecutionInfo, populated lazily on the worker.
        assert "extra_info" not in tasks["task-a"]
        assert "test_all" in tasks["task-a"]["test_sets"]
        assert "test_small" not in tasks["task-a"]["test_sets"]
        assert "test_small" in tasks["task-b"]["test_sets"]

    def test_idempotent_skips_existing(self, tmp_path: Path) -> None:
        repo = _make_fake_repo(tmp_path)
        out = tmp_path / "task_metadata.json"

        mod = _load_script()
        count1 = mod.generate_task_metadata(repo_dir=repo, output_path=out)
        mtime_after_first = out.stat().st_mtime

        count2 = mod.generate_task_metadata(repo_dir=repo, output_path=out)

        assert count1 == 2
        assert count2 == 0
        assert out.stat().st_mtime == mtime_after_first  # file untouched

    def test_force_overwrites_existing(self, tmp_path: Path) -> None:
        repo = _make_fake_repo(tmp_path)
        out = tmp_path / "task_metadata.json"

        mod = _load_script()
        mod.generate_task_metadata(repo_dir=repo, output_path=out)
        count = mod.generate_task_metadata(repo_dir=repo, output_path=out, force=True)

        assert count == 2

    def test_missing_repo_raises_when_clone_disabled(self, tmp_path: Path) -> None:
        import pytest

        out = tmp_path / "task_metadata.json"

        mod = _load_script()
        with pytest.raises(RuntimeError, match="OSWorld repo not found"):
            mod.generate_task_metadata(
                repo_dir=tmp_path / "nonexistent",
                output_path=out,
                clone_if_missing=False,
            )

    def test_no_config_or_evaluator_in_output(self, tmp_path: Path) -> None:
        """task_metadata.json must not contain config or evaluator (execution data)."""
        repo = _make_fake_repo(tmp_path)
        out = tmp_path / "task_metadata.json"

        mod = _load_script()
        mod.generate_task_metadata(repo_dir=repo, output_path=out)

        tasks = json.loads(out.read_text())
        for task in tasks:
            assert "config" not in task
            assert "evaluator" not in task


# ---------------------------------------------------------------------------
# OSWorldBenchmark.install() / uninstall()
# ---------------------------------------------------------------------------


def _make_fake_task_metadata(tmp_path: Path) -> dict:
    """Return a minimal task_metadata dict with one task for install() tests."""
    from osworld_cube.task import OSWorldTaskMetadata

    tm = OSWorldTaskMetadata(
        id="task-a",
        abstract_description="",
        instruction="Do A",
        domain="os",
        test_sets=["test_all"],
        snapshot="init_state",
        os_type="ubuntu",
        related_apps=["os"],
    )
    return {tm.id: tm}


def _install_patches(repo: Path, cache_dir: Path, fake_tm: dict):
    """Return a list of context managers that isolate install() from the real filesystem."""
    from osworld_cube.benchmark import OSWorldBenchmarkConfig, OSWorldTaskConfig

    return [
        patch.object(OSWorldBenchmarkConfig, "task_metadata", fake_tm),
        patch("osworld_cube.benchmark.OSWORLD_REPO_DIR", repo),
        patch.object(OSWorldTaskConfig, "task_execution_cache_dir", return_value=cache_dir),
        patch("osworld_cube.benchmark.ensure_proxy_config_in_env"),
        patch("osworld_cube.benchmark.load_dotenv"),
    ]


class TestInstall:
    def test_install_populates_execution_cache(self, tmp_path: Path) -> None:
        from contextlib import ExitStack

        from osworld_cube.benchmark import OSWorldBenchmarkConfig

        repo = _make_fake_repo(tmp_path)
        cache_dir = tmp_path / "cache"
        fake_tm = _make_fake_task_metadata(tmp_path)

        with ExitStack() as stack:
            for p in _install_patches(repo, cache_dir, fake_tm):
                stack.enter_context(p)
            OSWorldBenchmarkConfig.install()

        assert (cache_dir / "task-a.json").exists()
        exec_info = json.loads((cache_dir / "task-a.json").read_text())
        assert "config" in exec_info
        assert "evaluator" in exec_info

    def test_install_is_idempotent(self, tmp_path: Path) -> None:
        from contextlib import ExitStack

        from osworld_cube.benchmark import OSWorldBenchmarkConfig

        repo = _make_fake_repo(tmp_path)
        cache_dir = tmp_path / "cache"
        fake_tm = _make_fake_task_metadata(tmp_path)

        with ExitStack() as stack:
            for p in _install_patches(repo, cache_dir, fake_tm):
                stack.enter_context(p)
            OSWorldBenchmarkConfig.install()
            mtime = (cache_dir / "task-a.json").stat().st_mtime
            OSWorldBenchmarkConfig.install()  # second call

        assert (cache_dir / "task-a.json").stat().st_mtime == mtime  # file untouched

    def test_install_does_not_write_task_metadata_json(self, tmp_path: Path) -> None:
        from contextlib import ExitStack

        from osworld_cube.benchmark import OSWorldBenchmarkConfig

        repo = _make_fake_repo(tmp_path)
        cache_dir = tmp_path / "cache"
        metadata_json = tmp_path / "task_metadata.json"
        fake_tm = _make_fake_task_metadata(tmp_path)

        with ExitStack() as stack:
            for p in _install_patches(repo, cache_dir, fake_tm):
                stack.enter_context(p)
            OSWorldBenchmarkConfig.install()

        assert not metadata_json.exists()

    def test_uninstall_removes_execution_cache_and_repo(self, tmp_path: Path) -> None:
        from osworld_cube.benchmark import OSWorldBenchmarkConfig, OSWorldTaskConfig

        repo = _make_fake_repo(tmp_path)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "task-a.json").write_text("{}")

        with (
            patch("osworld_cube.benchmark.OSWORLD_REPO_DIR", repo),
            patch.object(OSWorldTaskConfig, "task_execution_cache_dir", return_value=cache_dir),
        ):
            OSWorldBenchmarkConfig.uninstall()

        assert not cache_dir.exists()
        assert not repo.exists()

    def test_uninstall_does_not_remove_task_metadata_json(self, tmp_path: Path) -> None:
        from osworld_cube.benchmark import OSWorldBenchmarkConfig, OSWorldTaskConfig

        repo = _make_fake_repo(tmp_path)
        cache_dir = tmp_path / "cache"
        metadata_json = tmp_path / "task_metadata.json"
        metadata_json.write_text("[]")

        with (
            patch("osworld_cube.benchmark.OSWORLD_REPO_DIR", repo),
            patch.object(OSWorldTaskConfig, "task_execution_cache_dir", return_value=cache_dir),
        ):
            OSWorldBenchmarkConfig.uninstall()

        assert metadata_json.exists()

    def test_install_raises_when_task_metadata_empty(self, tmp_path: Path) -> None:
        import pytest

        from osworld_cube.benchmark import OSWorldBenchmarkConfig

        with (
            patch.object(OSWorldBenchmarkConfig, "task_metadata", {}),
        ):
            with pytest.raises(RuntimeError, match="task_metadata is empty"):
                OSWorldBenchmarkConfig.install()

    def test_verify_installed_raises_when_cache_empty(self, tmp_path: Path) -> None:
        import pytest

        from osworld_cube.benchmark import OSWorldTaskConfig

        empty_cache = tmp_path / "cache"
        cfg = OSWorldTaskConfig(metadata=_make_task_metadata())
        with patch.object(OSWorldTaskConfig, "task_execution_cache_dir", return_value=empty_cache):
            with pytest.raises(RuntimeError, match="execution cache is empty"):
                cfg.verify_installed()

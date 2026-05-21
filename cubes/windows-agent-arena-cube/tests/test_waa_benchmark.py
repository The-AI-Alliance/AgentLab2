"""Tests for waa_cube — verifies compliance with the CUBE protocol ABCs."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

from cube.core import Action, Observation, TextContent
from cube.resource import ResourceHandle
from cube.task import TaskMetadata
from PIL import Image

from waa_cube.task import WAATaskExecutionInfo

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

PATCH_GUEST_AGENT = "cube_computer_tool.computer.GuestAgent"
PATCH_EVALUATOR = "waa_cube.task.Evaluator"
PATCH_SETUP_CTRL = "waa_cube.task.SetupController"
PATCH_SLEEP = "waa_cube.task.time.sleep"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_screenshot_bytes(w: int = 100, h: int = 100) -> bytes:
    img = Image.new("RGB", (w, h), color=(64, 64, 64))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_mock_guest(screenshot: bytes | None = None, axtree: str = "<root/>") -> MagicMock:
    guest = MagicMock()
    guest.get_screenshot.return_value = screenshot or _make_screenshot_bytes()
    guest.get_accessibility_tree.return_value = axtree
    guest.get_terminal_output.return_value = ""
    guest.execute_action.return_value = None
    guest.execute_python_command.return_value = {"returncode": 0, "output": ""}
    return guest


def _make_mock_handle(server_port: int = 15000) -> MagicMock:
    handle = MagicMock(spec=ResourceHandle)
    handle.endpoint = f"http://localhost:{server_port}"
    handle.run_id = "test-run-id-1234"
    return handle


def _make_task_metadata(
    task_id: str = "t1",
    instruction: str = "Do something",
) -> TaskMetadata:
    return TaskMetadata(id=task_id, abstract_description=instruction)


def _make_exec_info(
    domain: str = "vscode",
    snapshot: str = "vscode",
    evaluator: dict | None = None,
    related_apps: list[str] | None = None,
) -> WAATaskExecutionInfo:
    return WAATaskExecutionInfo(
        domain=domain,
        snapshot=snapshot,
        config=[],
        evaluator=evaluator if evaluator is not None else {"func": "check_json_settings", "expected": {}},
        related_apps=related_apps if related_apps is not None else ["vscode"],
    )


def _make_mock_infra(ready: bool = True) -> MagicMock:
    from cube.resource import InfraConfig

    infra = MagicMock(spec=InfraConfig)
    infra.provision_status.return_value = "ready" if ready else "needs_provisioning"
    return infra


# ---------------------------------------------------------------------------
# ComputerConfig
# ---------------------------------------------------------------------------


class TestComputerConfig:
    def test_defaults(self) -> None:
        from waa_cube.computer import ComputerConfig

        cfg = ComputerConfig()
        assert cfg.require_a11y_tree is True

    def test_cache_dir_is_waa(self) -> None:
        from waa_cube.computer import ComputerConfig

        cfg = ComputerConfig()
        assert "waa" in cfg.cache_dir.lower()

    def test_make_without_vm_succeeds(self) -> None:
        from waa_cube.computer import ComputerConfig

        computer = ComputerConfig().make()
        assert computer._vm is None
        assert computer._guest is None


# ---------------------------------------------------------------------------
# WAATask
# ---------------------------------------------------------------------------


class TestWAATask:
    def test_model_post_init_creates_tool_without_vm(self) -> None:
        from waa_cube.computer import ComputerConfig
        from waa_cube.task import WAATask

        task = WAATask(
            metadata=_make_task_metadata(),
            execution_info=_make_exec_info(),
            tool_config=ComputerConfig(),
        )
        assert task._computer is not None
        assert task._computer._vm is None
        assert task._resource_handle is None

    def test_os_type_is_always_windows(self) -> None:
        from waa_cube.computer import ComputerConfig
        from waa_cube.task import WAATask

        task = WAATask(
            metadata=_make_task_metadata(),
            execution_info=_make_exec_info(),
            tool_config=ComputerConfig(),
        )
        assert task._os_type() == "windows"

    def test_reset_with_infra_launches_vm(self) -> None:
        from cube.resource import InfraConfig

        from waa_cube.computer import ComputerConfig
        from waa_cube.task import WAATask

        mock_handle = _make_mock_handle()
        mock_infra = MagicMock(spec=InfraConfig)
        mock_infra.launch.return_value = mock_handle

        with (
            patch(PATCH_GUEST_AGENT) as mock_ga_cls,
            patch(PATCH_SETUP_CTRL),
            patch(PATCH_SLEEP),
        ):
            mock_ga_cls.return_value = _make_mock_guest()
            task = WAATask(
                metadata=_make_task_metadata(),
                execution_info=_make_exec_info(),
                tool_config=ComputerConfig(),
                infra=mock_infra,
            )
            obs, info = task.reset()

        mock_infra.launch.assert_called_once()
        assert task._resource_handle is mock_handle
        assert isinstance(obs, Observation)

    def test_reset_returns_goal_in_obs(self) -> None:
        from cube.resource import InfraConfig

        from waa_cube.computer import ComputerConfig
        from waa_cube.task import WAATask

        mock_handle = _make_mock_handle()
        mock_infra = MagicMock(spec=InfraConfig)
        mock_infra.launch.return_value = mock_handle

        with (
            patch(PATCH_GUEST_AGENT) as mock_ga_cls,
            patch(PATCH_SETUP_CTRL),
            patch(PATCH_SLEEP),
        ):
            mock_ga_cls.return_value = _make_mock_guest()
            task = WAATask(
                metadata=_make_task_metadata(),
                execution_info=_make_exec_info(snapshot="vscode"),
                tool_config=ComputerConfig(),
                infra=mock_infra,
            )
            obs, info = task.reset()

        texts = [c.data for c in obs.contents if isinstance(c, TextContent)]
        assert any("Do something" in t for t in texts)
        assert info["task_id"] == "t1"
        assert info["task_snapshot"] == "vscode"

    def test_evaluate_calls_evaluator(self) -> None:
        from waa_cube.computer import ComputerConfig
        from waa_cube.task import WAATask

        task = WAATask(
            metadata=_make_task_metadata(),
            execution_info=_make_exec_info(),
            tool_config=ComputerConfig(),
        )
        task._computer._guest = _make_mock_guest()

        with patch(PATCH_EVALUATOR) as mock_eval_cls:
            mock_eval_cls.return_value.evaluate.return_value = 1.0
            reward, info = task.evaluate(Observation.from_text("state"))

        assert reward == 1.0
        assert "evaluator" in info

    def test_evaluate_no_evaluator_returns_zero(self) -> None:
        from waa_cube.computer import ComputerConfig
        from waa_cube.task import WAATask

        task = WAATask(
            metadata=_make_task_metadata(task_id="no-eval"),
            execution_info=WAATaskExecutionInfo(),  # default — empty evaluator dict
            tool_config=ComputerConfig(),
        )
        reward, info = task.evaluate(Observation())
        assert reward == 0.0
        assert info.get("error") == "no_evaluator"

    def test_finished_reflects_computer_is_done(self) -> None:
        from waa_cube.computer import ComputerConfig
        from waa_cube.task import WAATask

        task = WAATask(
            metadata=_make_task_metadata(),
            execution_info=_make_exec_info(),
            tool_config=ComputerConfig(),
        )
        assert task.finished(Observation()) is False
        task._computer._is_done = True
        assert task.finished(Observation()) is True

    def test_close_closes_resource_handle(self) -> None:
        from waa_cube.computer import ComputerConfig
        from waa_cube.task import WAATask

        task = WAATask(
            metadata=_make_task_metadata(),
            execution_info=_make_exec_info(),
            tool_config=ComputerConfig(),
        )
        mock_handle = _make_mock_handle()
        task._resource_handle = mock_handle
        task._computer._guest = _make_mock_guest()

        task.close()
        mock_handle.close.assert_called_once()
        assert task._resource_handle is None

    def test_step_done_triggers_evaluate(self) -> None:
        from cube.resource import InfraConfig

        from waa_cube.computer import ComputerConfig
        from waa_cube.task import WAATask

        mock_handle = _make_mock_handle()
        mock_infra = MagicMock(spec=InfraConfig)
        mock_infra.launch.return_value = mock_handle

        task = WAATask(
            metadata=_make_task_metadata(),
            execution_info=_make_exec_info(),
            tool_config=ComputerConfig(observe_after_action=False),
            infra=mock_infra,
        )

        with (
            patch(PATCH_GUEST_AGENT) as mock_ga_cls,
            patch(PATCH_SETUP_CTRL),
            patch(PATCH_SLEEP),
            patch(PATCH_EVALUATOR) as mock_eval_cls,
        ):
            mock_ga_cls.return_value = _make_mock_guest()
            mock_eval_cls.return_value.evaluate.return_value = 1.0
            task.reset()
            env_out = task.step(Action(name="done", arguments={}))

        assert env_out.done is True
        assert env_out.reward == 1.0

    def test_infra_required_for_reset(self) -> None:
        from waa_cube.computer import ComputerConfig
        from waa_cube.task import WAATask

        task = WAATask(
            metadata=_make_task_metadata(),
            execution_info=_make_exec_info(snapshot="vscode"),
            tool_config=ComputerConfig(),
        )
        with patch(PATCH_GUEST_AGENT):
            try:
                task.reset()
                assert False, "Expected RuntimeError"
            except RuntimeError as exc:
                assert "requires an InfraConfig" in str(exc)


# ---------------------------------------------------------------------------
# WAABenchmark (config) + WAABenchmarkRuntime
# ---------------------------------------------------------------------------


class TestWAABenchmark:
    def test_benchmark_metadata(self) -> None:
        from waa_cube.benchmark import WAABenchmark

        assert WAABenchmark.benchmark_metadata.name == "waa"
        assert WAABenchmark.benchmark_metadata.num_tasks == 154
        assert "windows" in WAABenchmark.benchmark_metadata.tags
        assert WAABenchmark.task_config_class.__name__ == "WAATaskConfig"
        assert WAABenchmark.benchmark_class.__name__ == "WAABenchmarkRuntime"

    def test_task_metadata_loaded_at_import(self) -> None:
        """task_metadata.json is auto-loaded by __init_subclass__."""
        from waa_cube.benchmark import WAABenchmark

        assert len(WAABenchmark.task_metadata) > 100

    def test_task_metadata_has_required_fields(self) -> None:
        """Slim TaskMetadata: id + abstract_description only."""
        from waa_cube.benchmark import WAABenchmark

        for tid, meta in list(WAABenchmark.task_metadata.items())[:5]:
            assert meta.id == tid
            assert isinstance(meta.abstract_description, str)

    def test_make_returns_runtime_without_infra(self) -> None:
        from waa_cube.benchmark import WAABenchmark, WAABenchmarkRuntime
        from waa_cube.computer import ComputerConfig

        bench_config = WAABenchmark(tool_config=ComputerConfig(), infra=_make_mock_infra())
        runtime = bench_config.make()
        assert isinstance(runtime, WAABenchmarkRuntime)
        assert len(bench_config.task_metadata) > 100

    def test_subset_from_list_filters(self) -> None:
        from waa_cube.benchmark import WAABenchmark
        from waa_cube.computer import ComputerConfig

        bench = WAABenchmark(tool_config=ComputerConfig(), infra=_make_mock_infra())

        # Pick the first 10 task ids — domain lives in execution_info, not metadata,
        # so we filter by id alone.
        keep_ids = list(bench.task_metadata.keys())[:10]
        filtered = bench.subset_from_list(keep_ids)

        assert len(filtered.tasks()) == 10
        assert set(filtered.tasks().keys()) == set(keep_ids)

    def test_get_task_configs_yields_waa_task_config(self) -> None:
        from waa_cube.benchmark import WAABenchmark, WAATaskConfig
        from waa_cube.computer import ComputerConfig

        bench = WAABenchmark(tool_config=ComputerConfig(), infra=_make_mock_infra())

        configs = list(bench.get_task_configs())
        assert len(configs) == len(bench.task_metadata)
        for cfg in configs:
            assert isinstance(cfg, WAATaskConfig)
            assert cfg.task_id in bench.task_metadata

    def test_task_config_make_produces_waa_task(self) -> None:
        from waa_cube.benchmark import WAABenchmark
        from waa_cube.computer import ComputerConfig
        from waa_cube.task import WAATask

        bench_config = WAABenchmark(tool_config=ComputerConfig(), infra=_make_mock_infra())
        # make() also writes per-task execution-info cache files used by WAATaskConfig.make()
        bench_config.make()

        cfg = next(bench_config.get_task_configs())
        task = cfg.make()

        assert isinstance(task, WAATask)
        assert task.metadata.id == cfg.task_id

    def test_make_calls_provision_when_not_ready(self) -> None:
        from cube.resource import InfraConfig

        from waa_cube.benchmark import WAABenchmark
        from waa_cube.computer import ComputerConfig

        mock_infra = MagicMock(spec=InfraConfig)
        mock_infra.provision_status.return_value = "needs_provisioning"
        bench = WAABenchmark(tool_config=ComputerConfig(), infra=mock_infra)
        bench.make(infra=mock_infra)
        assert mock_infra.provision.call_count == len(bench.resources)

    def test_setup_invokes_cleanup_stale(self) -> None:
        """Runtime setup must sweep stale resources from previous runs before
        launching new VMs. Skipping this lets crashed-task orphans accumulate and
        bill until manual cleanup."""
        from waa_cube.benchmark import WAABenchmark
        from waa_cube.computer import ComputerConfig

        mock_infra = _make_mock_infra()
        mock_infra.cleanup_stale.return_value = []
        bench_config = WAABenchmark(tool_config=ComputerConfig(), infra=mock_infra)
        runtime = bench_config.make()
        try:
            mock_infra.cleanup_stale.assert_called_once()
        finally:
            runtime.close()

    def test_debug_tasks_overlay(self) -> None:
        """tasks_file overlays debug tasks onto shipped metadata."""
        import waa_cube
        from waa_cube.benchmark import WAABenchmark
        from waa_cube.computer import ComputerConfig

        debug_file = str(Path(waa_cube.__file__).parent / "debug_tasks.json")
        bench = WAABenchmark(
            tool_config=ComputerConfig(),
            tasks_file=debug_file,
            infra=_make_mock_infra(),
        )
        # Loading the debug overlay populates the per-task execution-info cache;
        # it does not mutate the class-level task_metadata registry. Verify the
        # overlay loader at least returns expected task ids.
        loaded = bench._load_task_metadata_from_file(debug_file)
        assert any("debug" in tid for tid in loaded)
        assert len(WAABenchmark.task_metadata) > 100

    def test_runtime_close_does_not_raise(self) -> None:
        from waa_cube.benchmark import WAABenchmark
        from waa_cube.computer import ComputerConfig

        runtime = WAABenchmark(tool_config=ComputerConfig(), infra=_make_mock_infra()).make()
        runtime.close()


# ---------------------------------------------------------------------------
# Azure integration
# ---------------------------------------------------------------------------


def test_waa_windows_resource_has_source_url() -> None:
    from waa_cube.azure import WAA_WINDOWS_RESOURCE

    assert WAA_WINDOWS_RESOURCE.name == "waa-windows-vm"
    assert WAA_WINDOWS_RESOURCE.source_url is not None
    assert "huggingface" in WAA_WINDOWS_RESOURCE.source_url


def test_waabenchmark_accepts_infra_field() -> None:
    from cube import LocalInfraConfig

    from waa_cube.benchmark import WAABenchmark
    from waa_cube.computer import ComputerConfig

    bench = WAABenchmark(tool_config=ComputerConfig())
    assert isinstance(bench.infra, LocalInfraConfig)


def test_waatask_accepts_infra_field() -> None:
    from waa_cube.computer import ComputerConfig
    from waa_cube.task import WAATask

    task = WAATask(
        metadata=_make_task_metadata(task_id="test-task", instruction="test"),
        execution_info=_make_exec_info(domain="notepad", snapshot="init_state"),
        tool_config=ComputerConfig(),
        infra=None,
    )
    assert task.infra is None


def test_infra_injected_into_task_configs() -> None:
    from cube.resource import InfraConfig

    from waa_cube.benchmark import WAABenchmark
    from waa_cube.computer import ComputerConfig

    mock_infra = MagicMock(spec=InfraConfig)
    mock_infra.provision_status.return_value = "ready"
    bench = WAABenchmark(tool_config=ComputerConfig(), infra=mock_infra)

    for cfg in bench.get_task_configs():
        assert cfg.infra is mock_infra

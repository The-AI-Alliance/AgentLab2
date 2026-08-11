"""Smoke tests for browsercomp_cube."""

from __future__ import annotations

from typing import Any

import pytest

from browsercomp_cube import BROWSECOMP_CONFIGS, BrowseCompBenchmarkConfig
from browsercomp_cube.crypto import decrypt, derive_key, encrypt
from browsercomp_cube.debug import DebugBrowseCompBenchmark, DebugBrowseCompBenchmarkConfig, get_debug_benchmark
from browsercomp_cube.task import BrowseCompExecutionInfo, BrowseCompTask, BrowseCompTaskConfig, BrowseCompTaskMetadata
from browsercomp_cube.tool import SubmitAnswerToolConfig
from cube.tool import ToolboxConfig


def test_browsecomp_benchmark_config_constructs() -> None:
    bench = BrowseCompBenchmarkConfig(scorer_model="gpt-5.4-mini")
    assert bench.name == "browsercomp-cube"
    assert bench.benchmark_metadata.num_tasks == 1266
    assert len(bench.task_metadata) == 1266
    first_id = next(iter(bench.task_metadata))
    assert first_id.startswith("browsecomp-")
    assert bench.scorer_model == "gpt-5.4-mini"


def test_browsecomp_configs_has_default() -> None:
    # Every cube exposes <NAME>_CONFIGS["default"] (recipe-template contract).
    assert "default" in list(BROWSECOMP_CONFIGS.keys())
    assert BROWSECOMP_CONFIGS["default"].scorer_model == "openai/gpt-4o"


def test_browsecomp_benchmark_config_round_trip() -> None:
    cfg = BrowseCompBenchmarkConfig(scorer_model="gpt-5.4-mini")
    rehydrated = BrowseCompBenchmarkConfig.model_validate_json(cfg.model_dump_json())
    assert rehydrated == cfg


_EXPECTED_SUBSETS = {
    "art": ("Art", 127),
    "geography": ("Geography", 70),
    "history": ("History", 125),
    "music": ("Music", 116),
    "other": ("Other", 197),
    "politics": ("Politics", 59),
    "science-and-technology": ("Science & technology", 173),
    "sports": ("Sports", 123),
    "tv-shows-and-movies": ("TV shows & movies", 205),
    "video-games": ("Video games", 71),
}


def test_named_subsets_partition_dataset() -> None:
    cfg = BrowseCompBenchmarkConfig(scorer_model="unused")
    assert set(BrowseCompBenchmarkConfig.named_subsets()) == set(_EXPECTED_SUBSETS)
    total = sum(len(cfg.named_subset(name).task_ids) for name in _EXPECTED_SUBSETS)
    assert total == cfg.benchmark_metadata.num_tasks == len(cfg.tasks())


@pytest.mark.parametrize(("name", "expected"), list(_EXPECTED_SUBSETS.items()))
def test_named_subset_filters_by_topic(name: str, expected: tuple[str, int]) -> None:
    expected_topic, expected_count = expected
    cfg = BrowseCompBenchmarkConfig(scorer_model="unused")
    sub = cfg.named_subset(name)
    tasks = list(sub.tasks().values())
    assert len(tasks) == expected_count
    assert {tm.topic for tm in tasks} == {expected_topic}
    rehydrated = BrowseCompBenchmarkConfig.model_validate_json(sub.model_dump_json())
    assert rehydrated.task_ids == sub.task_ids


def test_browsecomp_task_config_round_trip() -> None:
    metadata = BrowseCompTaskMetadata(id="browsecomp-test", topic="debug")
    cfg = BrowseCompTaskConfig(metadata=metadata, scorer_model="gpt-5.4-mini")
    rehydrated = BrowseCompTaskConfig.model_validate_json(cfg.model_dump_json())
    assert rehydrated == cfg
    assert isinstance(rehydrated.metadata, BrowseCompTaskMetadata)
    assert rehydrated.metadata.topic == "debug"


def test_debug_benchmark_constructs() -> None:
    cfg = get_debug_benchmark()
    assert isinstance(cfg, DebugBrowseCompBenchmarkConfig)
    assert len(cfg.task_metadata) == 2
    configs = list(cfg.get_task_configs())
    assert {c.task_id for c in configs} == {"browsecomp-debug-0000", "browsecomp-debug-0001"}
    bench = cfg.make()
    assert isinstance(bench, DebugBrowseCompBenchmark)
    bench.close()


def test_debug_benchmark_config_round_trip() -> None:
    cfg = DebugBrowseCompBenchmarkConfig()
    rehydrated = DebugBrowseCompBenchmarkConfig.model_validate_json(cfg.model_dump_json())
    assert rehydrated == cfg


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


def _make_task() -> BrowseCompTask:
    """Build a minimal BrowseCompTask without going through Benchmark.install()."""
    return BrowseCompTask(
        metadata=BrowseCompTaskMetadata(id="browsecomp-test"),
        tool_config=ToolboxConfig(tool_configs=[SubmitAnswerToolConfig()]),
        execution_info=BrowseCompExecutionInfo(problem="ignored", answer="ignored"),
        scorer_model="any-model",
    )


@pytest.fixture
def task() -> BrowseCompTask:
    return _make_task()


@pytest.mark.parametrize(
    ("grader_response", "expected"),
    [
        ("reasoning: looks fine\ncorrect: yes\nconfidence: 95", True),
        ("reasoning: mismatch\ncorrect: no\nconfidence: 10", False),
        ("CORRECT: YES", True),
        ("Correct:   No", False),
        ("correct:yes", True),
    ],
)
def test_grader_regex_parses_yes_no(
    task: BrowseCompTask,
    monkeypatch: pytest.MonkeyPatch,
    grader_response: str,
    expected: bool,
) -> None:
    def fake_completion(**_: Any) -> _FakeCompletion:
        return _FakeCompletion(grader_response)

    monkeypatch.setattr("browsercomp_cube.task.litellm.completion", fake_completion)
    is_correct, raw = task._call_grader("any prompt", "any-model")
    assert is_correct is expected
    assert raw == grader_response


def test_grader_regex_raises_when_no_match(
    task: BrowseCompTask,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_completion(**_: Any) -> _FakeCompletion:
        return _FakeCompletion("reasoning: ok\nconfidence: 99\n(no verdict line)")

    monkeypatch.setattr("browsercomp_cube.task.litellm.completion", fake_completion)
    with pytest.raises(ValueError, match="missing 'correct: yes/no'"):
        task._call_grader("any prompt", "any-model")


@pytest.mark.parametrize(
    ("plaintext", "password"),
    [
        ("hello world", "canary-123"),
        ("", "any-password"),
        ("Unicode: café — 日本語 🌍", "secret"),
        ("a" * 257, "k"),  # spans multiple SHA-256 blocks
    ],
)
def test_crypto_round_trip(plaintext: str, password: str) -> None:
    assert decrypt(encrypt(plaintext, password), password) == plaintext


def test_derive_key_length_and_determinism() -> None:
    k1 = derive_key("pw", 100)
    k2 = derive_key("pw", 100)
    assert len(k1) == 100
    assert k1 == k2
    assert derive_key("pw", 0) == b""
    assert derive_key("pw", 32) == derive_key("pw", 100)[:32]


def test_decrypt_with_wrong_password_does_not_match() -> None:
    ciphertext = encrypt("the secret", "right")
    with pytest.raises(UnicodeDecodeError):
        # Random XOR result is overwhelmingly invalid UTF-8 for this input.
        decrypt(ciphertext, "wrong-password-different-length")

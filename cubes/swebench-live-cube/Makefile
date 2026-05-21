.PHONY: test debug install

test:
	uv run cube test swebench-live-cube

debug:
	uv run python -m swebench_live_cube.debug

install:
	uv sync --all-extras
	uv pip install -e .
	uv run python scripts/create_task_metadata.py

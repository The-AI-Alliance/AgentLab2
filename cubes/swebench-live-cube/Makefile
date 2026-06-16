.PHONY: test debug install

test:
	uv run --no-sync cube test swebench-live-cube

debug:
	uv run --no-sync python -m swebench_live_cube.debug

install:
	uv sync --all-extras
	uv pip install -e .
	uv run --no-sync python scripts/create_task_metadata.py

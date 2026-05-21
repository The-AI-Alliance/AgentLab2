.PHONY: test debug install

test:
	uv run cube test swebench-verified-cube

debug:
	uv run python -m swebench_verified_cube.debug

install:
	uv sync --all-extras
	uv run python scripts/create_task_metadata.py

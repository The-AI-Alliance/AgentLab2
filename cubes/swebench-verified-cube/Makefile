.PHONY: test debug install

test:
	uv run --no-sync cube test swebench-verified-cube

debug:
	uv run --no-sync python -m swebench_verified_cube.debug

install:
	uv sync --all-extras
	uv pip install -e .
	uv run --no-sync python scripts/create_task_metadata.py

.PHONY: test debug install

test:
	uv run cube test swegym-lite-cube

debug:
	uv run python -m swegym_lite_cube.debug

install:
	uv sync --all-extras
	uv pip install -e .
	uv run python scripts/create_task_metadata.py

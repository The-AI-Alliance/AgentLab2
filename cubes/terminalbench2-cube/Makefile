.PHONY: test debug install

test:
	uv run cube test terminalbench2-cube

debug:
	uv run python -m terminalbench2_cube.debug

install:
	uv sync --all-extras
	uv pip install -e .
	uv run scripts/create_task_metadata.py

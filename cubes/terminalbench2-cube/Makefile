.PHONY: test debug install

test:
	uv run --no-sync cube test terminalbench2-cube

debug:
	uv run --no-sync python -m terminalbench2_cube.debug

install:
	uv sync --all-extras
	uv pip install -e .
	uv run --no-sync scripts/create_task_metadata.py

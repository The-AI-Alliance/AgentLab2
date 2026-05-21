.PHONY: test install debug

test:
	uv run --no-sync cube test workarena-cube

debug:
	uv run --no-sync python -m workarena_cube.debug

install:
	uv sync --all-extras
	uv pip install -e .
	uv run --no-sync scripts/generate_task_metadata.py
	uv run --no-sync playwright install chromium

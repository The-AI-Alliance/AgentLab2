.PHONY: test install debug

test:
	uv run cube test workarena-cube

debug:
	uv run python -m workarena_cube.debug

install:
	uv sync --all-extras
	uv pip install -e .
	uv run scripts/generate_task_metadata.py
	uv run playwright install chromium

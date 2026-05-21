.PHONY: test debug install

test:
	uv run cube test webarena-verified-cube

debug:
	uv run python -m webarena_verified_cube.debug

install:
	uv sync --all-extras
	uv run scripts/generate_task_metadata.py
	uv run playwright install chromium

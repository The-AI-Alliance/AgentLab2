.PHONY: test debug install

test:
	uv run --no-sync cube test webarena-verified-cube

debug:
	uv run --no-sync python -m webarena_verified_cube.debug

install:
	uv sync --all-extras
	uv pip install -e .
	uv run --no-sync scripts/generate_task_metadata.py
	uv run --no-sync playwright install chromium

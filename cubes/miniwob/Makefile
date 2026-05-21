.PHONY: test install debug

test:
	uv run --no-sync cube test miniwob-cube

debug:
	uv run --no-sync python -m miniwob_cube.debug

install:
	uv sync --all-extras
	uv pip install -e .
	uv run --no-sync playwright install chromium

.PHONY: test install debug

test:
	uv run cube test miniwob-cube

debug:
	uv run python -m miniwob_cube.debug

install:
	uv sync --all-extras
	uv pip install -e .
	uv run playwright install chromium

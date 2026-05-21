.PHONY: test install debug

test:
	uv run --no-sync cube test browsercomp-cube

debug:
	uv run --no-sync python -m browsercomp_cube.debug

install:
	uv sync --all-extras
	uv pip install -e .

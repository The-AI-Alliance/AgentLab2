.PHONY: test install debug

test:
	uv run cube test browsercomp-cube

debug:
	uv run python -m browsercomp_cube.debug

install:
	uv sync --all-extras
	uv pip install -e .

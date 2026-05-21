.PHONY: install debug test


install:
	uv sync --all-extras

debug:
	uv run python -m arithmetic_cube.debug

test:
	uv run cube test arithmetic-cube

.PHONY: install debug test


install:
	uv sync --all-extras
	uv pip install -e .

debug:
	uv run python -m arithmetic_cube.debug

test:
	uv run cube test arithmetic-cube

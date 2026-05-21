.PHONY: install debug test


install:
	uv sync --all-extras
	uv pip install -e .

debug:
	uv run --no-sync python -m arithmetic_cube.debug

test:
	uv run --no-sync cube test arithmetic-cube

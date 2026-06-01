.PHONY: test install

test:
	uv run python -m math_tool_use.debug

install:
	uv sync --all-extras
	pip install -e .

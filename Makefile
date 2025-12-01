.PHONY: help install format lint

help:
	@echo "make install    - Install dependencies in editable mode"
	@echo "make format     - Format code"
	@echo "make lint       - Lint and auto-fix"

install:
	uv sync
	uv pip install -e .

format:
	uv run ruff format .

lint:
	uv run ruff check --fix .

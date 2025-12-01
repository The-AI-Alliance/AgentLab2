.PHONY: help install format lint

help:
	@echo "make install    - Install dependencies"
	@echo "make format     - Format code"
	@echo "make lint       - Lint and auto-fix"

install:
	uv sync

format:
	uv run ruff format .

lint:
	uv run ruff check --fix .

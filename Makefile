.PHONY: help install ci-install update format lint lint-check test coverage hello debug xray report review

help:
	@echo "make install       - Install dependencies in editable mode"
	@echo "make ci-install    - Install dependencies with locked versions (for CI)"
	@echo "make update        - Update dependencies"
	@echo "make format        - Format code"
	@echo "make lint          - Lint and auto-fix (modifies files)"
	@echo "make lint-check    - Check lint without fixing (read-only, what CI runs)"
	@echo "make test          - Run unit tests"
	@echo "make coverage      - Run tests with coverage report"
	@echo "make hello         - Run hello_miniwob recipe"
	@echo "make debug         - Run hello_miniwob recipe in debug mode"
	@echo "make xray          - Run AL2 XRay viewer in debug mode"
	@echo "make report        - Print a markdown table of experiments in ~/cube_harness_results/"
	@echo "make review PR=<n> - Check out a PR and set up any cross-repo cube-standard dependency"

hello:
	@echo "🤖 Running hello_miniwob recipe"
	uv pip install -e "cubes/miniwob[debug]"
	uv run recipes/hello_miniwob.py

debug:
	@echo "🤖 Running hello_miniwob recipe in debug mode"
	uv pip install -e "cubes/miniwob[debug]"
	uv run recipes/hello_miniwob.py --limit 2

xray:
	uv run ch-xray --debug

report:
	uv run --active scripts/experiments_report.py $(ARGS)

install:
	@echo "🚀 Installing dependencies"
	@echo "Install requires sudo permissions to install Playwright dependencies. You may be prompted for your password."
	uv sync --all-extras
	uv run playwright install chromium --with-deps
	pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type prepare-commit-msg

ci-install:
	uv sync --frozen --all-extras
	uv run playwright install chromium --with-deps

update:
	@echo "🔄 Updating dependencies"
	uv sync --all-extras --upgrade
	uv run playwright install chromium --with-deps

lint:
	@echo "🧹 Linting code"
	uvx ruff check --fix .
	uvx ruff format .

lint-check:
	@echo "🧹 Checking lint"
	uvx ruff check --diff .
	uvx ruff format --diff .

test: install
	@echo "🧪 Running unit tests"
	uv run pytest -n 10 tests/ -v

coverage:
	@echo "📊 Running tests with coverage report"
	uv run pytest tests/ --cov=cube_harness --cov-report=term-missing

review:
	@if [ -z "$(PR)" ]; then echo "Usage: make review PR=<number>"; exit 1; fi
	@if ! git diff --quiet || ! git diff --cached --quiet; then \
	    echo "❌ Working tree has uncommitted changes. Stash or commit before running make review."; \
	    exit 1; \
	fi
	@echo "🔍 Checking out PR $(PR)"
	gh pr checkout $(PR) --repo The-AI-Alliance/cube-harness
	@DEPENDS_ON=$$(gh pr view $(PR) --repo The-AI-Alliance/cube-harness --json body --jq '.body' \
	    | grep -i '^Depends-on:' | head -1 | sed 's/Depends-on://I' | tr -d ' \r'); \
	if [ -n "$$DEPENDS_ON" ]; then \
	    echo "📦 Depends-on: $$DEPENDS_ON"; \
	    BRANCH=$$(echo "$$DEPENDS_ON" | cut -d/ -f2-); \
	    if [ -d "cube-standard" ]; then \
	        CURRENT=$$(git -C cube-standard rev-parse --abbrev-ref HEAD); \
	        if [ "$$CURRENT" != "$$BRANCH" ]; then \
	            echo "❌ cube-standard/ already exists but is on branch '$$CURRENT' (expected '$$BRANCH')."; \
	            echo "   Switch it manually: git -C cube-standard checkout $$BRANCH"; \
	            echo "   Or remove cube-standard/ and re-run make review PR=$(PR)."; \
	            exit 1; \
	        fi; \
	        echo "✅ cube-standard/ already on branch $$BRANCH"; \
	    else \
	        echo "Cloning cube-standard branch: $$BRANCH"; \
	        git clone --branch "$$BRANCH" https://github.com/The-AI-Alliance/cube-standard.git cube-standard; \
	    fi; \
	else \
	    echo "No Depends-on found — using cube-standard from PyPI"; \
	fi
	@if [ -d "cube-standard/.git" ]; then \
	    echo "📦 Installing cube-standard (all workspace packages)..."; \
	    uv pip install -e cube-standard --all-packages --all-extras; \
	fi
	@echo "✅ Ready to review PR $(PR)"

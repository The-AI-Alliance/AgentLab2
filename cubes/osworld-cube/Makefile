.PHONY: install test unit-test debug-docker

install:
	bash scripts/install_qemu.sh
	uv sync --all-extras
	uv pip install -e .
	uv run python scripts/create_task_metadata.py
	uv run playwright install chromium

test:
	uv run cube test osworld-cube

unit-test:
	uv run pytest -n auto tests/

debug-docker:
	uv run cube test osworld_cube.debug_docker

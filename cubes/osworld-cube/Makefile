.PHONY: install test unit-test debug-docker

install:
	bash scripts/install_qemu.sh
	uv sync --all-extras
	uv pip install -e .
	uv run --no-sync python scripts/create_task_metadata.py
	uv run --no-sync playwright install chromium

test:
	uv run --no-sync cube test osworld-cube

unit-test:
	uv run --no-sync pytest -n auto tests/

debug-docker:
	uv run --no-sync cube test osworld_cube.debug_docker

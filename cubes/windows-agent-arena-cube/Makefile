.PHONY: install test unit-test bootstrap-winrm build-image

install:
	bash scripts/install_system_deps.sh
	uv sync --all-extras
	uv pip install -e .
	uv run python scripts/create_task_metadata.py

test:
	uv run cube test waa-cube

unit-test:
	uv run pytest tests/

# One-time WinRM bootstrap on the base qcow2 (uses WAA's Flask guest agent
# on port 5000 — no VNC or manual step). Idempotent. Required before the
# first `make build-image`.
# Requires: WAA_BUILD_ADMIN_PASSWORD env var (Azure-complexity password to set
# on the Docker admin user — same value used later as PKR_VAR_admin_password).
# Refuses to run without a backup in ~/.cube/images/backups/ (pass
# --skip-backup-check to override).
bootstrap-winrm:
	uv run python packer/bootstrap_winrm.py

# Build the prepared WAA Windows image via Packer.
# Requires: packer, qemu-system-x86, ovmf, swtpm, and a base image at
# $HOME/.cube/images/waa-windows-vm.qcow2 that has been through `bootstrap-winrm`.
# Reads PKR_VAR_admin_password from the environment (must be set).
build-image:
	packer/run.sh

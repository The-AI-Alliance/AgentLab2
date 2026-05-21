"""Task-evaluation logic for OSWorld.

VM provisioning is handled exclusively via ``InfraConfig`` +
``cube.resource.VMResourceConfig`` (see ``benchmark.py``); the legacy
QEMU/Docker backends have been removed.

This package now only carries the task-eval submodules:
  - ``evaluator`` — OSWorld result evaluation
  - ``setup_controller`` — per-task environment setup
  - ``getters`` / ``metrics`` — evaluator helper modules
"""

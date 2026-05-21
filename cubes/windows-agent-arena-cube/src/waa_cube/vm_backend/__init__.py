"""Task-evaluation logic for Windows Agent Arena.

VM provisioning is handled exclusively via ``InfraConfig`` (see
``benchmark.py``); the legacy ``cube.vm`` Docker backend has been removed.
This package now only carries the task-eval submodules (``evaluator``,
``setup_controller``, ``getters``, ``metrics``).
"""

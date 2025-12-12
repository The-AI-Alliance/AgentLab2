from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("agentlab2")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["__version__"]

"""MCP server that exposes cube-harness tools via the Model Context Protocol."""

import asyncio
import functools
import inspect
import logging
from typing import Any, Literal

from cube.core import Action, Observation, TypedBaseModel
from cube.tool import AbstractAsyncTool, AbstractTool
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

from cube_harness.mcp.convert import observation_to_mcp_content

logger = logging.getLogger(__name__)


class McpServerConfig(TypedBaseModel):
    """Configuration for the MCP server."""

    server_name: str = "cube_harness"
    transport: Literal["stdio", "sse"] = "stdio"
    host: str = "localhost"
    port: int = 8000


class McpServer:
    """Adapts an cube-harness tool to serve its actions as MCP tools.

    Wraps any AbstractTool | AbstractAsyncTool (PlaywrightTool, BgymTool, Toolbox, etc.)
    and exposes its action_set as MCP tools using FastMCP.

    Usage:
        tool = PlaywrightConfig(headless=True).make()
        server = McpServer(tool=tool)
        server.run()  # blocks, serving MCP over stdio
    """

    def __init__(self, tool: AbstractTool | AbstractAsyncTool, config: McpServerConfig | None = None) -> None:
        self._tool = tool
        self._config = config or McpServerConfig()
        self._mcp = FastMCP(self._config.server_name)
        self._register_tools()

    @property
    def raw(self) -> FastMCP:
        """Escape hatch: access the underlying FastMCP server instance."""
        return self._mcp

    def _register_tools(self) -> None:
        """Register all tool actions as MCP tools on the FastMCP server."""
        for schema in self._tool.action_set:
            method = getattr(self._tool, schema.name)
            handler = _make_async_handler(self._tool, method, schema.name)
            self._mcp.add_tool(handler, name=schema.name, description=schema.description)

    def run(self) -> None:
        """Start the MCP server (blocks until shutdown)."""
        self._mcp.run(transport=self._config.transport)


def _make_async_handler(tool: AbstractTool | AbstractAsyncTool, method: Any, action_name: str) -> Any:
    """Create an async MCP handler that dispatches to tool.execute_action.

    The wrapper preserves the original method's signature via functools.wraps,
    allowing FastMCP to infer the parameter schema from type hints.

    Async tools (e.g. AsyncPlaywrightTool) are awaited directly.
    Sync tools are dispatched via asyncio.to_thread to avoid blocking the loop.
    """
    is_async = inspect.iscoroutinefunction(tool.execute_action)

    @functools.wraps(method)
    async def handler(*args: Any, **kwargs: Any) -> list[TextContent | ImageContent]:
        action = Action(name=action_name, arguments=kwargs)
        if is_async:
            assert isinstance(tool, AbstractAsyncTool), f"Expected async tool, got {type(tool)}"
            obs: Observation = await tool.execute_action(action)
        else:
            assert isinstance(tool, AbstractTool), f"Expected sync tool, got {type(tool)}"
            obs: Observation = await asyncio.to_thread(tool.execute_action, action)
        return observation_to_mcp_content(obs)

    # Override the return annotation so FastMCP treats the output as MCP content
    # (functools.wraps copies the original method's -> str annotation)
    handler.__annotations__["return"] = list[TextContent | ImageContent]

    return handler


def main() -> None:
    """CLI entry point for ch-mcp-server."""
    import argparse

    parser = argparse.ArgumentParser(description="Start an cube-harness MCP server")
    parser.add_argument(
        "--tool-config",
        type=str,
        required=True,
        help="Python import path to a ToolConfig instance, e.g. 'cube_browser_tool.PlaywrightConfig'",
    )
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--server-name", default="cube_harness")
    args = parser.parse_args()

    # Import and instantiate the ToolConfig
    module_path, class_name = args.tool_config.rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    config_cls = getattr(module, class_name)
    tool_config = config_cls()
    tool = tool_config.make()

    server_config = McpServerConfig(
        server_name=args.server_name,
        transport=args.transport,
        host=args.host,
        port=args.port,
    )
    server = McpServer(tool=tool, config=server_config)
    logger.info("Starting MCP server '%s' with transport=%s", server_config.server_name, server_config.transport)
    server.run()


if __name__ == "__main__":
    main()

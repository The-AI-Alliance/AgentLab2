"""Recipe: Expose the Playwright browser tool as an MCP server and verify tool calling.

Demonstrates the full MCP server roundtrip with a real browser tool:
1. Start a tiny HTTP server serving a test page with a clickable button
2. Create an AsyncPlaywrightTool (headless Chromium) via AsyncPlaywrightConfig.make()
3. Wrap it with McpServer, which registers AsyncBrowserActionSpace as MCP tools
4. Navigate to the test page (task-internal goto), then exercise MCP tools:
   - noop: simplest roundtrip, verifies page HTML is returned
   - browser_click: click the button and verify the page updates

Prerequisites:
    playwright install chromium

Usage:
    uv run recipes/tool_api.py
"""

import asyncio
import logging

from cube_browser_tool import AsyncPlaywrightConfig
from mcp.types import TextContent as MCPTextContent

from cube_harness.mcp.server import McpServer

LOG_FORMAT = "[%(levelname)s] %(asctime)s - %(name)s:%(lineno)d - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

TEST_PAGE = (
    "<html><body>"
    "<p>hi there!</p>"
    "<button id='btn' onclick=\"this.textContent='clicked'\">click me</button>"
    "</body></html>"
)
TEST_PORT = 8791


async def start_test_server() -> asyncio.Server:
    """Start a minimal HTTP server that serves a single page."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.read(4096)
        response = f"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n{TEST_PAGE}"
        writer.write(response.encode())
        await writer.drain()
        writer.close()

    return await asyncio.start_server(handle, "127.0.0.1", TEST_PORT)


EXPECTED_BROWSER_TOOLS = {
    "noop",
    "browser_wait",
    "browser_click",
    "browser_type",
    "browser_press_key",
    "browser_hover",
    "browser_back",
    "browser_forward",
    "browser_drag",
    "browser_select_option",
    "browser_mouse_click_xy",
    "browser_scroll",
}


async def main() -> None:
    # -- 0. Start test HTTP server --
    http_server = await start_test_server()
    logger.info("Test HTTP server started on port %d.", TEST_PORT)

    # -- 1. Create AsyncPlaywrightTool --
    tool = await AsyncPlaywrightConfig(use_screenshot=False).make()
    logger.info("AsyncPlaywrightTool initialized (headless Chromium).")

    try:
        # -- 2. Wrap with McpServer --
        server = McpServer(tool=tool)
        mcp = server.raw
        logger.info("McpServer created, tools registered on FastMCP.")

        # -- 3. List registered MCP tools --
        tools = await mcp.list_tools()
        logger.info("Discovered %d MCP tools:", len(tools))
        for t in tools:
            logger.info("  - %s: %s", t.name, t.description[:80] if t.description else "")

        tool_names = {t.name for t in tools}
        missing = EXPECTED_BROWSER_TOOLS - tool_names
        assert not missing, f"Missing expected browser tools: {missing}"
        logger.info("All %d BrowserActionSpace tools registered.", len(EXPECTED_BROWSER_TOOLS))

        # -- 4. Navigate to test page (task-internal, not an MCP tool) --
        await tool.goto(f"http://127.0.0.1:{TEST_PORT}")
        logger.info("Navigated to test page.")

        # -- 5. Test noop: verify page HTML is returned --
        logger.info("Calling noop...")
        blocks, _ = await mcp.call_tool("noop", {})
        blocks = list(blocks)  # convert from generator
        text_blocks = [b for b in blocks if isinstance(b, MCPTextContent)]
        page_html = " ".join(b.text for b in text_blocks)
        logger.info("  noop returned %d block(s), html snippet: %s", len(blocks), page_html[:120])
        assert "hi there!" in page_html, f"Expected 'hi there!' in page HTML, got: {page_html[:200]}"

        # -- 6. Test browser_click: click the button and verify the page updates --
        logger.info("Calling browser_click(selector='#btn')...")
        blocks, _ = await mcp.call_tool("browser_click", {"selector": "#btn"})
        blocks = list(blocks)  # convert from generator
        text_blocks = [b for b in blocks if isinstance(b, MCPTextContent)]
        page_html = " ".join(b.text for b in text_blocks)
        logger.info("  browser_click returned %d block(s), html snippet: %s", len(blocks), page_html[:120])
        assert "click me" not in page_html, f"Expected button text to change from 'click me', got: {page_html[:200]}"
        assert "clicked\n</button>" in page_html, (
            f"Expected button text content to be 'clicked', got: {page_html[:200]}"
        )

        logger.info("All good! Playwright MCP server roundtrip verified.")

    finally:
        await tool.close()
        http_server.close()
        await http_server.wait_closed()
        logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())

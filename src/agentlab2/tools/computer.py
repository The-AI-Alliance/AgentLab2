import logging

from PIL import Image

from agentlab2.environment import Tool

logger = logging.getLogger(__name__)


class Computer(Tool):
    def __init__(self, container_name: str, image_name: str, docker_sock: str = "/var/run/docker.sock"):
        self.container_name = container_name
        self.image_name = image_name
        self.docker_sock = docker_sock

    def mouse_click_xy(self, x: int, y: int, button: str = "left", double: bool = False):
        """Click at the specified (x, y) coordinates on the computer screen."""
        pass  # Implementation goes here

    def mouse_hover_xy(self, x: int, y: int):
        """Hover at the specified (x, y) coordinates on the computer screen."""
        pass  # Implementation goes here

    def mouse_drag_xy(self, start_x: int, start_y: int, end_x: int, end_y: int):
        """Drag the mouse from (start_x, start_y) to (end_x, end_y) on the computer screen."""
        pass  # Implementation goes here

    def keyboard_type(self, text: str):
        """Type the specified text using the computer keyboard."""
        pass  # Implementation goes here

    def run_cli_command(self, command: str) -> str:
        """Run a CLI command on the computer and return its output."""
        raise NotImplementedError

    def get_screenshot(self) -> Image.Image:
        """Capture and return a screenshot of the computer screen."""
        raise NotImplementedError

    def get_current_window_axtree(self) -> dict:
        """Get the accessibility tree of the current active window."""
        raise NotImplementedError

import json
import logging
import re

from bs4 import BeautifulSoup
from cube.core import Action
from litellm import Message
from litellm.types.utils import ChatCompletionMessageToolCall

logger = logging.getLogger(__name__)


def prune_html(html):
    html = re.sub(r"\n", " ", html)
    # remove html comments
    html = re.sub(r"<!--(.*?)-->", "", html, flags=re.MULTILINE)

    soup = BeautifulSoup(html, "html.parser")
    for tag in reversed(soup.find_all()):
        # remove body and html tags (not their content)
        if tag.name in ("html", "body"):
            tag.unwrap()
        # remove useless tags
        elif tag.name in ("style", "link", "script", "br"):
            tag.decompose()
        # remove / unwrap structural tags
        elif tag.name in ("div", "span", "i", "p") and len(tag.attrs) == 1 and tag.has_attr("bid"):
            if not tag.contents:
                tag.decompose()
            else:
                tag.unwrap()

    html = soup.prettify()

    return html


def parse_actions(llm_output: Message) -> tuple[list[Action], list[ChatCompletionMessageToolCall]]:
    """Decode an assistant message's tool calls into `Action`s.

    Tool calls whose `arguments` are not valid JSON are skipped and returned in a
    second list so the caller can emit a corrective `role="tool"` reply keyed to
    the call id — keeping the tool_call/tool_result pairing valid and letting the
    model retry. This decoder never raises on malformed model output.
    """
    actions: list[Action] = []
    malformed: list[ChatCompletionMessageToolCall] = []
    if hasattr(llm_output, "tool_calls") and llm_output.tool_calls:
        for tc in llm_output.tool_calls:
            arguments = tc.function.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    logger.warning("Malformed tool call arguments for %s: %s", tc.function.name, arguments[:200])
                    malformed.append(tc)
                    continue
            if tc.function.name is None:
                raise ValueError("Tool call must have a function name.")
            actions.append(Action(id=tc.id, name=tc.function.name, arguments=arguments))
    return actions, malformed

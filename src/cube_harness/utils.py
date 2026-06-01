import json
import re

from bs4 import BeautifulSoup
from cube.core import Action
from litellm import Message


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


def parse_actions(llm_output: Message) -> list[Action]:
    actions = []
    if hasattr(llm_output, "tool_calls") and llm_output.tool_calls:
        for tc in llm_output.tool_calls:
            arguments = tc.function.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError(f"Tool call arguments must be a dictionary object: {arguments}")
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON arguments in tool call: {arguments}")
            if tc.function.name is None:
                raise ValueError("Tool call must have a function name.")
            actions.append(Action(id=tc.id, name=tc.function.name, arguments=arguments))
    return actions

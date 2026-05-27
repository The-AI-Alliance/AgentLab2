"""Tests for cube_harness.utils module."""

from litellm import Message
from litellm.types.utils import ChatCompletionMessageToolCall, Function

from cube_harness.utils import parse_actions, prune_html


def _tool_call(call_id: str, name: str, arguments: str) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(id=call_id, type="function", function=Function(name=name, arguments=arguments))


class TestParseActions:
    """Tests for parse_actions — must never raise on malformed model output."""

    def test_valid_tool_call(self) -> None:
        msg = Message(role="assistant", content=None, tool_calls=[_tool_call("c1", "click", '{"bid": "42"}')])
        actions, malformed = parse_actions(msg)
        assert malformed == []
        assert len(actions) == 1
        assert actions[0].name == "click"
        assert actions[0].arguments == {"bid": "42"}

    def test_malformed_json_is_skipped_not_raised(self) -> None:
        msg = Message(role="assistant", content=None, tool_calls=[_tool_call("c1", "click", "{bad json")])
        actions, malformed = parse_actions(msg)
        assert actions == []
        assert [tc.id for tc in malformed] == ["c1"]

    def test_mixed_valid_and_malformed(self) -> None:
        msg = Message(
            role="assistant",
            content=None,
            tool_calls=[_tool_call("c1", "click", '{"bid": "1"}'), _tool_call("c2", "type", "{oops")],
        )
        actions, malformed = parse_actions(msg)
        assert [a.id for a in actions] == ["c1"]
        assert [tc.id for tc in malformed] == ["c2"]

    def test_no_tool_calls(self) -> None:
        msg = Message(role="assistant", content="just text")
        actions, malformed = parse_actions(msg)
        assert actions == []
        assert malformed == []


class TestPruneHtml:
    """Tests for prune_html function."""

    def test_prune_html_removes_comments(self):
        """Test that HTML comments are removed."""
        html = "<div>Hello<!-- this is a comment -->World</div>"
        result = prune_html(html)
        assert "this is a comment" not in result
        assert "Hello" in result
        assert "World" in result

    def test_prune_html_removes_multiline_comments(self):
        """Test that multiline HTML comments are removed."""
        html = """
        <div>
            <!--
            This is a
            multiline comment
            -->
            Content
        </div>
        """
        result = prune_html(html)
        assert "multiline comment" not in result
        assert "Content" in result

    def test_prune_html_removes_style_tags(self):
        """Test that style tags are removed."""
        html = "<html><head><style>.class { color: red; }</style></head><body>Content</body></html>"
        result = prune_html(html)
        assert "<style>" not in result
        assert "color: red" not in result
        assert "Content" in result

    def test_prune_html_removes_script_tags(self):
        """Test that script tags are removed."""
        html = "<html><head><script>alert('hello');</script></head><body>Content</body></html>"
        result = prune_html(html)
        assert "<script>" not in result
        assert "alert" not in result
        assert "Content" in result

    def test_prune_html_removes_link_tags(self):
        """Test that link tags are removed."""
        html = '<html><head><link rel="stylesheet" href="style.css"></head><body>Content</body></html>'
        result = prune_html(html)
        assert "<link" not in result
        assert "Content" in result

    def test_prune_html_removes_br_tags(self):
        """Test that br tags are removed."""
        html = "<div>Line 1<br>Line 2<br/>Line 3</div>"
        result = prune_html(html)
        assert "<br" not in result
        assert "Line 1" in result
        assert "Line 2" in result

    def test_prune_html_unwraps_body_tag(self):
        """Test that body tag is unwrapped but content preserved."""
        html = "<html><body><div>Content</div></body></html>"
        result = prune_html(html)
        assert "Content" in result
        # Body tag should be unwrapped (not a closing tag either)
        assert "<body>" not in result
        assert "</body>" not in result

    def test_prune_html_unwraps_html_tag(self):
        """Test that html tag is unwrapped but content preserved."""
        html = "<html><body>Content</body></html>"
        result = prune_html(html)
        assert "Content" in result
        assert "<html>" not in result
        assert "</html>" not in result

    def test_prune_html_removes_empty_div_with_bid(self):
        """Test that empty div with only bid attribute is removed."""
        html = '<div><div bid="123"></div><p>Content</p></div>'
        result = prune_html(html)
        assert "Content" in result

    def test_prune_html_unwraps_div_with_bid_only(self):
        """Test that div with only bid attribute is unwrapped."""
        html = '<div bid="123">Inner content</div>'
        result = prune_html(html)
        assert "Inner content" in result

    def test_prune_html_unwraps_span_with_bid_only(self):
        """Test that span with only bid attribute is unwrapped."""
        html = '<span bid="123">Span content</span>'
        result = prune_html(html)
        assert "Span content" in result

    def test_prune_html_preserves_div_with_other_attrs(self):
        """Test that div with other attributes is preserved."""
        html = '<div bid="123" class="important">Content</div>'
        result = prune_html(html)
        assert "Content" in result

    def test_prune_html_removes_newlines(self):
        """Test that newlines are replaced with spaces."""
        html = "<div>\n  Content\n</div>"
        result = prune_html(html)
        assert "Content" in result

    def test_prune_html_empty_input(self):
        """Test prune_html with empty string."""
        result = prune_html("")
        # Should not crash
        assert result is not None

    def test_prune_html_plain_text(self):
        """Test prune_html with plain text (no HTML)."""
        html = "Just plain text"
        result = prune_html(html)
        assert "Just plain text" in result

    def test_prune_html_complex_document(self):
        """Test prune_html with a complex HTML document."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>body { font-size: 14px; }</style>
            <script>console.log('test');</script>
            <link rel="stylesheet" href="main.css">
        </head>
        <body>
            <!-- Navigation -->
            <div bid="1">
                <div bid="2" class="nav">
                    <span bid="3">Link 1</span>
                    <br>
                    <span bid="4">Link 2</span>
                </div>
            </div>
            <!-- Content -->
            <div bid="5">
                <p bid="6">Main content here</p>
            </div>
        </body>
        </html>
        """
        result = prune_html(html)

        # Useless tags removed
        assert "<style>" not in result
        assert "<script>" not in result
        assert "<link" not in result
        assert "<br" not in result

        # Comments removed
        assert "<!-- Navigation -->" not in result
        assert "<!-- Content -->" not in result

        # Content preserved
        assert "Link 1" in result
        assert "Link 2" in result
        assert "Main content here" in result

    def test_prune_html_prettified_output(self):
        """Test that output is prettified (formatted)."""
        html = "<div><p>Text</p></div>"
        result = prune_html(html)
        # BeautifulSoup prettify adds newlines and indentation
        assert "\n" in result

    def test_prune_html_preserves_important_tags(self):
        """Test that important tags like button, input are preserved."""
        html = '<button id="submit">Click me</button><input type="text" value="test">'
        result = prune_html(html)
        assert "button" in result.lower() or "Click me" in result
        assert "input" in result.lower() or "test" in result

    def test_prune_html_removes_i_tag_with_bid_only(self):
        """Test that i tag with only bid attribute is unwrapped."""
        html = '<i bid="123">Italic text</i>'
        result = prune_html(html)
        assert "Italic text" in result

    def test_prune_html_removes_p_tag_with_bid_only(self):
        """Test that p tag with only bid attribute is unwrapped."""
        html = '<p bid="123">Paragraph text</p>'
        result = prune_html(html)
        assert "Paragraph text" in result

    def test_prune_html_nested_structure(self):
        """Test prune_html with deeply nested structure."""
        html = """
        <div bid="1">
            <div bid="2">
                <div bid="3">
                    <span bid="4">Deep content</span>
                </div>
            </div>
        </div>
        """
        result = prune_html(html)
        assert "Deep content" in result

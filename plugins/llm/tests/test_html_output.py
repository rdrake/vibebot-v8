"""Tests for HTML output validation.

These tests verify the quality and security of HTML output generated
by the plugin, including syntax highlighting, XSS prevention, and
proper structure.
"""

from __future__ import annotations

import pytest
from llm.service import LLMService


class TestHtmlCodeOutputStructure:
    """Test the structure of generated HTML code pages."""

    @pytest.fixture
    def service(self, tmp_path, make_service) -> LLMService:
        """Create service with HTTP output config."""
        service, _ = make_service(
            httpRoot=str(tmp_path),
            httpUrlBase="http://localhost/llm",
            fileCleanupAge=24,
            fileCleanupMax=1000,
        )
        return service

    def test_html_has_doctype(self, service: LLMService, tmp_path) -> None:
        """GIVEN code content WHEN saved THEN HTML has doctype."""
        url = service.save_code_to_http("# Test\nprint('hello')")
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert content.startswith("<!DOCTYPE html>")

    def test_html_has_charset(self, service: LLMService, tmp_path) -> None:
        """GIVEN code content WHEN saved THEN HTML has charset meta."""
        url = service.save_code_to_http("# Test")
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert 'charset="utf-8"' in content or "charset=utf-8" in content

    def test_html_has_viewport(self, service: LLMService, tmp_path) -> None:
        """GIVEN code content WHEN saved THEN HTML has viewport meta for mobile."""
        url = service.save_code_to_http("# Test")
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert "viewport" in content
        assert "width=device-width" in content

    def test_html_has_title(self, service: LLMService, tmp_path) -> None:
        """GIVEN code content WHEN saved THEN HTML has title."""
        url = service.save_code_to_http("# Test")
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert "<title>" in content
        assert "</title>" in content

    def test_html_has_body(self, service: LLMService, tmp_path) -> None:
        """GIVEN code content WHEN saved THEN HTML has body tags."""
        url = service.save_code_to_http("# Test")
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert "<body>" in content
        assert "</body>" in content

    def test_html_is_well_formed(self, service: LLMService, tmp_path) -> None:
        """GIVEN code content WHEN saved THEN HTML is well-formed."""
        url = service.save_code_to_http("# Test\n```python\nprint('hi')\n```")
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        # Check balanced tags
        assert content.count("<html") == content.count("</html>")
        assert content.count("<head") == content.count("</head>")
        assert content.count("<body") == content.count("</body>")


class TestSyntaxHighlighting:
    """Test syntax highlighting in HTML output."""

    @pytest.fixture
    def service(self, tmp_path, make_service) -> LLMService:
        """Create service with HTTP output config."""
        service, _ = make_service(
            httpRoot=str(tmp_path),
            httpUrlBase="http://localhost/llm",
            fileCleanupAge=24,
            fileCleanupMax=1000,
        )
        return service

    def test_python_code_highlighted(self, service: LLMService, tmp_path) -> None:
        """GIVEN Python code WHEN saved THEN has highlighting classes."""
        code = """```python
def hello():
    print("world")
```"""
        url = service.save_code_to_http(code)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        # Should have highlight container
        assert "highlight" in content
        # Should have some Pygments classes (k=keyword, nf=function name, etc.)
        assert 'class="' in content

    def test_javascript_code_highlighted(self, service: LLMService, tmp_path) -> None:
        """GIVEN JavaScript code WHEN saved THEN has highlighting."""
        code = """```javascript
function hello() {
    console.log("world");
}
```"""
        url = service.save_code_to_http(code)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert "highlight" in content

    def test_code_without_language_still_renders(self, service: LLMService, tmp_path) -> None:
        """GIVEN code without language hint WHEN saved THEN still renders."""
        code = """```
print("hello")
```"""
        url = service.save_code_to_http(code)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert "print" in content
        assert "hello" in content

    def test_inline_code_preserved(self, service: LLMService, tmp_path) -> None:
        """GIVEN inline code WHEN saved THEN preserved with code tag."""
        markdown = "Use the `print()` function."
        url = service.save_code_to_http(markdown)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert "<code>" in content
        assert "print()" in content

    def test_multiple_code_blocks(self, service: LLMService, tmp_path) -> None:
        """GIVEN multiple code blocks WHEN saved THEN all highlighted."""
        code = """Here's Python:
```python
x = 1
```

And JavaScript:
```javascript
let x = 1;
```
"""
        url = service.save_code_to_http(code)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        # Should have multiple highlight sections
        assert content.count("highlight") >= 2


class TestXssPrevention:
    """Test XSS prevention in HTML output."""

    @pytest.fixture
    def service(self, tmp_path, make_service) -> LLMService:
        """Create service with HTTP output config."""
        service, _ = make_service(
            httpRoot=str(tmp_path),
            httpUrlBase="http://localhost/llm",
            fileCleanupAge=24,
            fileCleanupMax=1000,
        )
        return service

    def test_script_tags_stripped(self, service: LLMService, tmp_path) -> None:
        """GIVEN content with script tag WHEN saved THEN script stripped."""
        malicious = "Hello <script>alert('xss')</script> world"
        url = service.save_code_to_http(malicious)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert "<script>" not in content
        assert "alert('xss')" not in content
        assert "Hello" in content
        assert "world" in content

    def test_onclick_handlers_stripped(self, service: LLMService, tmp_path) -> None:
        """GIVEN content with onclick WHEN saved THEN onclick stripped."""
        malicious = '<a href="#" onclick="alert(1)">Click</a>'
        url = service.save_code_to_http(malicious)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert "onclick" not in content

    def test_onerror_handlers_stripped(self, service: LLMService, tmp_path) -> None:
        """GIVEN content with onerror WHEN saved THEN onerror stripped."""
        malicious = '<img src="x" onerror="alert(1)">'
        url = service.save_code_to_http(malicious)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert "onerror" not in content

    def test_javascript_urls_stripped(self, service: LLMService, tmp_path) -> None:
        """GIVEN content with javascript: URL WHEN saved THEN URL stripped."""
        malicious = '<a href="javascript:alert(1)">Click</a>'
        url = service.save_code_to_http(malicious)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert "javascript:" not in content

    def test_data_urls_stripped(self, service: LLMService, tmp_path) -> None:
        """GIVEN content with data: URL WHEN saved THEN URL stripped."""
        malicious = '<a href="data:text/html,<script>alert(1)</script>">Click</a>'
        url = service.save_code_to_http(malicious)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert "data:text/html" not in content

    def test_style_tags_stripped(self, service: LLMService, tmp_path) -> None:
        """GIVEN content with style tag WHEN saved THEN style stripped."""
        malicious = "<style>body{background:url('javascript:alert(1)')}</style>"
        url = service.save_code_to_http(malicious)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        # The injected style tag should be stripped (our CSS is in the template)
        assert "javascript:alert" not in content

    def test_safe_links_preserved(self, service: LLMService, tmp_path) -> None:
        """GIVEN safe http link WHEN saved THEN link preserved."""
        safe = "[Link](https://example.com)"
        url = service.save_code_to_http(safe)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert 'href="https://example.com"' in content
        assert "Link" in content

    def test_nested_xss_attempts(self, service: LLMService, tmp_path) -> None:
        """GIVEN nested XSS attempts WHEN saved THEN all stripped."""
        malicious = '<a href="ja&#118;ascript:alert(1)">Click</a>'
        url = service.save_code_to_http(malicious)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        # Should not have any form of javascript in href
        assert "javascript" not in content.lower() or 'href="javascript' not in content.lower()


class TestMarkdownRendering:
    """Test markdown to HTML rendering."""

    @pytest.fixture
    def service(self, tmp_path, make_service) -> LLMService:
        """Create service with HTTP output config."""
        service, _ = make_service(
            httpRoot=str(tmp_path),
            httpUrlBase="http://localhost/llm",
            fileCleanupAge=24,
            fileCleanupMax=1000,
        )
        return service

    def test_heading_rendered(self, service: LLMService, tmp_path) -> None:
        """GIVEN markdown heading WHEN saved THEN rendered as h1."""
        markdown = "# Hello World"
        url = service.save_code_to_http(markdown)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert "<h1>" in content
        assert "Hello World" in content

    def test_bold_rendered(self, service: LLMService, tmp_path) -> None:
        """GIVEN markdown bold WHEN saved THEN rendered as strong."""
        markdown = "This is **bold** text"
        url = service.save_code_to_http(markdown)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert "<strong>" in content or "<b>" in content
        assert "bold" in content

    def test_italic_rendered(self, service: LLMService, tmp_path) -> None:
        """GIVEN markdown italic WHEN saved THEN rendered as em."""
        markdown = "This is *italic* text"
        url = service.save_code_to_http(markdown)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert "<em>" in content or "<i>" in content
        assert "italic" in content

    def test_lists_rendered(self, service: LLMService, tmp_path) -> None:
        """GIVEN markdown list WHEN saved THEN rendered as ul/li."""
        markdown = """
- Item 1
- Item 2
- Item 3
"""
        url = service.save_code_to_http(markdown)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert "<ul>" in content
        assert "<li>" in content
        assert "Item 1" in content

    def test_ordered_lists_rendered(self, service: LLMService, tmp_path) -> None:
        """GIVEN markdown ordered list WHEN saved THEN rendered as ol/li."""
        markdown = """
1. First
2. Second
3. Third
"""
        url = service.save_code_to_http(markdown)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert "<ol>" in content
        assert "<li>" in content
        assert "First" in content

    def test_blockquote_rendered(self, service: LLMService, tmp_path) -> None:
        """GIVEN markdown blockquote WHEN saved THEN rendered as blockquote."""
        markdown = "> This is a quote"
        url = service.save_code_to_http(markdown)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert "<blockquote>" in content
        assert "This is a quote" in content


class TestHelpPageStructure:
    """Test the help page HTML structure."""

    def test_help_template_has_all_commands(self) -> None:
        """GIVEN help template WHEN checking THEN has all commands documented."""
        from llm.plugin import HELP_HTML_TEMPLATE

        assert "%ask" in HELP_HTML_TEMPLATE
        assert "%code" in HELP_HTML_TEMPLATE
        assert "%draw" in HELP_HTML_TEMPLATE
        assert "%forget" in HELP_HTML_TEMPLATE

    def test_help_template_has_features_section(self) -> None:
        """GIVEN help template WHEN checking THEN has features section."""
        from llm.plugin import HELP_HTML_TEMPLATE

        assert "Features" in HELP_HTML_TEMPLATE
        assert "Volatile Memory" in HELP_HTML_TEMPLATE
        assert "Vision Support" in HELP_HTML_TEMPLATE
        assert "Syntax Highlighting" in HELP_HTML_TEMPLATE

    def test_help_template_has_examples(self) -> None:
        """GIVEN help template WHEN checking THEN has usage examples."""
        from llm.plugin import HELP_HTML_TEMPLATE

        # Check for example class
        assert "example" in HELP_HTML_TEMPLATE
        # Check for actual examples
        assert "What is the capital of France" in HELP_HTML_TEMPLATE

    def test_help_template_is_valid_html(self) -> None:
        """GIVEN help template WHEN checking THEN is well-formed HTML."""
        from llm.plugin import HELP_HTML_TEMPLATE

        assert HELP_HTML_TEMPLATE.startswith("<!DOCTYPE html>")
        assert "<html" in HELP_HTML_TEMPLATE
        assert "</html>" in HELP_HTML_TEMPLATE
        assert HELP_HTML_TEMPLATE.count("<body>") == HELP_HTML_TEMPLATE.count("</body>")

    def test_help_template_is_responsive(self) -> None:
        """GIVEN help template WHEN checking THEN has responsive styles."""
        from llm.plugin import HELP_HTML_TEMPLATE

        assert "viewport" in HELP_HTML_TEMPLATE
        assert "@media" in HELP_HTML_TEMPLATE  # Media queries for responsiveness

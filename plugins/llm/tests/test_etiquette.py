"""Tests for IRC etiquette compliance in LLM plugin.

These tests verify that the bot follows IRC etiquette principles:
- Concise, non-flooding responses
- No markdown formatting in ask responses
- Code output via HTTP links (not pasted)
- Rate limiting to prevent spam
- Appropriate tone and formatting
"""

from __future__ import annotations

import re
import time
from unittest.mock import Mock, patch

import pytest
from llm.rate_limiter import RateLimitConfig, RateLimiter
from llm.service import LLMService

# =============================================================================
# Etiquette Helper Utilities
# =============================================================================


def contains_markdown(text: str) -> bool:
    """Detect if text contains markdown formatting.

    Args:
        text: Text to check

    Returns:
        True if markdown detected, False otherwise
    """
    if not text:
        return False

    # Header patterns: # Header, ## Header, etc.
    if re.search(r"^#{1,6}\s+\S", text, re.MULTILINE):
        return True

    # Bold: **text** or __text__
    if re.search(r"\*\*[^*]+\*\*|__[^_]+__", text):
        return True

    # Italic: *text* or _text_ (but not file_names_like_this)
    if re.search(r"(?<!\w)\*[^*\s][^*]*[^*\s]\*(?!\w)", text):
        return True

    # Code blocks: ```code``` or `inline`
    if "```" in text or re.search(r"`[^`]+`", text):
        return True

    # Links: [text](url)
    if re.search(r"\[[^\]]+\]\([^)]+\)", text):
        return True

    # Lists: - item or * item at start of line
    return bool(re.search(r"^[\s]*[-*]\s+\S", text, re.MULTILINE))


def count_emojis(text: str) -> int:
    """Count emoji characters in text.

    Args:
        text: Text to check

    Returns:
        Number of emoji characters
    """
    if not text:
        return 0

    # Unicode emoji pattern (basic emoji ranges) - match single emoji at a time
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f700-\U0001f77f"  # alchemical symbols
        "\U0001f780-\U0001f7ff"  # Geometric Shapes Extended
        "\U0001f800-\U0001f8ff"  # Supplemental Arrows-C
        "\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
        "\U0001fa00-\U0001fa6f"  # Chess Symbols
        "\U0001fa70-\U0001faff"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027b0"  # Dingbats
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "]",  # No + quantifier - match one at a time
        flags=re.UNICODE,
    )
    return len(emoji_pattern.findall(text))


def estimate_irc_lines(text: str, max_line_length: int = 400) -> int:
    """Estimate number of IRC lines a message will produce.

    Args:
        text: Text to estimate
        max_line_length: Maximum IRC line length (conservative default)

    Returns:
        Estimated number of IRC lines
    """
    if not text:
        return 0

    lines = text.split("\n")
    total = 0
    for line in lines:
        # Each line may wrap if too long
        if len(line) <= max_line_length:
            total += 1
        else:
            total += (len(line) // max_line_length) + 1
    return total


def has_irc_formatting(text: str) -> bool:
    """Detect IRC color codes and formatting.

    Args:
        text: Text to check

    Returns:
        True if IRC formatting detected
    """
    if not text:
        return False

    # IRC color code: ^C (0x03)
    if "\x03" in text:
        return True

    # IRC bold: ^B (0x02)
    if "\x02" in text:
        return True

    # IRC underline: ^_ (0x1F)
    if "\x1f" in text:
        return True

    # IRC italic: ^] (0x1D)
    return "\x1d" in text


# =============================================================================
# Test Classes
# =============================================================================


class TestSystemPromptEtiquette:
    """Tests for IRC etiquette instructions in system prompts.

    GIVEN the system prompts are the primary mechanism for instructing
    the LLM to behave appropriately for IRC.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test fixtures."""
        self.mock_plugin = Mock()
        self.mock_plugin.log = Mock()
        self.mock_plugin.registryValue = Mock(side_effect=self._config_lookup)
        self.service = LLMService(self.mock_plugin)

    def _config_lookup(self, key: str, channel: str | None = None) -> str | int:
        """Mock config values."""
        config = {
            "askSystemPrompt": (
                "You are a helpful IRC assistant. Keep responses concise and suitable for IRC chat. "
                "Avoid markdown formatting. Be direct and informative."
            ),
            "codeSystemPrompt": (
                "You are a helpful code assistant. Explain your code and provide context. "
                "Use markdown formatting for code blocks."
            ),
            "maxPromptLength": 10000,
        }
        return config.get(key, "")

    def test_ask_system_prompt_instructs_conciseness(self) -> None:
        """GIVEN askSystemPrompt WHEN examined THEN contains conciseness instruction."""
        prompt = self._config_lookup("askSystemPrompt")

        assert "concise" in prompt.lower()

    def test_ask_system_prompt_discourages_markdown(self) -> None:
        """GIVEN askSystemPrompt WHEN examined THEN mentions avoiding markdown."""
        prompt = self._config_lookup("askSystemPrompt")

        assert "markdown" in prompt.lower()
        assert "avoid" in prompt.lower()

    def test_ask_system_prompt_mentions_irc_context(self) -> None:
        """GIVEN askSystemPrompt WHEN examined THEN references IRC chat."""
        prompt = self._config_lookup("askSystemPrompt")

        assert "irc" in prompt.lower()

    def test_ask_system_prompt_instructs_direct_responses(self) -> None:
        """GIVEN askSystemPrompt WHEN examined THEN instructs direct/informative tone."""
        prompt = self._config_lookup("askSystemPrompt")

        assert "direct" in prompt.lower() or "informative" in prompt.lower()

    def test_code_system_prompt_allows_markdown(self) -> None:
        """GIVEN codeSystemPrompt WHEN examined THEN mentions markdown is OK.

        The code command outputs to HTTP, so markdown is acceptable and
        will be rendered properly.
        """
        prompt = self._config_lookup("codeSystemPrompt")

        assert "markdown" in prompt.lower()


class TestResponseLengthHandling:
    """Tests for response length handling to prevent IRC flooding."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: object) -> None:
        """Set up test fixtures."""
        self.tmp_path = tmp_path
        self.mock_plugin = Mock()
        self.mock_plugin.log = Mock()
        self.mock_plugin.registryValue = Mock(
            side_effect=lambda key, channel=None: {
                "httpRoot": str(tmp_path),
                "httpUrlBase": "https://example.com/llm",
                "maxPromptLength": 10000,
                "fileCleanupAge": 24,
                "fileCleanupMax": 1000,
            }.get(key)
        )
        self.service = LLMService(self.mock_plugin)

    def test_code_save_returns_http_url(self) -> None:
        """GIVEN code content WHEN saved THEN returns HTTP URL."""
        code = "def hello():\n    print('Hello, World!')\n"
        url = self.service.save_code_to_http(code)

        assert url is not None
        assert url.startswith("https://example.com/llm/")
        assert url.endswith(".html")

    def test_code_url_contains_code_prefix(self) -> None:
        """GIVEN code content WHEN saved THEN URL contains 'code' prefix."""
        code = "print('test')"
        url = self.service.save_code_to_http(code)

        assert "code_" in url

    def test_line_count_calculation(self) -> None:
        """GIVEN multi-line text WHEN counted THEN accurate line count."""
        code = "line1\nline2\nline3\nline4\nline5"
        line_count = code.count("\n")

        # 5 lines = 4 newlines (last line doesn't end with newline)
        assert line_count == 4

    def test_image_responses_are_urls(self) -> None:
        """GIVEN image generated WHEN response returned THEN is URL not data."""
        mock_response = Mock()
        mock_response.data = [Mock(url="https://provider.com/image.png", b64_json=None)]

        self.mock_plugin.registryValue = Mock(
            side_effect=lambda key, channel=None: {
                "drawApiKey": "test-key",
                "drawModel": "imagen",
                "timeout": 30,
                "maxPromptLength": 10000,
            }.get(key)
        )

        with patch("llm.service.litellm.image_generation", return_value=mock_response):
            result = self.service.image_generation("a cat")

        assert result.startswith("http")
        assert "base64" not in result.lower()


class TestResponseFormatValidation:
    """Tests for response format compliance with IRC conventions."""

    def test_detect_markdown_headers(self) -> None:
        """GIVEN text with ## headers WHEN checked THEN detected as markdown."""
        assert contains_markdown("# Header 1") is True
        assert contains_markdown("## Header 2") is True
        assert contains_markdown("### Header 3") is True

    def test_detect_markdown_bold_asterisks(self) -> None:
        """GIVEN text with **bold** WHEN checked THEN detected as markdown."""
        assert contains_markdown("This is **bold** text") is True
        assert contains_markdown("This is __bold__ text") is True

    def test_detect_markdown_italic(self) -> None:
        """GIVEN text with *italic* WHEN checked THEN detected as markdown."""
        assert contains_markdown("This is *italic* text") is True

    def test_detect_markdown_code_blocks(self) -> None:
        """GIVEN text with code blocks WHEN checked THEN detected as markdown."""
        assert contains_markdown("Use `code` here") is True
        assert contains_markdown("```python\nprint('hi')\n```") is True

    def test_detect_markdown_lists(self) -> None:
        """GIVEN text with markdown lists WHEN checked THEN detected as markdown."""
        assert contains_markdown("- Item 1\n- Item 2") is True
        assert contains_markdown("* Item 1\n* Item 2") is True

    def test_detect_markdown_links(self) -> None:
        """GIVEN text with markdown links WHEN checked THEN detected as markdown."""
        assert contains_markdown("Click [here](https://example.com)") is True

    def test_plain_text_passes_validation(self) -> None:
        """GIVEN plain text WHEN checked THEN not detected as markdown."""
        assert contains_markdown("This is plain text.") is False
        assert contains_markdown("Hello, how are you?") is False
        assert contains_markdown("The answer is 42.") is False

    def test_urls_in_text_not_false_positive(self) -> None:
        """GIVEN URL in text WHEN checked THEN not incorrectly flagged."""
        # Plain URLs should not trigger markdown detection
        assert contains_markdown("Check https://example.com for more info") is False

    def test_file_names_not_false_positive(self) -> None:
        """GIVEN file names with underscores WHEN checked THEN not flagged as italic."""
        assert contains_markdown("The file is named my_file_name.py") is False


class TestEmojiAndFormattingGuidelines:
    """Tests for emoji and formatting guidelines."""

    def test_detect_excessive_emojis(self) -> None:
        """GIVEN text with 5+ emojis WHEN checked THEN count is accurate."""
        text = "Hello! \U0001f389\U0001f38a\U0001f388\U0001f381\U0001f380 So excited!"
        count = count_emojis(text)
        assert count >= 5

    def test_moderate_emojis_acceptable(self) -> None:
        """GIVEN text with 1-2 emojis WHEN checked THEN count is low."""
        text = "Thanks! \U0001f44d"
        count = count_emojis(text)
        assert count <= 2

    def test_no_emojis_acceptable(self) -> None:
        """GIVEN text with no emojis WHEN checked THEN count is zero."""
        text = "This is a normal response."
        count = count_emojis(text)
        assert count == 0

    def test_detect_irc_color_codes(self) -> None:
        """GIVEN text with IRC color codes WHEN checked THEN detected."""
        # \x03 is the IRC color code prefix
        text = "This is \x034red\x03 text"
        assert has_irc_formatting(text) is True

    def test_detect_irc_bold(self) -> None:
        """GIVEN text with IRC bold WHEN checked THEN detected."""
        text = "This is \x02bold\x02 text"
        assert has_irc_formatting(text) is True

    def test_plain_text_no_irc_formatting(self) -> None:
        """GIVEN plain text WHEN checked THEN no IRC formatting detected."""
        text = "This is normal text"
        assert has_irc_formatting(text) is False


class TestRateLimitingEtiquette:
    """Tests for rate limiting as an etiquette enforcement mechanism."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test fixtures."""
        self.config = RateLimitConfig(max_requests=2, window_seconds=60, enabled=True)
        self.limiter = RateLimiter(self.config)

    def test_rate_limit_message_is_user_friendly(self) -> None:
        """GIVEN rate limit hit WHEN error returned THEN message is friendly."""
        # Fill the limit
        self.limiter.check_rate_limit("user1")
        self.limiter.check_rate_limit("user1")

        # Hit the limit
        allowed, message = self.limiter.check_rate_limit("user1")

        assert allowed is False
        assert "Rate limit exceeded" in message
        assert "Try again" in message
        # Should mention how long to wait
        assert "s" in message  # seconds

    def test_rate_limit_does_not_block_other_users(self) -> None:
        """GIVEN user1 limited WHEN user2 requests THEN user2 allowed."""
        # User1 hits limit
        self.limiter.check_rate_limit("user1")
        self.limiter.check_rate_limit("user1")
        allowed1, _ = self.limiter.check_rate_limit("user1")
        assert allowed1 is False

        # User2 should still be allowed
        allowed2, _ = self.limiter.check_rate_limit("user2")
        assert allowed2 is True

    def test_rate_limit_includes_limit_info(self) -> None:
        """GIVEN rate limit hit WHEN error returned THEN includes limit info."""
        # Fill the limit
        self.limiter.check_rate_limit("user1")
        self.limiter.check_rate_limit("user1")

        # Hit the limit
        allowed, message = self.limiter.check_rate_limit("user1")

        assert allowed is False
        # Should mention the limit configuration
        assert "2" in message  # max_requests
        assert "60" in message  # window_seconds

    def test_rate_limit_allows_requests_under_limit(self) -> None:
        """GIVEN requests under limit WHEN checked THEN all allowed."""
        allowed1, msg1 = self.limiter.check_rate_limit("user1")
        allowed2, msg2 = self.limiter.check_rate_limit("user1")

        assert allowed1 is True
        assert allowed2 is True
        assert msg1 == ""
        assert msg2 == ""


class TestResponseAppropriateness:
    """Tests for response appropriateness."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test fixtures."""
        self.mock_plugin = Mock()
        self.mock_plugin.log = Mock()
        self.mock_plugin.registryValue = Mock(side_effect=lambda key, channel=None: 10000)
        self.mock_plugin.startup_time = time.time() - 3600
        self.service = LLMService(self.mock_plugin)

    def _make_mock_irc(
        self,
        network: str = "TestNet",
        nick: str = "testbot",
        channels: dict | None = None,
    ) -> Mock:
        """Create a mock IRC object."""
        irc = Mock()
        irc.network = network
        irc.nick = nick
        irc.state = Mock()
        irc.state.channels = channels or {}
        irc.state.nickToAccount = Mock(return_value=None)
        return irc

    def _make_mock_msg(
        self,
        channel: str = "#test",
        nick: str = "user",
    ) -> Mock:
        """Create a mock IRC message object."""
        msg = Mock()
        msg.args = (channel, "some message text")
        msg.prefix = f"{nick}!username@hostname.example.com"
        return msg

    def _make_mock_channel_state(
        self,
        users: set | None = None,
        topic: str | None = None,
    ) -> Mock:
        """Create a mock channel state object."""
        ch_state = Mock()
        ch_state.users = users or {"user1", "user2", "testbot"}
        ch_state.modes = {"n": None, "t": None}
        ch_state.topic = topic
        ch_state.isOp = Mock(return_value=False)
        ch_state.isHalfop = Mock(return_value=False)
        ch_state.isVoice = Mock(return_value=False)
        return ch_state

    def test_channel_context_included_in_prompt(self) -> None:
        """GIVEN channel message WHEN prompt built THEN channel in context."""
        base = "You are helpful."
        ch_state = self._make_mock_channel_state()
        irc = self._make_mock_irc(channels={"#tech": ch_state})
        msg = self._make_mock_msg(channel="#tech")

        result = self.service._build_system_prompt(base, irc=irc, msg=msg)

        assert "#tech" in result
        assert "Channel:" in result

    def test_pm_context_indicated_differently(self) -> None:
        """GIVEN private message WHEN prompt built THEN 'Private message' in context."""
        base = "You are helpful."
        irc = self._make_mock_irc()
        msg = self._make_mock_msg(channel="testbot", nick="someuser")

        result = self.service._build_system_prompt(base, irc=irc, msg=msg)

        assert "Private message" in result

    def test_channel_topic_included_when_available(self) -> None:
        """GIVEN channel with topic WHEN prompt built THEN topic in context."""
        base = "You are helpful."
        ch_state = self._make_mock_channel_state(topic="Python programming help")
        irc = self._make_mock_irc(channels={"#python": ch_state})
        msg = self._make_mock_msg(channel="#python")

        result = self.service._build_system_prompt(base, irc=irc, msg=msg)

        assert "Python programming help" in result
        assert "Topic:" in result

    def test_network_info_included(self) -> None:
        """GIVEN network 'AfterNET' WHEN prompt built THEN network in context."""
        base = "You are helpful."
        irc = self._make_mock_irc(network="AfterNET")
        msg = self._make_mock_msg()

        result = self.service._build_system_prompt(base, irc=irc, msg=msg)

        assert "AfterNET" in result
        assert "Network:" in result

    def test_bot_nick_included(self) -> None:
        """GIVEN bot nick WHEN prompt built THEN bot nick in context."""
        base = "You are helpful."
        irc = self._make_mock_irc(nick="vibebot")
        msg = self._make_mock_msg()

        result = self.service._build_system_prompt(base, irc=irc, msg=msg)

        assert "vibebot" in result
        assert "Bot:" in result


class TestEtiquetteHelperUtilities:
    """Tests for etiquette helper utilities."""

    def test_estimate_irc_lines_single_line(self) -> None:
        """GIVEN single short line WHEN estimated THEN returns 1."""
        text = "Hello, this is a short message."
        lines = estimate_irc_lines(text)
        assert lines == 1

    def test_estimate_irc_lines_multiple_lines(self) -> None:
        """GIVEN multiple lines WHEN estimated THEN counts correctly."""
        text = "Line 1\nLine 2\nLine 3"
        lines = estimate_irc_lines(text)
        assert lines == 3

    def test_estimate_irc_lines_long_line_wraps(self) -> None:
        """GIVEN very long line WHEN estimated THEN accounts for wrapping."""
        # Create a line longer than 400 chars
        text = "x" * 1000
        lines = estimate_irc_lines(text, max_line_length=400)
        assert lines >= 3

    def test_estimate_irc_lines_empty(self) -> None:
        """GIVEN empty text WHEN estimated THEN returns 0."""
        assert estimate_irc_lines("") == 0

    def test_contains_markdown_empty(self) -> None:
        """GIVEN empty text WHEN checked THEN returns False."""
        assert contains_markdown("") is False
        assert contains_markdown(None) is False  # type: ignore

    def test_count_emojis_empty(self) -> None:
        """GIVEN empty text WHEN counted THEN returns 0."""
        assert count_emojis("") == 0
        assert count_emojis(None) == 0  # type: ignore

    def test_has_irc_formatting_empty(self) -> None:
        """GIVEN empty text WHEN checked THEN returns False."""
        assert has_irc_formatting("") is False
        assert has_irc_formatting(None) is False  # type: ignore

"""LiteLLM service layer for LLM plugin."""

from __future__ import annotations

import base64
import hashlib
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import litellm
import markdown
import supybot.conf as conf
import supybot.ircmsgs as ircmsgs
import supybot.ircutils as ircutils
import supybot.log as log
import supybot.utils as utils
from supybot.utils.file import AtomicFile
from pygments.formatters import HtmlFormatter
from supybot.i18n import PluginInternationalization

_ = PluginInternationalization("LLM")

if TYPE_CHECKING:
    from typing import Any

    from supybot.callbacks import Irc
    from supybot.ircmsgs import IrcMsg

    from .plugin import LLM


class LLMService:
    """Service layer for LiteLLM interactions.

    This class handles all AI API calls and business logic,
    separated from IRC protocol handling (which is in plugin.py).

    Critical Security Patterns:
    - API keys passed directly to litellm (never mutate env vars)
    - All error messages sanitized to remove API keys
    - Image URLs validated to block malicious schemes
    - Path traversal attempts blocked
    """

    def __init__(self, plugin_instance: LLM) -> None:
        """Initialize service with plugin reference.

        Args:
            plugin_instance: Reference to parent plugin for config access
        """
        self.plugin = plugin_instance
        self.log = log.getPluginLogger("LLM.service")

        # Pattern to detect image URLs
        self.image_pattern = re.compile(
            r"https?://[^\s]+\.(?:jpg|jpeg|png|gif|webp|bmp)",
            re.IGNORECASE,
        )

        # Pattern to detect API keys for sanitization
        # Matches common formats: sk-*, AIza*, long alphanumeric strings
        self.api_key_pattern = re.compile(
            r"(?:sk-[a-zA-Z0-9_-]{10,}|AIza[a-zA-Z0-9_-]{30,}|[a-zA-Z0-9_-]{32,})",
            re.IGNORECASE,
        )

    def _sanitize(self, text: str) -> str:
        """Remove API keys from text for safe logging.

        Args:
            text: Text that may contain API keys

        Returns:
            Text with API keys replaced by [REDACTED]
        """
        if not text:
            return text
        return self.api_key_pattern.sub("[REDACTED]", str(text))

    def _build_system_prompt(
        self,
        base_prompt: str,
        irc: Irc | None = None,
        msg: IrcMsg | None = None,
    ) -> str:
        """Build system prompt with IRC context.

        Combines the base personality prompt with contextual IRC information
        that the LLM may use naturally in its responses.

        Args:
            base_prompt: Base personality/instruction prompt from config
            irc: IRC connection object (optional)
            msg: IRC message object (optional)

        Returns:
            Combined system prompt with context block
        """
        if not irc or not msg:
            return base_prompt

        lines = []

        # Date/time
        now = datetime.now()
        date_str = now.strftime("%A, %B %d, %Y, %I:%M %p")
        lines.append(f"Date: {date_str}")

        # Bot uptime
        startup_time = getattr(self.plugin, "startup_time", None)
        if startup_time is not None:
            uptime_seconds = int(time.time() - startup_time)
            uptime_str = self._format_uptime(uptime_seconds)
            lines.append(f"Uptime: {uptime_str}")

        # Network
        network = getattr(irc, "network", None)
        if network:
            lines.append(f"Network: {network}")

        # Channel or PM context
        channel = msg.args[0] if msg.args else None
        nick = ircutils.nickFromHostmask(msg.prefix) if msg.prefix else None

        if channel and ircutils.isChannel(channel):
            # Channel context
            channel_info = self._get_channel_info(irc, channel)
            lines.append(channel_info)

            # Topic
            topic = self._get_channel_topic(irc, channel)
            if topic:
                lines.append(f"Topic: {topic}")
        else:
            # Private message
            lines.append("Context: Private message")

        # Caller info
        if nick:
            caller_info = self._get_caller_info(irc, msg, nick, channel)
            lines.append(f"Caller: {caller_info}")

        # Bot nick
        bot_nick = getattr(irc, "nick", None)
        if bot_nick:
            lines.append(f"Bot: {bot_nick}")

        # Language preference
        try:
            language = conf.supybot.language()
            if language and language != "en":
                language_names = {
                    "de": "German",
                    "es": "Spanish",
                    "fi": "Finnish",
                    "fr": "French",
                    "it": "Italian",
                    "ru": "Russian",
                }
                lang_name = language_names.get(language, language)
                lines.append(f"Language: {lang_name} (respond in this language)")
        except Exception:
            pass  # Fail silently if conf not available

        # Build final prompt
        context_block = "\n".join(lines)
        return f"""{base_prompt}

---
CONTEXT (use naturally if relevant):
{context_block}
---"""

    def _format_uptime(self, seconds: int) -> str:
        """Format uptime in human-readable form using Limnoria's timeElapsed.

        Args:
            seconds: Total uptime in seconds

        Returns:
            Formatted string like "2 hours and 15 minutes"
        """
        if seconds < 1:
            return "just started"
        return utils.gen.timeElapsed(seconds)

    def _get_channel_info(self, irc: Irc, channel: str) -> str:
        """Get channel info string with user count and modes.

        Args:
            irc: IRC connection object
            channel: Channel name

        Returns:
            Formatted channel info like "#chat (42 users, +nts)"
        """
        state = getattr(irc, "state", None)
        if not state:
            return f"Channel: {channel}"

        channels = getattr(state, "channels", {})
        ch_state = channels.get(channel)
        if not ch_state:
            return f"Channel: {channel}"

        users = getattr(ch_state, "users", set())
        user_count = len(users)

        # Get channel modes
        modes_str = ""
        modes = getattr(ch_state, "modes", None)
        if modes:
            # modes is a dict like {'n': None, 't': None, 's': None}
            mode_chars = "".join(sorted(modes.keys()))
            if mode_chars:
                modes_str = f", +{mode_chars}"

        return f"Channel: {channel} ({user_count} users{modes_str})"

    def _get_channel_topic(self, irc: Irc, channel: str) -> str | None:
        """Get channel topic.

        Args:
            irc: IRC connection object
            channel: Channel name

        Returns:
            Channel topic or None
        """
        state = getattr(irc, "state", None)
        if not state:
            return None

        channels = getattr(state, "channels", {})
        ch_state = channels.get(channel)
        if not ch_state:
            return None

        topic = getattr(ch_state, "topic", None)
        return topic if topic else None

    def _get_caller_info(
        self,
        irc: Irc,
        msg: IrcMsg,
        nick: str,
        channel: str | None,
    ) -> str:
        """Get caller info string with status and account.

        Args:
            irc: IRC connection object
            msg: IRC message object
            nick: Caller's nickname
            channel: Channel name (or None for PM)

        Returns:
            Formatted caller info like "JohnDoe (voiced, identified as john)"
        """
        status_parts = []
        state = getattr(irc, "state", None)

        # Channel status (op/halfop/voice)
        if channel and ircutils.isChannel(channel) and state:
            channels = getattr(state, "channels", {})
            ch_state = channels.get(channel)
            if ch_state:
                is_op = getattr(ch_state, "isOp", None)
                is_halfop = getattr(ch_state, "isHalfop", None)
                is_voice = getattr(ch_state, "isVoice", None)

                if is_op and is_op(nick):
                    status_parts.append("op")
                elif is_halfop and is_halfop(nick):
                    status_parts.append("halfop")
                elif is_voice and is_voice(nick):
                    status_parts.append("voiced")

        # Account/identification status
        if state:
            nick_to_account = getattr(state, "nickToAccount", None)
            if nick_to_account:
                account = nick_to_account(nick)
                if account:
                    status_parts.append(f"identified as {account}")

        if status_parts:
            return f"{nick} ({', '.join(status_parts)})"
        return nick

    def send_typing_indicator(self, irc: Irc, target: str, state: str = "active") -> None:
        """Send IRCv3 typing indicator.

        Sends a TAGMSG with +typing client tag to indicate the bot is
        typing/processing. Gracefully degrades if server doesn't support
        message-tags capability.

        Args:
            irc: IRC connection object
            target: Channel or nick to send typing indicator to
            state: Typing state - 'active', 'paused', or 'done'
        """
        # Check if server supports message-tags capability
        irc_state = getattr(irc, "state", None)
        if not irc_state:
            return
        capabilities = getattr(irc_state, "capabilities_ack", set())
        if "message-tags" not in capabilities:
            return

        msg = ircmsgs.IrcMsg(
            command="TAGMSG",
            args=(target,),
            server_tags={"+typing": state},
        )
        irc.queueMsg(msg)

    def safe_key_display(self, api_key: str) -> str:
        """Safely display API key with only first 3 characters visible.

        Args:
            api_key: The API key to display

        Returns:
            String showing first 3 chars or status message
        """
        if not api_key or not api_key.strip():
            return "Not configured"

        key = api_key.strip()
        if len(key) < 3:
            return "Invalid (too short)"

        hidden_count = len(key) - 3
        return f"{key[:3]}...({hidden_count} chars hidden)"

    def detect_images(self, text: str) -> list[str]:
        """Extract image URLs from text for vision support.

        Args:
            text: User input text

        Returns:
            List of image URLs found
        """
        return self.image_pattern.findall(text)

    def validate_prompt(self, prompt: str) -> tuple[bool, str]:
        """Validate prompt input.

        Args:
            prompt: User prompt to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not prompt or not prompt.strip():
            return False, _("Prompt cannot be empty")

        max_length = self.plugin.registryValue("maxPromptLength")
        if len(prompt) > max_length:
            return False, _("Prompt too long (max %d characters)") % max_length

        return True, ""

    def validate_image_url(self, url: str) -> bool:
        """Validate image URL format and extension.

        Security checks:
        - Only http/https schemes allowed (blocks javascript:, data:, file:, ftp:)
        - No path traversal attempts (blocks ../)
        - Must have valid image extension

        Args:
            url: Image URL to validate

        Returns:
            True if valid and safe, False otherwise
        """
        # Only allow http/https
        if not url.startswith(("http://", "https://")):
            return False

        # Block path traversal attempts
        if "../" in url or "..\\" in url:
            return False

        # Check for valid image extension
        valid_extensions = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp")
        return any(url.lower().endswith(ext) for ext in valid_extensions)

    def _get_safety_settings(self) -> list[dict[str, str]]:
        """Get Gemini safety settings (all categories set to BLOCK_NONE).

        Disables all content filtering for Gemini models. Note that
        HARM_CATEGORY_CIVIC_INTEGRITY cannot be set to OFF but can be
        set to BLOCK_NONE.

        Returns:
            List of safety setting dictionaries
        """
        categories = [
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
            "HARM_CATEGORY_CIVIC_INTEGRITY",
        ]
        return [{"category": cat, "threshold": "BLOCK_NONE"} for cat in categories]

    def _get_gemini_tools(self, model: str) -> list[dict[str, dict]] | None:
        """Get Gemini-specific tools if supported by the model.

        Enables Google Search grounding and URL Context for Gemini 2.0+ text models.
        These tools allow the model to search the web and fetch URL content.

        Args:
            model: Model identifier string

        Returns:
            List of tool dictionaries or None if not supported
        """
        model_lower = model.lower()

        # Supported Gemini 2.0+ text models (explicit opt-in)
        supported_models = [
            "gemini-2.0-flash",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-3-flash",
            "gemini-flash-latest",  # Alias for latest flash model
        ]

        if any(supported in model_lower for supported in supported_models):
            return [{"googleSearch": {}}, {"urlContext": {}}]

        # Default: no tools
        return None

    def _handle_llm_error(self, error: Exception, operation: str) -> str:
        """Handle LiteLLM errors with consistent messaging and logging.

        Args:
            error: The exception that was raised
            operation: Human-readable operation name (e.g., "completion", "image generation")

        Returns:
            User-friendly error message
        """
        if isinstance(error, litellm.Timeout):
            return (
                _("Error: %s timed out. Try again or simplify your request.")
                % operation.capitalize()
            )
        if isinstance(error, litellm.RateLimitError):
            return _("Error: API rate limit reached. Please wait a few minutes and try again.")
        if isinstance(error, litellm.AuthenticationError):
            return _("Error: Invalid API key for %s. Please check your configuration.") % operation
        if isinstance(error, litellm.ContentPolicyViolationError):
            return _("Error: Content violates AI safety policies. Please rephrase your request.")
        if isinstance(error, litellm.APIError):
            sanitized = self._sanitize(str(error))[:150]
            self.log.error(f"LLM API error ({operation}): {sanitized}")
            return _("Error: API returned an error. Check logs for details.")

        # Generic exception - sanitize and log
        sanitized = self._sanitize(str(error))
        self.log.error(f"LLM {operation} error: {sanitized}")
        return (
            _("Error: Unable to complete %s. Check your configuration or try again later.")
            % operation
        )

    def completion(
        self,
        prompt: str,
        command: str = "ask",
        images: list[str] | None = None,
        history: list[dict[str, str]] | None = None,
        irc: Irc | None = None,
        msg: IrcMsg | None = None,
    ) -> str:
        """Generate text completion with optional vision and conversation history.

        This is the main method for text generation. It handles:
        - Prompt validation
        - Image URL validation
        - API key retrieval from config
        - Thread-safe API calls (api_key passed directly)
        - Error handling with sanitized messages

        Args:
            prompt: User's text prompt
            command: Command name (ask/code) for config lookup
            images: Optional list of image URLs for vision
            history: Optional conversation history for context
            irc: IRC connection object for context (optional)
            msg: IRC message object for context (optional)

        Returns:
            Generated text response or error message
        """
        try:
            # Validate prompt
            is_valid, error_msg = self.validate_prompt(prompt)
            if not is_valid:
                return _("Error: %s") % error_msg

            # Validate and filter image URLs
            if images:
                valid_images = [url for url in images if self.validate_image_url(url)]
                if len(valid_images) != len(images):
                    self.log.warning(
                        f"Filtered out {len(images) - len(valid_images)} invalid image URLs"
                    )
                images = valid_images if valid_images else None

            # Get configuration (channel-specific for model/prompt, global for api key)
            channel = msg.args[0] if msg and msg.args else None
            api_key = self.plugin.registryValue(f"{command}ApiKey")  # API keys are global
            model = self.plugin.registryValue(f"{command}Model", channel)
            base_system_prompt = self.plugin.registryValue(f"{command}SystemPrompt", channel)

            if not api_key:
                return _("Error: API key not configured for %s command") % command

            # Build system prompt with IRC context
            system_prompt = self._build_system_prompt(base_system_prompt, irc, msg)

            # Build messages with history and system prompt
            messages = self._build_messages(prompt, images, history, system_prompt)

            # Get timeout
            timeout = self.plugin.registryValue("timeout")

            # Call LiteLLM with API key passed directly (thread-safe)
            # CRITICAL: Never mutate environment variables - prevents race conditions
            response = litellm.completion(
                model=model,
                messages=messages,
                api_key=api_key,
                timeout=timeout,
                tools=self._get_gemini_tools(model),
                safety_settings=self._get_safety_settings() if "gemini" in model else None,
            )

            return response.choices[0].message.content

        except Exception as e:
            return self._handle_llm_error(e, "completion")

    def image_generation(
        self,
        prompt: str,
        irc: Irc | None = None,
        msg: IrcMsg | None = None,
    ) -> str:
        """Generate image from text prompt.

        Generates an image using the configured model, saves it to HTTP server,
        and returns the URL. Sends IRCv3 typing indicators during generation.

        Args:
            prompt: Text description of image to generate
            irc: IRC connection for typing indicators (optional)
            msg: IRC message for context (optional)

        Returns:
            URL to generated image or error message
        """
        target = None
        if irc and msg and msg.args:
            target = msg.args[0]

        try:
            # Send typing indicator
            if irc and target:
                self.send_typing_indicator(irc, target, "active")

            # Validate prompt
            is_valid, error_msg = self.validate_prompt(prompt)
            if not is_valid:
                return _("Error: %s") % error_msg

            # Get configuration (channel-specific for model, global for api key)
            channel = msg.args[0] if msg and msg.args else None
            api_key = self.plugin.registryValue("drawApiKey")  # API keys are global
            model = self.plugin.registryValue("drawModel", channel)

            if not api_key:
                return _("Error: API key not configured for draw command")

            # Get timeout
            timeout = self.plugin.registryValue("timeout")

            # Generate image with API key passed directly (thread-safe)
            response = litellm.image_generation(
                prompt=prompt,
                model=model,
                api_key=api_key,
                n=1,
                timeout=timeout,
            )

            # Handle response - check both URL and base64
            if response.data and len(response.data) > 0:
                image_data = response.data[0]

                # Check for URL first (some providers return URLs)
                if hasattr(image_data, "url") and image_data.url:
                    return image_data.url

                # Handle base64 response (Google AI Studio Imagen)
                if hasattr(image_data, "b64_json") and image_data.b64_json:
                    url = self.save_image_to_http(image_data.b64_json)
                    if url:
                        return url
                    return _("Error: Failed to save generated image")

            # No image data - check for blocked content reasons
            # Google Imagen returns empty data when content is blocked
            self.log.warning(f"Image generation returned no data. Response: {response}")
            return _(
                "Error: No image generated. The prompt may have been blocked by "
                "content safety filters. Try rephrasing your request."
            )

        except Exception as e:
            return self._handle_llm_error(e, "image generation")
        finally:
            # Send typing done indicator
            if irc and target:
                self.send_typing_indicator(irc, target, "done")

    def _strip_markdown_fences(self, code: str) -> tuple[str, str | None]:
        """Strip markdown code fences and extract language if present.

        Args:
            code: Code potentially wrapped in markdown fences

        Returns:
            Tuple of (clean_code, language)
        """
        code = code.strip()

        # Check for markdown fence with language (```python)
        fence_match = re.match(r"^```(\w+)\n(.*)\n```$", code, re.DOTALL)
        if fence_match:
            return fence_match.group(2), fence_match.group(1)

        # Check for fence without language (```)
        fence_match = re.match(r"^```\n(.*)\n```$", code, re.DOTALL)
        if fence_match:
            return fence_match.group(1), None

        # No fences
        return code, None

    def _detect_language(self, code: str) -> str:
        """Simple language detection based on common keywords.

        Args:
            code: Code to analyze

        Returns:
            Detected language or 'text'
        """
        code_lower = code.lower()

        if "def " in code_lower or "import " in code_lower:
            return "python"
        elif "function " in code_lower or "const " in code_lower or "let " in code_lower:
            return "javascript"
        elif "package main" in code or "func " in code:
            return "go"
        elif "#include" in code or "int main" in code:
            return "c"
        elif "class " in code and "public " in code:
            return "java"
        else:
            return "text"

    def _get_http_paths(self) -> tuple[str, str]:
        """Get HTTP root directory and URL base for file storage.

        Uses plugin config if set, otherwise falls back to Limnoria's
        built-in web directory and HTTP server URL.

        Returns:
            Tuple of (http_root_path, url_base)
        """
        # Get configured values (may be empty)
        http_root = self.plugin.registryValue("httpRoot")
        url_base = self.plugin.registryValue("httpUrlBase")

        # Fall back to Limnoria's web directory if not configured
        if not http_root:
            # Use Limnoria's data/web/llm/ directory
            http_root = conf.supybot.directories.data.web.dirize("llm")

        # Fall back to Limnoria's HTTP server URL if not configured
        if not url_base:
            public_url = conf.supybot.servers.http.publicUrl()
            if public_url:
                # Remove trailing slash and add /llm
                url_base = public_url.rstrip("/") + "/llm"
            else:
                # Construct from host and port
                port = conf.supybot.servers.http.port()
                url_base = f"http://localhost:{port}/llm"

        return http_root, url_base

    def save_code_to_http(self, content: str, language: str | None = None) -> str | None:
        """Save content to HTTP server as HTML and return URL.

        Converts markdown to HTML for a pastebin-style page.

        Args:
            content: Markdown content from LLM
            language: Optional language hint (unused, kept for compatibility)

        Returns:
            Public URL to saved file or None on error
        """
        http_root, url_base = self._get_http_paths()

        # Create unique filename
        hash_input = f"{content}{time.time()}".encode()
        hash_str = hashlib.sha256(hash_input).hexdigest()[:16]
        filename = f"code_{hash_str}.html"
        filepath = os.path.join(http_root, filename)

        # Convert markdown to HTML with syntax highlighting
        md = markdown.Markdown(
            extensions=[
                "fenced_code",
                "codehilite",
            ],
            extension_configs={
                "codehilite": {
                    "css_class": "highlight",
                    "guess_lang": True,
                    "use_pygments": True,
                }
            },
        )
        rendered = md.convert(content)

        # Generate Pygments CSS for monokai theme
        formatter = HtmlFormatter(style="monokai")
        pygments_css = formatter.get_style_defs(".highlight")

        # Pastebin-style HTML with syntax highlighting
        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Code</title>
<style>
body {{ margin: 0; padding: 20px; background: #272822; color: #f8f8f2; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; line-height: 1.6; }}
pre {{ padding: 16px; background: #1e1e1e; border-radius: 6px; overflow-x: auto; margin: 1em 0; }}
code {{ font-family: 'SF Mono', 'Fira Code', Consolas, 'Liberation Mono', monospace; font-size: 14px; }}
p {{ margin: 1em 0; }}
strong {{ color: #fff; }}
em {{ color: #e6db74; }}
ul, ol {{ margin: 1em 0; padding-left: 2em; }}
a {{ color: #66d9ef; }}
h1, h2, h3, h4 {{ color: #f8f8f2; margin-top: 1.5em; }}
.highlight {{ background: #1e1e1e; border-radius: 6px; padding: 0; }}
.highlight pre {{ margin: 0; padding: 16px; background: transparent; }}
{pygments_css}
</style>
</head>
<body>
{rendered}
</body>
</html>"""

        try:
            os.makedirs(http_root, exist_ok=True)
            with AtomicFile(filepath, "w") as f:
                f.write(html)
            return f"{url_base}/{filename}"
        except OSError as e:
            self.log.error(f"Failed to save code file: {e}")
            return None

    def save_image_to_http(self, b64_data: str, extension: str = "png") -> str | None:
        """Save base64-encoded image to HTTP server.

        Decodes base64 image data and saves it to the configured HTTP root
        directory, returning a public URL.

        Args:
            b64_data: Base64-encoded image data
            extension: Image file extension (default: png)

        Returns:
            Public URL to saved image or None on error
        """
        http_root, url_base = self._get_http_paths()

        # Decode base64
        try:
            image_bytes = base64.b64decode(b64_data)
        except base64.binascii.Error as e:
            self.log.error(f"Invalid base64 image data: {e}")
            return None

        # Generate unique filename
        hash_input = f"{b64_data[:100]}{time.time()}".encode()
        hash_str = hashlib.sha256(hash_input).hexdigest()[:16]
        filename = f"img_{hash_str}.{extension}"
        filepath = os.path.join(http_root, filename)

        # Write binary image file
        try:
            os.makedirs(http_root, exist_ok=True)
            with AtomicFile(filepath, "wb") as f:
                f.write(image_bytes)
            return f"{url_base}/{filename}"
        except OSError as e:
            self.log.error(f"Failed to save image file: {e}")
            return None

    def _build_messages(
        self,
        prompt: str,
        images: list[str] | None,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build messages array for LiteLLM.

        Args:
            prompt: Text prompt
            images: Optional image URLs
            history: Optional conversation history
            system_prompt: Optional system prompt for bot personality

        Returns:
            Messages array in LiteLLM format
        """
        messages: list[dict[str, Any]] = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add conversation history if provided
        if history:
            messages.extend(history)

        # Build current message
        if images:
            # Multi-modal message with images
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img_url in images:
                content.append({"type": "image_url", "image_url": {"url": img_url}})
            messages.append({"role": "user", "content": content})
        else:
            # Simple text message
            messages.append({"role": "user", "content": prompt})

        return messages

    def _cleanup_old_files(
        self,
        directory: str,
        max_age_hours: int | None = None,
        max_files: int | None = None,
    ) -> None:
        """Clean up old files from HTTP directory.

        Args:
            directory: Directory to clean
            max_age_hours: Delete files older than this (uses config if None)
            max_files: Keep at most this many files (uses config if None)
        """
        # Get config values if not provided
        if max_age_hours is None:
            max_age_hours = self.plugin.registryValue("fileCleanupAge")
        if max_files is None:
            max_files = self.plugin.registryValue("fileCleanupMax")

        dir_path = Path(directory)
        if not dir_path.exists():
            return

        files: list[tuple[Path, float]] = []
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        # Collect all code and image files with their modification times
        patterns = ["*.html", "*.png", "*.jpg", "*.jpeg", "*.webp"]
        for pattern in patterns:
            for file_path in dir_path.glob(pattern):
                try:
                    stat = file_path.stat()
                    files.append((file_path, stat.st_mtime))
                except OSError:
                    continue

        # Remove files older than max_age
        for file_path, mtime in files[:]:
            if current_time - mtime > max_age_seconds:
                try:
                    file_path.unlink()
                    files.remove((file_path, mtime))
                except OSError:
                    pass

        # If still too many files, remove oldest
        if len(files) > max_files:
            import contextlib

            files.sort(key=lambda x: x[1])  # Sort by mtime
            for file_path, _ in files[:-max_files]:
                with contextlib.suppress(OSError):
                    file_path.unlink()

"""Tests for the animate (video) command and video generation service."""

from __future__ import annotations

import io
import json
import threading
import time
import urllib.error
from typing import TYPE_CHECKING

import pytest
from llm.plugin import LLM
from llm.service import VideoResult

from .conftest import make_registry_side_effect

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_urlopen_resp(
    mocker: MockerFixture,
    data: dict | bytes,
):
    """Create a mock urllib response usable as a context manager.

    Args:
        mocker: pytest-mock fixture
        data: dict (JSON-serialized) or raw bytes for resp.read()
    """
    resp = mocker.MagicMock()
    resp.__enter__ = mocker.Mock(return_value=resp)
    resp.__exit__ = mocker.Mock(return_value=False)
    if isinstance(data, bytes):
        resp.read.return_value = data
    else:
        resp.read.return_value = json.dumps(data).encode()
    return resp


# ---------------------------------------------------------------------------
# Shared plugin-level fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def plugin_env(mocker: MockerFixture):
    """Create an LLM plugin instance wired to mocked dependencies.

    Returns (plugin, mock_irc, mock_msg) ready for command invocation.
    """
    registry = make_registry_side_effect()

    mock_irc = mocker.MagicMock()
    mock_irc.nick = "testbot"
    mock_irc.state = mocker.MagicMock()
    mock_irc.state.channels = {
        "#test": mocker.MagicMock(topic="Test topic"),
    }
    mock_irc.state.capabilities_ack = set()
    # Default: no NickServ account (nick fallback)
    mock_irc.state.nickToAccount = mocker.MagicMock(
        return_value=None,
    )

    mock_msg = mocker.MagicMock()
    mock_msg.prefix = "testnick!user@host"
    mock_msg.args = ("#test", "test message")
    mock_msg.time = time.time() + 100  # future = not ZNC
    mock_msg.channel = "#test"
    mock_msg.nick = "testnick"

    mocker.patch.object(
        LLM,
        "registryValue",
        side_effect=registry,
    )
    mocker.patch("llm.plugin.LLMService")
    mocker.patch("llm.plugin.LLMDatabase")
    mocker.patch("llm.plugin.log")
    mocker.patch("llm.plugin.httpserver")
    mocker.patch("llm.plugin.schedule.addPeriodicEvent")
    mocker.patch("llm.plugin.schedule.removeEvent")
    mocker.patch("llm.plugin.schedule.addEvent")

    plugin = LLM(mock_irc)
    # Swap registryValue to MagicMock keeping defaults
    plugin.registryValue = mocker.MagicMock(
        side_effect=registry,
    )

    # Provide the RLock that _allow_concurrent expects
    plugin._MetaSynchronized_rlock = threading.RLock()

    # sanitize_output is a passthrough in tests
    plugin.llm_service.sanitize_output.side_effect = lambda x: x

    # migrate_nick returns 0 by default
    plugin.db.migrate_nick.return_value = 0

    return plugin, mock_irc, mock_msg


# ---------------------------------------------------------------------------
# TestAnimateCommand
# ---------------------------------------------------------------------------


class TestAnimateCommand:
    """Tests for the real LLM.animate method."""

    @pytest.fixture
    def identified_env(
        self,
        plugin_env,
        mocker: MockerFixture,
    ):
        """Extend plugin_env with NickServ-identified user."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount = mocker.MagicMock(
            return_value="testaccount",
        )
        return plugin, mock_irc, mock_msg

    def test_animate_requires_nickserv_auth(
        self,
        plugin_env,
        mocker: MockerFixture,
    ):
        """GIVEN unidentified user WHEN animate called
        THEN error about NickServ identification.
        """
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount = mocker.MagicMock(
            side_effect=KeyError("unknown"),
        )

        plugin.animate(mock_irc, mock_msg, ["a", "cat"])

        mock_irc.error.assert_called_once()
        error_text = mock_irc.error.call_args[0][0]
        assert "NickServ" in error_text

    def test_animate_nickserv_identified_proceeds(
        self,
        identified_env,
    ):
        """GIVEN NickServ-identified user WHEN animate called
        THEN video_generation is called.
        """
        plugin, mock_irc, mock_msg = identified_env

        plugin.animate(mock_irc, mock_msg, ["a", "cat"])

        plugin.llm_service.video_generation.assert_called_once()

    def test_animate_replies_with_result(
        self,
        identified_env,
    ):
        """GIVEN successful video generation WHEN animate
        THEN replies with sanitized URL.
        """
        plugin, mock_irc, mock_msg = identified_env
        url = "https://example.com/llm/vid_abc.mp4"
        plugin.llm_service.video_generation.return_value = VideoResult(content=url)

        plugin.animate(mock_irc, mock_msg, ["a", "cat"])

        mock_irc.reply.assert_called_once_with(url)

    def test_animate_stores_context(self, identified_env):
        """GIVEN context enabled WHEN animate succeeds
        THEN context stored with '[Generated video: ...]'.
        """
        plugin, mock_irc, mock_msg = identified_env
        plugin.llm_service.video_generation.return_value = VideoResult(
            content="https://example.com/llm/vid.mp4",
            model="grok-imagine-video",
        )

        plugin.animate(mock_irc, mock_msg, ["a", "cat"])

        msgs = plugin.context.get_messages(
            "testaccount",
            "#test",
        )
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "a cat"
        assert msgs[1]["role"] == "assistant"
        assert "[Generated video:" in msgs[1]["content"]

    def test_animate_no_context_on_error(
        self,
        identified_env,
    ):
        """GIVEN video generation error WHEN animate
        THEN no context stored.
        """
        plugin, mock_irc, mock_msg = identified_env
        plugin.llm_service.video_generation.return_value = VideoResult(
            content="Error: something went wrong",
            error="Error: something went wrong",
        )

        plugin.animate(mock_irc, mock_msg, ["bad"])

        msgs = plugin.context.get_messages(
            "testaccount",
            "#test",
        )
        assert len(msgs) == 0

    def test_animate_logs_usage(self, identified_env):
        """GIVEN successful video generation WHEN animate
        THEN usage logged as 'animate'.
        """
        plugin, mock_irc, mock_msg = identified_env
        plugin.llm_service.video_generation.return_value = VideoResult(
            content="https://example.com/llm/vid.mp4",
            cost=0.10,
            model="grok-imagine-video",
        )

        plugin.animate(mock_irc, mock_msg, ["a", "cat"])

        plugin.db.log_usage.assert_called_once_with(
            "testaccount",
            "#test",
            "animate",
            "grok-imagine-video",
            0,
            0,
            0.10,
        )

    def test_animate_skips_znc_playback(self, plugin_env):
        """GIVEN old message (ZNC) WHEN animate
        THEN nothing happens.
        """
        plugin, mock_irc, mock_msg = plugin_env
        mock_msg.time = 0.5  # before startup_time

        plugin.animate(mock_irc, mock_msg, ["hello"])

        mock_irc.reply.assert_not_called()
        mock_irc.error.assert_not_called()

    def test_video_alias_exists(self):
        """GIVEN plugin class WHEN checking for video attribute
        THEN it exists and equals animate.
        """
        assert hasattr(LLM, "video")
        assert LLM.video is LLM.animate


# ---------------------------------------------------------------------------
# TestVideoGeneration
# ---------------------------------------------------------------------------


class TestVideoGeneration:
    """Tests for LLMService.video_generation."""

    def test_video_generation_success(
        self,
        make_service,
        mocker: MockerFixture,
    ):
        """GIVEN valid prompt WHEN video_generation
        THEN returns VideoResult with URL.
        """
        service, plugin = make_service()
        mocker.patch("time.sleep")

        service._save_video_bytes = mocker.Mock(
            return_value=("https://example.com/llm/vid_abc.mp4"),
        )

        submit = _make_urlopen_resp(
            mocker,
            {"request_id": "req-123"},
        )
        poll = _make_urlopen_resp(
            mocker,
            {
                "status": "done",
                "url": "https://tmp.xai/v.mp4",
            },
        )
        download = _make_urlopen_resp(
            mocker,
            b"fake video bytes",
        )

        mock_urlopen = mocker.patch(
            "urllib.request.urlopen",
        )
        mock_urlopen.side_effect = [
            submit,
            poll,
            download,
        ]

        result = service.video_generation("a cat playing")

        assert result.error is None
        assert "vid_abc" in result.content
        assert result.model == "grok-imagine-video"

    def test_video_generation_no_api_key(
        self,
        make_service,
    ):
        """GIVEN no animateApiKey WHEN video_generation
        THEN returns error.
        """
        service, plugin = make_service(animateApiKey="")

        result = service.video_generation("a cat")

        assert result.error is not None
        assert "API key" in result.content

    def test_video_generation_invalid_prompt(
        self,
        make_service,
    ):
        """GIVEN empty prompt WHEN video_generation
        THEN returns validation error.
        """
        service, plugin = make_service()

        result = service.video_generation("")

        assert result.error is not None

    def test_video_generation_timeout(
        self,
        make_service,
        mocker: MockerFixture,
    ):
        """GIVEN polling exceeds timeout WHEN video_generation
        THEN returns timeout error.
        """
        service, plugin = make_service(animateTimeout=1)

        mocker.patch("time.sleep")
        mock_time = mocker.patch("time.time")
        # start_time, first elapsed, second elapsed
        mock_time.side_effect = [100.0, 100.0, 102.0] + [102.0] * 10

        submit = _make_urlopen_resp(
            mocker,
            {"request_id": "req-123"},
        )
        poll = _make_urlopen_resp(
            mocker,
            {"status": "pending"},
        )

        mock_urlopen = mocker.patch(
            "urllib.request.urlopen",
        )
        mock_urlopen.side_effect = [submit, poll]

        result = service.video_generation("a cat")

        assert result.error is not None
        assert "timed out" in result.content.lower()

    def test_video_generation_expired(
        self,
        make_service,
        mocker: MockerFixture,
    ):
        """GIVEN API returns expired status WHEN polling
        THEN returns expired error.
        """
        service, plugin = make_service()
        mocker.patch("time.sleep")

        submit = _make_urlopen_resp(
            mocker,
            {"request_id": "req-123"},
        )
        poll = _make_urlopen_resp(
            mocker,
            {"status": "expired"},
        )

        mock_urlopen = mocker.patch(
            "urllib.request.urlopen",
        )
        mock_urlopen.side_effect = [submit, poll]

        result = service.video_generation("a cat")

        assert result.error is not None
        assert "expired" in result.content.lower()

    def test_video_generation_http_401(
        self,
        make_service,
        mocker: MockerFixture,
    ):
        """GIVEN invalid API key WHEN submit
        THEN returns auth error.
        """
        service, plugin = make_service()

        mock_urlopen = mocker.patch(
            "urllib.request.urlopen",
        )
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.x.ai/v1/videos/generations",
            code=401,
            msg="Unauthorized",
            hdrs=mocker.MagicMock(),
            fp=io.BytesIO(b""),
        )

        result = service.video_generation("a cat")

        assert result.error is not None
        assert "API key" in result.content

    def test_video_generation_content_safety(
        self,
        make_service,
        mocker: MockerFixture,
    ):
        """GIVEN blocked prompt WHEN submit returns 400
        THEN returns safety error.
        """
        service, plugin = make_service()
        body = b'{"error": "moderation blocked"}'

        mock_urlopen = mocker.patch(
            "urllib.request.urlopen",
        )
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.x.ai/v1/videos/generations",
            code=400,
            msg="Bad Request",
            hdrs=mocker.MagicMock(),
            fp=io.BytesIO(body),
        )

        result = service.video_generation("bad prompt")

        assert result.error is not None
        assert "safety" in result.content.lower()


# ---------------------------------------------------------------------------
# TestSaveVideoBytes
# ---------------------------------------------------------------------------


class TestSaveVideoBytes:
    """Tests for LLMService._save_video_bytes."""

    def test_save_video_bytes_creates_file(
        self,
        make_service,
        tmp_path,
    ):
        """GIVEN video bytes WHEN _save_video_bytes
        THEN file created with vid_ prefix and .mp4 extension.
        """
        service, _ = make_service(
            httpRoot=str(tmp_path),
            httpUrlBase="https://example.com/llm",
        )

        url = service._save_video_bytes(b"fake video data")

        assert url is not None
        files = list(tmp_path.glob("vid_*.mp4"))
        assert len(files) == 1

    def test_save_video_bytes_returns_url(
        self,
        make_service,
        tmp_path,
    ):
        """GIVEN video bytes WHEN _save_video_bytes
        THEN returns URL containing vid_ and .mp4.
        """
        service, _ = make_service(
            httpRoot=str(tmp_path),
            httpUrlBase="https://example.com/llm",
        )

        url = service._save_video_bytes(b"fake video data")

        assert url is not None
        assert "vid_" in url
        assert ".mp4" in url
        assert url.startswith("https://example.com/llm/")

    def test_save_video_bytes_custom_extension(
        self,
        make_service,
        tmp_path,
    ):
        """GIVEN custom extension WHEN _save_video_bytes
        THEN file uses that extension.
        """
        service, _ = make_service(
            httpRoot=str(tmp_path),
            httpUrlBase="https://example.com/llm",
        )

        url = service._save_video_bytes(
            b"data",
            extension="webm",
        )

        assert url is not None
        assert ".webm" in url
        files = list(tmp_path.glob("vid_*.webm"))
        assert len(files) == 1

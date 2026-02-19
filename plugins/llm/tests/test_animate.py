"""Tests for the animate (video) command and video generation service."""

from __future__ import annotations

import threading
import time
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


def _mock_response(mocker: MockerFixture, *, json: dict | None = None, status: int = 200):
    """Create a mock requests.Response.

    Args:
        mocker: pytest-mock fixture
        json: JSON body to return from .json()
        status: HTTP status code
    """
    resp = mocker.MagicMock()
    resp.status_code = status
    resp.ok = 200 <= status < 400
    if json is not None:
        resp.json.return_value = json
    resp.raise_for_status.side_effect = None if resp.ok else _make_http_error(mocker, status)
    return resp


def _make_http_error(mocker: MockerFixture, status: int):
    """Create a requests.HTTPError for raise_for_status."""
    import requests

    resp = mocker.MagicMock()
    resp.status_code = status
    resp.text = ""
    return requests.HTTPError(response=resp)


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

    # is_user_flagged returns False by default (user not flagged)
    plugin.db.is_user_flagged.return_value = False

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
            prompt="a cat",
            status="success",
            error_detail="",
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

    def test_animate_wrapper_requires_llm_animate_capability(self):
        """GIVEN plugin source WHEN checking animate wrapper THEN llm.animate capability is required."""
        import inspect

        source = inspect.getsource(LLM)
        assert 'animate = wrap(animate, [("checkCapability", "llm.animate"), "text"])' in source


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

        # Mock time so poll loop doesn't actually sleep 60s
        mock_time_mod = mocker.patch("llm.service.time")
        mock_time_mod.time.side_effect = [100.0, 100.0, 100.0] + [100.0] * 10

        service._download_and_save_video = mocker.Mock(
            return_value="https://example.com/llm/vid_abc.mp4",
        )

        submit_resp = _mock_response(mocker, json={"request_id": "req-123"})
        poll_resp = _mock_response(
            mocker, json={"video": {"url": "https://tmp.xai/v.mp4"}, "model": "grok-imagine-video"}
        )

        mock_session = mocker.MagicMock()
        mock_session.post.return_value = submit_resp
        mock_session.get.return_value = poll_resp
        mocker.patch("requests.Session", return_value=mock_session)

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

        # Patch the time module reference in llm.service's namespace (not globally)
        # so the logging framework's internal time.time() calls are unaffected.
        mock_time_mod = mocker.patch("llm.service.time")
        # start_time, first elapsed, second elapsed
        mock_time_mod.time.side_effect = [100.0, 100.0, 102.0] + [102.0] * 10

        submit_resp = _mock_response(mocker, json={"request_id": "req-123"})
        poll_resp = _mock_response(mocker, json={"status": "pending"})

        mock_session = mocker.MagicMock()
        mock_session.post.return_value = submit_resp
        mock_session.get.return_value = poll_resp
        mocker.patch("requests.Session", return_value=mock_session)

        result = service.video_generation("a cat")

        assert result.error is not None
        assert "timed out" in result.content.lower()

    def test_video_generation_timeout_stashes_to_db(
        self,
        make_service,
        mocker: MockerFixture,
    ):
        """GIVEN polling times out and DB available WHEN video_generation
        THEN stashes request_id to pending_tasks via DB.
        """
        service, plugin = make_service(animateTimeout=1)

        mock_db = mocker.MagicMock()
        mock_db.save_pending_task.return_value = 7
        plugin.db = mock_db

        mock_time_mod = mocker.patch("llm.service.time")
        mock_time_mod.time.side_effect = [100.0, 100.0, 102.0] + [102.0] * 10

        submit_resp = _mock_response(mocker, json={"request_id": "req-456"})
        poll_resp = _mock_response(mocker, json={"status": "pending"})

        mock_session = mocker.MagicMock()
        mock_session.post.return_value = submit_resp
        mock_session.get.return_value = poll_resp
        mocker.patch("requests.Session", return_value=mock_session)

        result = service.video_generation("a cat")

        mock_db.save_pending_task.assert_called_once()
        call_kwargs = mock_db.save_pending_task.call_args[1]
        assert call_kwargs["task_type"] == "animate"
        assert '"request_id"' in call_kwargs["request_data"]
        assert "req-456" in call_kwargs["request_data"]
        assert "retry" in result.content.lower() or "timed out" in result.content.lower()

    def test_video_generation_expired(
        self,
        make_service,
        mocker: MockerFixture,
    ):
        """GIVEN API returns expired status WHEN polling
        THEN returns expired error.
        """
        service, plugin = make_service()

        mock_time_mod = mocker.patch("llm.service.time")
        mock_time_mod.time.side_effect = [100.0, 100.0, 100.0] + [100.0] * 10

        submit_resp = _mock_response(mocker, json={"request_id": "req-123"})
        poll_resp = _mock_response(mocker, json={"status": "expired"})

        mock_session = mocker.MagicMock()
        mock_session.post.return_value = submit_resp
        mock_session.get.return_value = poll_resp
        mocker.patch("requests.Session", return_value=mock_session)

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
        import requests

        service, plugin = make_service()

        error_resp = mocker.MagicMock()
        error_resp.status_code = 401
        error_resp.text = "Unauthorized"

        mock_session = mocker.MagicMock()
        mock_session.post.return_value = error_resp
        error_resp.raise_for_status.side_effect = requests.HTTPError(response=error_resp)
        mocker.patch("requests.Session", return_value=mock_session)

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
        import requests

        service, plugin = make_service()

        error_resp = mocker.MagicMock()
        error_resp.status_code = 400
        error_resp.text = '{"error": "moderation blocked"}'

        mock_session = mocker.MagicMock()
        mock_session.post.return_value = error_resp
        error_resp.raise_for_status.side_effect = requests.HTTPError(response=error_resp)
        mocker.patch("requests.Session", return_value=mock_session)

        result = service.video_generation("bad prompt")

        assert result.error is not None
        assert "safety" in result.content.lower()

    def test_video_generation_persists_immediately_after_submit(
        self,
        make_service,
        mocker: MockerFixture,
    ):
        """GIVEN provider returns request_id WHEN video_generation submits
        THEN persists job to DB before entering poll loop.
        """
        service, plugin = make_service(animateTimeout=1)

        mock_db = mocker.MagicMock()
        mock_db.save_pending_task.return_value = 42
        plugin.db = mock_db

        # Time: submit at 100, first poll at 160, exceeds timeout
        mock_time_mod = mocker.patch("llm.service.time")
        mock_time_mod.time.side_effect = [100.0, 100.0, 102.0] + [102.0] * 10

        submit_resp = _mock_response(mocker, json={"request_id": "req-durable"})
        poll_resp = _mock_response(mocker, json={"status": "pending"})

        mock_session = mocker.MagicMock()
        mock_session.post.return_value = submit_resp
        mock_session.get.return_value = poll_resp
        mocker.patch("requests.Session", return_value=mock_session)

        service.video_generation("a cat")

        # The immediate persist should have been called with next_attempt_at
        # set to submitted_at + timeout (so background doesn't race foreground)
        mock_db.save_pending_task.assert_called_once()
        call_kwargs = mock_db.save_pending_task.call_args[1]
        assert call_kwargs["task_type"] == "animate"
        assert '"request_id"' in call_kwargs["request_data"]
        assert "req-durable" in call_kwargs["request_data"]
        assert "origin_request_id" in call_kwargs
        # next_attempt_at should be submitted_at + timeout, NOT submitted_at
        assert call_kwargs["next_attempt_at"] > call_kwargs["submitted_at"]

    def test_video_generation_deletes_persisted_row_on_success(
        self,
        make_service,
        mocker: MockerFixture,
    ):
        """GIVEN job persisted after submit WHEN foreground poll completes
        THEN deletes the persisted row.
        """
        service, plugin = make_service()

        mock_time_mod = mocker.patch("llm.service.time")
        mock_time_mod.time.side_effect = [100.0, 100.0, 100.0] + [100.0] * 10

        mock_db = mocker.MagicMock()
        mock_db.save_pending_task.return_value = 42
        plugin.db = mock_db

        service._download_and_save_video = mocker.Mock(
            return_value="https://example.com/llm/vid_abc.mp4",
        )

        submit_resp = _mock_response(mocker, json={"request_id": "req-success"})
        poll_resp = _mock_response(
            mocker, json={"video": {"url": "https://tmp.xai/v.mp4"}, "model": "grok-imagine-video"}
        )

        mock_session = mocker.MagicMock()
        mock_session.post.return_value = submit_resp
        mock_session.get.return_value = poll_resp
        mocker.patch("requests.Session", return_value=mock_session)

        result = service.video_generation("a cat playing")

        assert result.error is None
        mock_db.delete_pending_task.assert_called_once_with(42)

    def test_video_generation_deletes_persisted_row_on_terminal_error(
        self,
        make_service,
        mocker: MockerFixture,
    ):
        """GIVEN job persisted after submit WHEN provider returns expired status
        THEN deletes the persisted row.
        """
        service, plugin = make_service()

        mock_time_mod = mocker.patch("llm.service.time")
        mock_time_mod.time.side_effect = [100.0, 100.0, 100.0] + [100.0] * 10

        mock_db = mocker.MagicMock()
        mock_db.save_pending_task.return_value = 42
        plugin.db = mock_db

        submit_resp = _mock_response(mocker, json={"request_id": "req-expired"})
        poll_resp = _mock_response(mocker, json={"status": "expired"})

        mock_session = mocker.MagicMock()
        mock_session.post.return_value = submit_resp
        mock_session.get.return_value = poll_resp
        mocker.patch("requests.Session", return_value=mock_session)

        result = service.video_generation("a cat")

        assert result.error is not None
        mock_db.delete_pending_task.assert_called_once_with(42)

    def test_video_generation_timeout_updates_existing_row(
        self,
        make_service,
        mocker: MockerFixture,
    ):
        """GIVEN job persisted after submit WHEN foreground times out
        THEN updates existing row for immediate retry (not insert duplicate).
        """
        service, plugin = make_service(animateTimeout=1)

        mock_db = mocker.MagicMock()
        mock_db.save_pending_task.return_value = 42
        plugin.db = mock_db

        mock_time_mod = mocker.patch("llm.service.time")
        mock_time_mod.time.side_effect = [100.0, 100.0, 102.0] + [102.0] * 10

        submit_resp = _mock_response(mocker, json={"request_id": "req-timeout"})
        poll_resp = _mock_response(mocker, json={"status": "pending"})

        mock_session = mocker.MagicMock()
        mock_session.post.return_value = submit_resp
        mock_session.get.return_value = poll_resp
        mocker.patch("requests.Session", return_value=mock_session)

        service.video_generation("a cat")

        # Should NOT insert a second row — save_pending_task called only once
        # (at submit time, not again at timeout)
        assert mock_db.save_pending_task.call_count == 1
        # Should release the existing row for immediate background retry
        mock_db.release_pending_task.assert_called_once()
        release_kwargs = mock_db.release_pending_task.call_args
        assert release_kwargs[0][0] == 42  # task_id


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

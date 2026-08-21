"""Tests for @animate — text-to-video against a self-hosted vLLM box.

Animate is the only generation path where the media does not exist when the
command returns. A 4s clip measured 67.7s at 25 steps and 171s at 50 on the
reference box (2026-08-20), so submission and delivery are split: the command
stashes a job id in ``pending_tasks`` and the existing pending-task machinery
polls until the clip lands. Most of what is worth testing here is that seam —
that a submission is either stashed or honestly reported as untracked, and
that polling distinguishes "not yet" from "never".

The other half is the promise made to the user. ``generate_video`` returns an
acknowledgement and never a URL, because a tool that hands back a link to a
video that does not exist yet is exactly the shape of the image-fabrication
bug guarded in test_image_fabrication_guard.py.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from llm.service import LLMService

_URL = "http://video.example.com:14205"
# Deliberately low-entropy and self-describing: a realistic-looking token here
# trips the gitleaks pre-commit hook, and nothing in these tests inspects the
# value beyond "is it set".
_KEY = "not-a-real-token-for-tests"


@pytest.fixture
def animate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured animate deployment: URL in the registry, key in the env."""
    monkeypatch.setenv("ANIMATE_API_KEY", _KEY)


def _service(make_service: Callable[..., tuple[LLMService, Mock]], **overrides: Any):
    overrides.setdefault("animateApiUrl", _URL)
    return make_service(**overrides)


class TestAnimateAvailability:
    """Both halves of the credential are required, and they fail independently."""

    def test_unavailable_without_url(self, make_service, animate_env) -> None:
        """GIVEN a key but no URL WHEN checked THEN animate is unavailable."""
        service, _ = make_service(animateApiUrl="")
        assert service.animate_available() is False

    def test_unavailable_without_key(self, make_service, monkeypatch) -> None:
        """GIVEN a URL but no key WHEN checked THEN animate is unavailable."""
        monkeypatch.delenv("ANIMATE_API_KEY", raising=False)
        service, _ = _service(make_service)
        assert service.animate_available() is False

    def test_available_with_both(self, make_service, animate_env) -> None:
        """GIVEN URL and key WHEN checked THEN animate is available."""
        service, _ = _service(make_service)
        assert service.animate_available() is True

    def test_unsafe_url_disables_rather_than_dials(self, make_service, animate_env) -> None:
        """GIVEN a non-http URL WHEN checked THEN the feature is off.

        A typo in the registry must not become a request to something that is
        not a video server; this request carries a bearer token.
        """
        service, _ = make_service(animateApiUrl="file:///etc/passwd")
        assert service.animate_available() is False
        assert service._animate_base_url() == ""

    def test_trailing_slash_is_normalised(self, make_service, animate_env) -> None:
        """GIVEN a URL with a trailing slash WHEN read THEN paths do not double up."""
        service, _ = make_service(animateApiUrl=_URL + "/")
        assert service._animate_base_url() == _URL


class TestAnimateForm:
    """The submission form carries the config, and the audio switch is real."""

    def test_audio_on_requests_t2va(self, make_service, animate_env) -> None:
        """GIVEN animateAudio on WHEN a form is built THEN the task is t2va."""
        service, _ = _service(make_service, animateAudio=True)
        extra = json.loads(service._animate_form("a forest", None)["extra_params"])
        assert extra["task"] == "t2va"
        assert extra["audio_flow_shift"] == 3

    def test_audio_off_requests_t2v_and_drops_audio_knob(self, make_service, animate_env) -> None:
        """GIVEN animateAudio off WHEN a form is built THEN no audio is requested."""
        service, _ = _service(make_service, animateAudio=False)
        extra = json.loads(service._animate_form("a forest", None)["extra_params"])
        assert extra["task"] == "t2v"
        assert "audio_flow_shift" not in extra

    def test_steps_and_size_come_from_config(self, make_service, animate_env) -> None:
        """GIVEN step/size config WHEN a form is built THEN it carries them."""
        service, _ = _service(make_service, animateSteps=25, animateSize="640x352")
        form = service._animate_form("a forest", None)
        assert form["num_inference_steps"] == "25"
        assert form["size"] == "640x352"

    def test_empty_model_is_omitted(self, make_service, animate_env) -> None:
        """GIVEN no configured model WHEN a form is built THEN no model field is sent.

        A single-model box picks its own; sending "" would be a request for a
        model literally named empty string.
        """
        service, _ = _service(make_service, animateModel="")
        assert "model" not in service._animate_form("a forest", None)

    def test_configured_model_is_sent(self, make_service, animate_env) -> None:
        """GIVEN a configured model WHEN a form is built THEN it is sent."""
        service, _ = _service(make_service, animateModel="/work/models/H3")
        assert service._animate_form("a forest", None)["model"] == "/work/models/H3"

    def test_duration_rides_extra_params(self, make_service, animate_env) -> None:
        """GIVEN a duration WHEN a form is built THEN it is in extra_params."""
        service, _ = _service(make_service, animateDuration=6)
        extra = json.loads(service._animate_form("a forest", None)["extra_params"])
        assert extra["duration"] == 6


class TestVideoGenerationSubmit:
    """Submission returns immediately; what matters is what it promises."""

    def test_unconfigured_returns_error_without_calling_out(
        self, make_service, monkeypatch, mocker
    ) -> None:
        """GIVEN no config WHEN submitting THEN it errors and makes no request."""
        monkeypatch.delenv("ANIMATE_API_KEY", raising=False)
        service, _ = make_service(animateApiUrl="")
        request = mocker.patch.object(service, "_animate_request")

        result = service.video_generation("a forest", reply_target="#chan")

        assert result.error is not None
        assert result.queued is False
        request.assert_not_called()

    def test_successful_submit_stashes_and_promises_delivery(
        self, make_service, animate_env, mocker
    ) -> None:
        """GIVEN a job id WHEN submitting THEN it stashes and promises a later link."""
        service, _ = _service(make_service)
        mocker.patch.object(service, "_animate_request", return_value=(200, {"id": "video_gen_1"}))
        stash = mocker.patch.object(service, "_stash_timeout", return_value=True)

        result = service.video_generation(
            "a forest",
            nick="alice",
            reply_target="#chan",
            is_channel=True,
            account="alice",
        )

        assert result.queued is True
        assert result.job_id == "video_gen_1"
        assert result.error is None
        # The acknowledgement must not contain a URL — there is nothing to link.
        assert "http" not in result.content

        kwargs = stash.call_args.kwargs
        assert kwargs["task_type"] == "animate"
        assert kwargs["reply_target"] == "#chan"
        assert kwargs["request_data"]["job_id"] == "video_gen_1"

    def test_rejected_submit_reports_the_server_reason(
        self, make_service, animate_env, mocker
    ) -> None:
        """GIVEN a 400 WHEN submitting THEN the server's reason reaches the user."""
        service, _ = _service(make_service)
        mocker.patch.object(
            service,
            "_animate_request",
            return_value=(400, {"error": {"message": "unsupported size"}}),
        )
        stash = mocker.patch.object(service, "_stash_timeout")

        result = service.video_generation("a forest", reply_target="#chan")

        assert result.error is not None
        assert "unsupported size" in result.content
        stash.assert_not_called()

    def test_unreachable_server_does_not_raise(self, make_service, animate_env, mocker) -> None:
        """GIVEN a network error WHEN submitting THEN it returns an error result."""
        service, _ = _service(make_service)
        mocker.patch.object(service, "_animate_request", side_effect=OSError("no route"))

        result = service.video_generation("a forest", reply_target="#chan")

        assert result.error is not None
        assert result.queued is False

    def test_failed_stash_does_not_promise_delivery(
        self, make_service, animate_env, mocker
    ) -> None:
        """GIVEN stashing fails WHEN submitting THEN it says so rather than promising.

        The job is running on the box either way; the lie worth avoiding is
        telling someone a clip is coming when nothing will ever collect it.
        """
        service, _ = _service(make_service)
        mocker.patch.object(service, "_animate_request", return_value=(200, {"id": "video_gen_1"}))
        mocker.patch.object(service, "_stash_timeout", return_value=False)

        result = service.video_generation("a forest", reply_target="#chan")

        assert result.error is not None
        assert result.job_id == "video_gen_1"
        assert result.queued is False

    def test_no_reply_target_skips_stashing(self, make_service, animate_env, mocker) -> None:
        """GIVEN no delivery target WHEN submitting THEN nothing is stashed.

        A stashed row with an empty target emits a malformed PRIVMSG on every
        delivery attempt — the trap _stash_timeout documents for completions.
        """
        service, _ = _service(make_service)
        mocker.patch.object(service, "_animate_request", return_value=(200, {"id": "video_gen_1"}))
        stash = mocker.patch.object(service, "_stash_timeout")

        result = service.video_generation("a forest", reply_target="")

        assert result.queued is True
        stash.assert_not_called()


def _task(**overrides: Any) -> Mock:
    task = Mock()
    task.task_type = "animate"
    task.nick = "alice"
    task.reply_target = "#chan"
    task.is_channel = True
    task.prompt_preview = "a forest"
    task.model = ""
    for key, value in overrides.items():
        setattr(task, key, value)
    return task


class TestRetryVideo:
    """Polling must tell "not yet" apart from "never", and never resubmit."""

    def test_still_rendering_raises_for_backoff(self, make_service, animate_env, mocker) -> None:
        """GIVEN an in-progress job WHEN polled THEN it raises to be retried."""
        import litellm

        service, _ = _service(make_service)
        mocker.patch.object(
            service, "_animate_request", return_value=(200, {"status": "in_progress"})
        )

        with pytest.raises(litellm.Timeout):
            service._retry_video(_task(), {"job_id": "video_gen_1"})

    def test_completed_job_is_downloaded_and_published(
        self, make_service, animate_env, mocker
    ) -> None:
        """GIVEN a completed job WHEN polled THEN the clip is published."""
        service, _ = _service(make_service)
        mocker.patch.object(
            service,
            "_animate_request",
            side_effect=[(200, {"status": "completed"}), (200, b"\x00\x00\x00 ftypmp42")],
        )
        save = mocker.patch.object(
            service, "_save_video_bytes", return_value="https://paste.example/img/vid_a.mp4"
        )

        result = service._retry_video(_task(), {"job_id": "video_gen_1"})

        assert result.status == "completed"
        assert result.content == "https://paste.example/img/vid_a.mp4"
        assert save.call_args.args[0] == b"\x00\x00\x00 ftypmp42"

    def test_missing_job_id_is_terminal(self, make_service, animate_env) -> None:
        """GIVEN malformed request data WHEN polled THEN it fails terminally."""
        service, _ = _service(make_service)
        result = service._retry_video(_task(), {})
        assert result.status == "failed_terminal"

    def test_vanished_job_is_terminal(self, make_service, animate_env, mocker) -> None:
        """GIVEN a 404 WHEN polled THEN it fails terminally rather than retrying.

        The server restarted or reaped the job. More polling cannot bring it
        back, so retrying until expiry only delays telling the user.
        """
        service, _ = _service(make_service)
        mocker.patch.object(
            service, "_animate_request", return_value=(404, {"error": {"message": "not found"}})
        )

        result = service._retry_video(_task(), {"job_id": "gone"})

        assert result.status == "failed_terminal"

    def test_failed_job_reports_its_reason(self, make_service, animate_env, mocker) -> None:
        """GIVEN a failed job WHEN polled THEN the server's reason is carried out."""
        service, _ = _service(make_service)
        mocker.patch.object(
            service,
            "_animate_request",
            return_value=(200, {"status": "failed", "error": {"message": "OOM on device"}}),
        )

        result = service._retry_video(_task(), {"job_id": "video_gen_1"})

        assert result.status == "failed_terminal"
        assert "OOM on device" in result.reason

    def test_transient_server_error_raises_rather_than_failing(
        self, make_service, animate_env, mocker
    ) -> None:
        """GIVEN a 503 WHEN polled THEN it raises so the job is retried.

        A bounced box must not lose a job that is very likely still running.
        """
        import litellm

        service, _ = _service(make_service)
        mocker.patch.object(service, "_animate_request", return_value=(503, {"detail": "busy"}))

        with pytest.raises(litellm.Timeout):
            service._retry_video(_task(), {"job_id": "video_gen_1"})

    def test_download_failure_is_terminal(self, make_service, animate_env, mocker) -> None:
        """GIVEN a completed job with an empty download WHEN polled THEN it fails."""
        service, _ = _service(make_service)
        mocker.patch.object(
            service,
            "_animate_request",
            side_effect=[(200, {"status": "completed"}), (200, b"")],
        )

        result = service._retry_video(_task(), {"job_id": "video_gen_1"})

        assert result.status == "failed_terminal"

    def test_publish_failure_is_terminal(self, make_service, animate_env, mocker) -> None:
        """GIVEN a clip that cannot be published WHEN polled THEN it fails."""
        service, _ = _service(make_service)
        mocker.patch.object(
            service,
            "_animate_request",
            side_effect=[(200, {"status": "completed"}), (200, b"mp4bytes")],
        )
        mocker.patch.object(service, "_save_video_bytes", return_value=None)

        result = service._retry_video(_task(), {"job_id": "video_gen_1"})

        assert result.status == "failed_terminal"

    def test_deconfigured_animate_is_terminal(self, make_service, monkeypatch) -> None:
        """GIVEN animate turned off WHEN a stashed job is polled THEN it fails cleanly."""
        monkeypatch.delenv("ANIMATE_API_KEY", raising=False)
        service, _ = make_service(animateApiUrl="")

        result = service._retry_video(_task(), {"job_id": "video_gen_1"})

        assert result.status == "failed_terminal"


class TestPollCadence:
    """Animate polls a running job; it does not retry a failing one."""

    def test_animate_backoff_is_flat(self, make_service) -> None:
        """GIVEN repeated animate polls WHEN backoff is computed THEN it stays flat.

        Every poll before the clip lands raises Timeout, so an exponential
        curve here is applied to the expected case rather than a fault. In
        prod on 2026-08-21 that put a 135s render on a 30/60/120 ladder and
        delivered it at 210s — over a minute of dead air after the video was
        already sitting on the box. The value is also the poll cadence, since
        the plugin arms its next wakeup from it, so it is what the user waits
        past the render finishing.
        """
        service, _ = _service(make_service)
        delays = [service._compute_backoff(n, "animate") for n in range(5)]
        assert delays == [10, 10, 10, 10, 10]

    def test_other_task_types_keep_exponential_backoff(self, make_service) -> None:
        """GIVEN a draw retry WHEN backoff is computed THEN it still doubles.

        There a repeat means the provider call keeps failing, which is exactly
        when backing off is the right move.
        """
        service, _ = _service(make_service)
        delays = [service._compute_backoff(n, "draw") for n in range(4)]
        assert delays == [30, 60, 120, 240]

    def test_backoff_is_capped_for_retry_paths(self, make_service) -> None:
        """GIVEN many failed attempts WHEN backoff is computed THEN it caps."""
        service, _ = _service(make_service)
        assert service._compute_backoff(20, "ask") == 300


class TestVideoUpload:
    """Video rides the image uploader, but the two are not interchangeable."""

    def test_mp4_upload_accepts_an_mp4_reply(self, make_service, mocker) -> None:
        """GIVEN an mp4 upload WHEN the host returns vid_*.mp4 THEN the URL is used."""
        service, _ = make_service(imageUploadUrl="https://paste.example/img/")
        mocker.patch.object(
            service,
            "_upload_image_bytes",
            return_value="https://paste.example/img/vid_a.mp4",
        )
        assert service._save_video_bytes(b"mp4bytes") == "https://paste.example/img/vid_a.mp4"

    def test_image_upload_rejects_a_video_reply(self, make_service, mocker) -> None:
        """GIVEN a png upload WHEN the host returns an .mp4 THEN it is not trusted.

        Images may legitimately come back under a different image extension
        (the host recompresses), but a video URL is never a valid answer to an
        image POST.
        """
        service, plugin = make_service(imageUploadUrl="https://paste.example/img/")
        payload = json.dumps(
            {"results": [{"success": True, "filePath": "/img/vid_a.mp4"}]}
        ).encode()

        opener = mocker.MagicMock()
        response = opener.open.return_value.__enter__.return_value
        response.read.return_value = payload
        mocker.patch("urllib.request.build_opener", return_value=opener)

        assert service._upload_image_bytes(b"\x89PNG", "png") is None

    def test_local_fallback_when_upload_is_off(self, make_service, tmp_path, mocker) -> None:
        """GIVEN no upload host WHEN publishing THEN the clip is written locally."""
        service, _ = make_service(
            imageUploadUrl="", httpRoot=str(tmp_path), httpUrlBase="https://bot.example/llm"
        )
        mocker.patch.object(service, "_upload_image_bytes", return_value=None)

        url = service._save_video_bytes(b"mp4bytes")

        assert url is not None
        assert url.endswith(".mp4")
        written = list(tmp_path.glob("vid_*.mp4"))
        assert len(written) == 1
        assert written[0].read_bytes() == b"mp4bytes"


class TestGenerateVideoTool:
    """The tool queues work; it must never hand back a link."""

    def _assistant(self, animate_fn):
        from llm.assistant import AssistantToolExecutor

        return AssistantToolExecutor(
            db=Mock(),
            context=Mock(),
            nick="alice",
            channel="#chan",
            animate_fn=animate_fn,
        )

    def test_tool_returns_the_acknowledgement(self) -> None:
        """GIVEN a queued job WHEN the tool runs THEN it returns the acknowledgement."""
        from llm.assistant import ToolCallbackResult

        assistant = self._assistant(lambda p: ToolCallbackResult(True, "Rendering your video."))
        out = assistant._tool_generate_video({"prompt": "a forest"})
        assert "Rendering your video." in out
        assert "http" not in out

    def test_tool_reports_failure(self) -> None:
        """GIVEN a rejected submission WHEN the tool runs THEN it reports the error."""
        from llm.assistant import ToolCallbackResult

        assistant = self._assistant(lambda p: ToolCallbackResult(False, "server said no"))
        out = assistant._tool_generate_video({"prompt": "a forest"})
        assert "server said no" in out

    def test_tool_requires_a_prompt(self) -> None:
        """GIVEN an empty prompt WHEN the tool runs THEN it errors without submitting."""
        called = []
        assistant = self._assistant(lambda p: called.append(p))
        assistant._tool_generate_video({"prompt": "   "})
        assert called == []

    def test_tool_unavailable_without_callback(self) -> None:
        """GIVEN no animate callback WHEN the tool runs THEN it says so."""
        assistant = self._assistant(None)
        out = assistant._tool_generate_video({"prompt": "a forest"})
        assert "not available" in out.lower()

    def test_tool_is_chat_only(self) -> None:
        """GIVEN the tool spec WHEN checked THEN it is not on unattended routes.

        A video is ~70s of exclusive GPU time, so it stays on the route where
        a human just asked for one — not verse, which narrates ambiently, and
        not remind_action, which fires with nobody present.
        """
        from llm.assistant import get_tools_for_profile
        from llm.profile import PROFILE_CHAT, PROFILE_REMIND_ACTION, PROFILE_VERSE

        def names(profile: str) -> set[str]:
            return {t["function"]["name"] for t in get_tools_for_profile(profile)}

        assert "generate_video" in names(PROFILE_CHAT)
        assert "generate_video" not in names(PROFILE_VERSE)
        assert "generate_video" not in names(PROFILE_REMIND_ACTION)

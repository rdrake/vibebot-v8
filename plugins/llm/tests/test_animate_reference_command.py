"""@animate with an image URL in the line — the command and chat entry points.

The rewrite that makes text-only @animate work is the wrong move once a
reference image is attached: the picture already fixes the subject, so a
planner turn can only drift from it. With a URL in the line the command skips
the planner entirely and sends what the user typed.

The two entry points both matter. "@animate <url> ..." is the command;
"vibebot animate <url> ..." never reaches it and arrives as chat, which is the
trap documented in service.py's EXPLICIT_VIDEO notes — so the tool callback
has to pick the reference out of the user's message too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass

_IMG = "https://pics.example.com/cat.png"


def _video_result(mocker, content: str = "Rendering your video — I'll post the link here."):
    from llm.service import VideoResult

    return VideoResult(content=content, job_id="video_gen_1", queued=True, model="")


@pytest.fixture
def animate_plugin(plugin_env, mocker):
    """A plugin whose account check passes and whose video path is stubbed."""
    from llm.service import ReferenceImage

    plugin, mock_irc, mock_msg = plugin_env
    mock_irc.state.nickToAccount.return_value = "test_account"
    plugin.llm_service.assistant_request.side_effect = None
    plugin.llm_service.video_generation.return_value = _video_result(mocker)
    # No split_reference_url stub: the plugin calls the real module-level
    # parser, so these tests exercise the same URL handling production does.
    plugin.llm_service.fetch_reference_image.return_value = ReferenceImage(
        data=b"\xff\xd8\xffcat", extension="jpg"
    )
    mocker.patch.object(plugin, "_verse_context_for", return_value=None)
    return plugin, mock_irc, mock_msg


class TestAnimateCommandWithReference:
    """A URL in the line means: use the picture, and do not reword the ask."""

    def test_reference_skips_the_planner(self, animate_plugin) -> None:
        """GIVEN a URL in the prompt WHEN @animate runs THEN no planner turn happens."""
        plugin, mock_irc, mock_msg = animate_plugin

        plugin.animate(mock_irc, mock_msg, [_IMG, "make", "the", "cat", "dance"])

        plugin.llm_service.assistant_request.assert_not_called()
        plugin.llm_service.video_generation.assert_called_once()

    def test_users_words_reach_the_box_verbatim(self, animate_plugin) -> None:
        """GIVEN a URL and words WHEN it submits THEN the words are unrewritten."""
        plugin, mock_irc, mock_msg = animate_plugin

        plugin.animate(mock_irc, mock_msg, [_IMG, "make", "the", "cat", "dance"])

        call = plugin.llm_service.video_generation.call_args
        assert call.args[0] == "make the cat dance"
        assert call.kwargs["reference"].data == b"\xff\xd8\xffcat"

    def test_url_only_line_still_animates(self, animate_plugin) -> None:
        """GIVEN only a URL WHEN it submits THEN a default motion prompt is used.

        The server requires a prompt, and "animate this" with no words is a
        request people will make.
        """
        plugin, mock_irc, mock_msg = animate_plugin

        plugin.animate(mock_irc, mock_msg, [_IMG])

        prompt = plugin.llm_service.video_generation.call_args.args[0]
        assert prompt.strip()
        assert "http" not in prompt

    def test_unfetchable_reference_spends_no_gpu(self, animate_plugin) -> None:
        """GIVEN a reference that will not fetch WHEN @animate runs THEN it says so.

        Rendering the text alone would silently ignore the picture the user
        chose, which reads as the bot ignoring them.
        """
        plugin, mock_irc, mock_msg = animate_plugin
        plugin.llm_service.fetch_reference_image.return_value = None

        plugin.animate(mock_irc, mock_msg, [_IMG, "make", "it", "dance"])

        plugin.llm_service.video_generation.assert_not_called()
        plugin.llm_service.assistant_request.assert_not_called()
        said = " ".join(
            str(c) for c in mock_irc.error.call_args_list + mock_irc.reply.call_args_list
        )
        assert "image" in said.lower()

    def test_reference_turn_is_still_booked(self, animate_plugin, mocker) -> None:
        """GIVEN a reference render WHEN it is submitted THEN a usage row is written.

        No planner means no tokens, but the request still happened and the
        animate row is the only record that it did.
        """
        plugin, mock_irc, mock_msg = animate_plugin
        logged = mocker.patch.object(plugin, "_store_context_and_log_usage")

        plugin.animate(mock_irc, mock_msg, [_IMG, "make", "it", "dance"])

        assert logged.call_args.args[2] == "animate"
        assert logged.call_args.args[5].job_id == "video_gen_1"

    def test_plain_prompt_still_plans(self, animate_plugin) -> None:
        """GIVEN no URL WHEN @animate runs THEN the planner path is untouched."""
        plugin, mock_irc, mock_msg = animate_plugin

        plugin.animate(mock_irc, mock_msg, ["a", "pine", "forest"])

        plugin.llm_service.assistant_request.assert_called_once()
        plugin.llm_service.video_generation.assert_not_called()


class TestChatVideoWithReference:
    """ "vibebot animate <url> ..." is chat, and must still use the picture."""

    def test_tool_callback_uses_the_message_image(self, animate_plugin) -> None:
        """GIVEN a URL in the user's line WHEN the tool fires THEN it conditions the clip."""
        plugin, mock_irc, mock_msg = animate_plugin
        mock_msg.args = ("#test", f"vibebot animate {_IMG} make the cat dance")

        plugin._animate_for_assistant(
            mock_irc,
            mock_msg,
            "a cat dancing on a table",
            nick="test_account",
            channel="#test",
            account="test_account",
        )

        assert plugin.llm_service.video_generation.call_args.kwargs["reference"] is not None

    def test_url_never_reaches_the_video_prompt(self, animate_plugin) -> None:
        """GIVEN a model prompt containing a URL WHEN the tool fires THEN it is stripped.

        The video model renders stray text on screen, so a URL left in the
        prompt becomes part of the picture.
        """
        plugin, mock_irc, mock_msg = animate_plugin
        mock_msg.args = ("#test", f"vibebot animate {_IMG} make the cat dance")

        plugin._animate_for_assistant(
            mock_irc,
            mock_msg,
            f"{_IMG} a cat dancing",
            nick="test_account",
            channel="#test",
            account="test_account",
        )

        assert "http" not in plugin.llm_service.video_generation.call_args.args[0]

    def test_plain_chat_request_is_unchanged(self, animate_plugin) -> None:
        """GIVEN no URL anywhere WHEN the tool fires THEN nothing is fetched."""
        plugin, mock_irc, mock_msg = animate_plugin
        mock_msg.args = ("#test", "vibebot make me a video of a corgi")

        plugin._animate_for_assistant(
            mock_irc,
            mock_msg,
            "a corgi riding a unicorn",
            nick="test_account",
            channel="#test",
            account="test_account",
        )

        plugin.llm_service.fetch_reference_image.assert_not_called()
        assert plugin.llm_service.video_generation.call_args.kwargs.get("reference") is None

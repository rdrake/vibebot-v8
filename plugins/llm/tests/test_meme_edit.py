"""@meme --draw: the captioned meme goes through xAI's image edit.

The captions are already pixels when the edit runs, so the text survives
whatever the model does to the picture, and a refusal costs the picture, not
the meme. Every outcome is booked: xAI bills refusals too.
"""

from __future__ import annotations

import base64
import io
import json
import urllib.error
from typing import TYPE_CHECKING

import pytest
from llm import meme
from llm.service import ImageResult, LLMService

if TYPE_CHECKING:
    pass

CAPTIONED = "https://paste.boxlabs.uk/img/img_captioned.png"
EDITED = "https://paste.boxlabs.uk/img/img_edited.png"
PNG = b"\x89PNG\r\n\x1a\nedited"


@pytest.fixture
def service(make_service, monkeypatch) -> LLMService:  # type: ignore[no-untyped-def]
    svc, _plugin = make_service(memeEditModel="xai/grok-imagine-image", drawTimeout=30)
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    svc._save_image_bytes = lambda data, ext="png": EDITED if data == PNG else None  # type: ignore[method-assign]
    return svc


def _response(payload: dict) -> io.BytesIO:
    body = io.BytesIO(json.dumps(payload).encode())
    body.__enter__ = lambda self=body: self  # type: ignore[attr-defined]
    body.__exit__ = lambda self=body, *a: None  # type: ignore[attr-defined]
    return body


class TestImageEdit:
    def test_success_hosts_the_result_and_prices_from_ticks(self, service, mocker) -> None:
        opened = mocker.patch(
            "urllib.request.urlopen",
            return_value=_response(
                {
                    "data": [{"b64_json": base64.b64encode(PNG).decode()}],
                    "usage": {"cost_in_usd_ticks": 220000000},
                }
            ),
        )

        result = service.image_edit(CAPTIONED, "a tired dad in a lawn chair", channel="#c")

        assert result.url == EDITED and result.error is None
        assert result.cost == pytest.approx(0.022)
        request = opened.call_args.args[0]
        body = json.loads(request.data)
        assert request.full_url == "https://api.x.ai/v1/images/edits"
        assert body["model"] == "grok-imagine-image"
        assert body["image"] == {"url": CAPTIONED}
        assert "a tired dad in a lawn chair" in body["prompt"]
        assert "Keep every letter" in body["prompt"]
        assert request.get_header("Authorization") == "Bearer xai-test"

    def test_refusal_is_an_error_and_still_billed(self, service, mocker) -> None:
        err = urllib.error.HTTPError(
            "https://api.x.ai/v1/images/edits",
            400,
            "Bad Request",
            {},  # type: ignore[arg-type]
            io.BytesIO(
                b'{"error": "content moderation", "usage": {"cost_in_usd_ticks": 200000000}}'
            ),
        )
        mocker.patch("urllib.request.urlopen", side_effect=err)

        result = service.image_edit(CAPTIONED, "something refused")

        assert result.url is None and "refused" in result.error
        assert result.cost == pytest.approx(0.02)

    def test_network_failure_is_free(self, service, mocker) -> None:
        mocker.patch("urllib.request.urlopen", side_effect=OSError("boom"))

        result = service.image_edit(CAPTIONED, "x")

        assert result.error == "Image edit failed." and result.cost == 0.0

    def test_non_xai_model_is_refused_before_any_call(self, make_service, mocker) -> None:
        svc, _ = make_service(memeEditModel="gemini/gemini-2.5-flash-image")
        opened = mocker.patch("urllib.request.urlopen")

        result = svc.image_edit(CAPTIONED, "x")

        assert "xai/" in result.error
        opened.assert_not_called()


_TEMPLATES = meme.parse_templates(
    [
        {"id": "spirit", "name": "Fake Spirit Halloween Costume", "lines": 5, "keywords": []},
        {"id": "drake", "name": "Drakeposting", "lines": 2, "keywords": []},
    ]
)


@pytest.fixture
def meme_plugin(plugin_env, mocker):
    from llm.service import MemePick

    plugin, mock_irc, mock_msg = plugin_env
    mock_irc.state.nickToAccount.return_value = "test_account"
    mocker.patch.object(
        meme.CachedCatalog, "get", return_value=meme.MemeCatalog(_TEMPLATES), autospec=True
    )
    plugin.llm_service._download_and_save_image.return_value = CAPTIONED
    plugin.llm_service.image_edit.return_value = ImageResult(
        content=EDITED, cost=0.022, model="xai/grok-imagine-image", url=EDITED
    )
    plugin.llm_service.meme_pick.return_value = MemePick(
        '{"template": null, "reason": "Nothing fits."}', "test-model"
    )
    mocker.patch.object(meme, "canonical_url", side_effect=lambda url, **_kw: url)
    return plugin, mock_irc, mock_msg


class TestDrawFlag:
    def test_draw_edits_the_captioned_meme(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin

        plugin.meme(
            mock_irc,
            mock_msg,
            [
                "--draw",
                "a tired dad asleep in a lawn chair",
                "spirit | My Dad | Includes: | - Nothing",
            ],
        )

        args, kwargs = plugin.llm_service.image_edit.call_args
        assert args == (CAPTIONED, "a tired dad asleep in a lawn chair")
        assert mock_irc.reply.call_args.args[0] == EDITED
        rows = [(c.args[3], c.args[6]) for c in plugin.db.log_usage.call_args_list]
        assert rows == [("memegen", 0.0), ("xai/grok-imagine-image", 0.022)]

    def test_refused_edit_keeps_the_captioned_meme(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        plugin.llm_service.image_edit.return_value = ImageResult(
            content="",
            cost=0.02,
            model="xai/grok-imagine-image",
            error="The image model refused that edit.",
        )

        plugin.meme(mock_irc, mock_msg, ["--draw", "x", "drake | a | b"])

        reply = mock_irc.reply.call_args.args[0]
        assert reply.startswith(CAPTIONED) and "picture not added" in reply and "refused" in reply
        assert plugin.db.log_usage.call_args.kwargs["status"] == "error"

    def test_no_draw_means_no_edit(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin

        plugin.meme(mock_irc, mock_msg, ["drake | a | b"])

        plugin.llm_service.image_edit.assert_not_called()

    def test_picker_draw_reaches_the_editor(self, meme_plugin) -> None:
        from llm.service import MemePick

        plugin, mock_irc, mock_msg = meme_plugin
        plugin.llm_service.meme_pick.return_value = MemePick(
            '{"template": "spirit", "lines": ["My Dad", "Includes:", "- Nothing"], '
            '"draw": "a tired dad asleep in a lawn chair"}',
            "test-model",
        )

        plugin.meme(mock_irc, mock_msg, ["spirit halloween costume of my dad"])

        assert plugin.llm_service.image_edit.call_args.args == (
            CAPTIONED,
            "a tired dad asleep in a lawn chair",
        )
        assert mock_irc.reply.call_args.args[0].startswith(EDITED)

    def test_tool_draw_parameter(self, meme_plugin) -> None:
        plugin, _, mock_msg = meme_plugin
        _, handlers = plugin._build_meme_tool(mock_msg)

        payload = json.loads(
            handlers["make_meme"](
                {"template": "drake", "lines": ["a", "b"], "draw": "make drake a sysadmin"}
            ).content
        )

        assert plugin.llm_service.image_edit.call_args.args == (CAPTIONED, "make drake a sysadmin")
        assert payload == {"status": "ok", "message": EDITED}

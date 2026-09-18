"""@meme: the user names the template and the captions, nothing is planned.

No LLM turn, no image model, no cost row. The command resolves the name
through the catalog, asks memegen for the PNG through the existing
rehost-a-provider-URL path, and posts the link.
"""

from __future__ import annotations

import json

import pytest
from llm import meme

from .conftest import make_registry_side_effect

_TEMPLATES = meme.parse_templates(
    [
        {
            "id": "drake",
            "name": "Drakeposting",
            "lines": 2,
            "keywords": [],
            "example": {"text": ["left on unread", "left on read"]},
        },
        {
            "id": "db",
            "name": "Distracted Boyfriend",
            "lines": 3,
            "keywords": ["girlfriend"],
            "example": {"text": ["me", "new", "old"]},
        },
    ]
)

_HOSTED = "https://paste.boxlabs.uk/img/img_abc.png"


@pytest.fixture
def meme_plugin(plugin_env, mocker):
    plugin, mock_irc, mock_msg = plugin_env
    mock_irc.state.nickToAccount.return_value = "test_account"
    mocker.patch.object(
        meme.CachedCatalog, "get", return_value=meme.MemeCatalog(_TEMPLATES), autospec=True
    )
    plugin.llm_service._download_and_save_image.return_value = _HOSTED
    return plugin, mock_irc, mock_msg


class TestMemeCommand:
    def test_posts_the_rehosted_image(self, meme_plugin) -> None:
        plugin, mock_irc, _ = meme_plugin

        plugin.meme(mock_irc, meme_plugin[2], ["drake | left on unread | left on read"])

        fetched = plugin.llm_service._download_and_save_image.call_args.args[0]
        assert fetched == "https://api.memegen.link/images/drake/left_on_unread/left_on_read.png"
        assert mock_irc.reply.call_args.args[0] == _HOSTED

    def test_unknown_template_suggests_and_fetches_nothing(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin

        plugin.meme(mock_irc, mock_msg, ["drake boyfriend | a | b"])

        plugin.llm_service._download_and_save_image.assert_not_called()
        err = mock_irc.error.call_args.args[0]
        assert "drake boyfriend" in err and "db" in err

    def test_missing_captions_show_the_example(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin

        plugin.meme(mock_irc, mock_msg, ["drake"])

        err = mock_irc.error.call_args.args[0]
        assert "2 captions" in err and "left on unread" in err

    def test_fetch_failure_is_reported(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        plugin.llm_service._download_and_save_image.return_value = None

        plugin.meme(mock_irc, mock_msg, ["drake | a | b"])

        mock_irc.reply.assert_not_called()
        assert "memegen" in mock_irc.error.call_args.args[0]

    def test_list_filters_by_word(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin

        plugin.meme(mock_irc, mock_msg, ["list boyfriend"])

        reply = mock_irc.reply.call_args.args[0]
        assert "db" in reply and "Distracted Boyfriend" in reply and "3" in reply
        assert "drake" not in reply

    def test_catalog_unavailable_is_an_error(self, meme_plugin, mocker) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        meme.CachedCatalog.get.return_value = None

        plugin.meme(mock_irc, mock_msg, ["drake | a | b"])

        assert "unavailable" in mock_irc.error.call_args.args[0]

    def test_requires_an_account(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        mock_irc.state.nickToAccount.return_value = None

        plugin.meme(mock_irc, mock_msg, ["drake | a | b"])

        plugin.llm_service._download_and_save_image.assert_not_called()


class TestMakeMemeTool:
    """The chat tool: grok transcribes the name the user said, code resolves it."""

    def test_handler_returns_the_hosted_url_as_message(self, meme_plugin) -> None:
        plugin, _, _ = meme_plugin
        schemas, handlers = plugin._build_meme_tool()

        payload = json.loads(
            handlers["make_meme"]({"template": "drake", "lines": ["a", "b"]}).content
        )

        assert payload == {"status": "ok", "message": _HOSTED}
        assert schemas[0]["function"]["name"] == "make_meme"

    def test_handler_error_carries_the_suggestions(self, meme_plugin) -> None:
        plugin, _, _ = meme_plugin
        _, handlers = plugin._build_meme_tool()

        payload = json.loads(
            handlers["make_meme"]({"template": "drake boyfriend", "lines": ["a"]}).content
        )

        assert "drake" in payload["error"] and "db" in payload["error"]
        plugin.llm_service._download_and_save_image.assert_not_called()

    def test_handler_tolerates_junk_arguments(self, meme_plugin) -> None:
        plugin, _, _ = meme_plugin
        _, handlers = plugin._build_meme_tool()

        payload = json.loads(handlers["make_meme"]({"template": 7, "lines": "a|b"}).content)

        assert "error" in payload

    def test_schema_tells_the_model_not_to_choose(self, meme_plugin) -> None:
        plugin, _, _ = meme_plugin
        schemas, _ = plugin._build_meme_tool()
        desc = schemas[0]["function"]["parameters"]["properties"]["template"]["description"]
        assert "as the user" in desc


class TestChatWiring:
    def _result(self):
        from llm.service import AssistantResult

        return AssistantResult(
            content="ok",
            grounding_used=False,
            prompt_tokens=1,
            completion_tokens=1,
            cost=0.0,
            model="m",
        )

    def test_ask_advertises_make_meme_when_enabled(self, meme_plugin) -> None:
        plugin, irc, msg = meme_plugin
        plugin.registryValue.side_effect = make_registry_side_effect(
            {"memeEnabled": True, "ircLookupEnabled": False}
        )
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._result()

        plugin.ask(irc, msg, ["make a drake meme"])

        kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        names = [t["function"]["name"] for t in kwargs["extra_tools"]]
        assert names == ["make_meme"]
        assert "make_meme" in kwargs["extra_handlers"]

    def test_ask_omits_make_meme_when_disabled(self, meme_plugin) -> None:
        plugin, irc, msg = meme_plugin
        plugin.registryValue.side_effect = make_registry_side_effect(
            {"memeEnabled": False, "ircLookupEnabled": False}
        )
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._result()

        plugin.ask(irc, msg, ["hello"])

        assert plugin.llm_service.assistant_request.call_args.kwargs["extra_tools"] is None


class TestCustomAndAliases:
    def test_image_url_template_fetches_the_custom_render(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin

        plugin.meme(mock_irc, mock_msg, ["https://i.imgflip.com/1c1uej.jpg | top | bottom"])

        fetched = plugin.llm_service._download_and_save_image.call_args.args[0]
        assert fetched.startswith(
            "https://api.memegen.link/images/custom/top/bottom.png?background="
        )
        assert mock_irc.reply.call_args.args[0] == _HOSTED

    def test_registry_aliases_reach_the_resolver(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        plugin.registryValue.side_effect = make_registry_side_effect(
            {"memeAliases": ["hotline=drake"]}
        )

        plugin.meme(mock_irc, mock_msg, ["hotline bling | a | b"])

        fetched = plugin.llm_service._download_and_save_image.call_args.args[0]
        assert fetched == "https://api.memegen.link/images/drake/a/b.png"

    def test_tool_accepts_an_image_url_as_template(self, meme_plugin) -> None:
        plugin, _, _ = meme_plugin
        _, handlers = plugin._build_meme_tool()

        payload = json.loads(
            handlers["make_meme"](
                {"template": "https://i.imgflip.com/1c1uej.jpg", "lines": ["a", "b"]}
            ).content
        )

        assert payload["status"] == "ok"

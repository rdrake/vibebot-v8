"""Tests for the verse storybook tool."""

from __future__ import annotations


def test_storybook_config_defaults():
    import llm.config as cfg

    group = cfg.LLM
    assert group.verseStorybookEnabled is not None
    assert int(group.verseStorybookMaxImages()) == 3
    assert int(group.verseStorybookMaxPerTurn()) == 1
    assert int(group.verseStorybookCooldownSeconds()) == 300
    assert int(group.verseStorybookDailyImageCap()) == 30
    assert int(group.verseStorybookMaxChars()) == 6000
    assert int(group.verseStorybookImageTimeout()) == 45


def test_image_result_has_url_field():
    from llm.service import ImageResult

    r = ImageResult(content="msg", url="https://h/llm/img_a.jpg")
    assert r.url == "https://h/llm/img_a.jpg"
    assert ImageResult(content="x").url is None  # default


def test_extract_json_object():
    from llm.service import _extract_json_object as ex

    assert ex('{"a": 1}') == {"a": 1}
    assert ex('Here ye go! ```json\n{"a": 2}\n``` enjoy') == {"a": 2}
    assert ex('prose {"a": {"b": 3}} more prose') == {"a": {"b": 3}}
    assert ex("not json at all") is None
    assert ex("") is None

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


def test_embed_illustrations_basic():
    from llm.service import LLMService as S

    md = "Intro [[illustration:1]] middle [[illustration:11]] end"
    illos = {1: ("cat", "u1.jpg"), 11: ("dog", "u11.jpg")}
    out, used = S._embed_illustrations(md, illos)
    assert "![cat](u1.jpg)" in out and "*cat*" in out
    assert "![dog](u11.jpg)" in out
    assert used == {1, 11}


def test_embed_duplicate_marker_first_wins():
    from llm.service import LLMService as S

    md = "[[illustration:2]] x [[illustration:2]]"
    out, used = S._embed_illustrations(md, {2: ("c", "u.jpg")})
    assert out.count("![c](u.jpg)") == 1
    assert "[[illustration:2]]" not in out
    assert used == {2}


def test_embed_orphan_marker_removed():
    from llm.service import LLMService as S

    out, used = S._embed_illustrations("a [[illustration:9]] b", {1: ("c", "u")})
    assert "[[illustration:9]]" not in out and used == set()


def test_embed_strips_user_injected_image():
    from llm.service import LLMService as S

    md = "![evil](http://evil/p.png) tale [[illustration:1]]"
    out, used = S._embed_illustrations(S._strip_untrusted_markup(md), {1: ("c", "u.jpg")})
    assert "evil" not in out
    assert "![c](u.jpg)" in out

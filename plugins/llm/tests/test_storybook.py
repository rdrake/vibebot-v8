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


def test_generate_story_struct_parses_and_validates(make_service, mocker):
    service, plugin = make_service(verseStorybookMaxImages=3, verseStorybookMaxChars=6000)
    payload = (
        "Here is your tale! ```json\n"
        '{"title":"The Tin Fox","story_markdown":"Once [[illustration:1]] fin.",'
        '"illustrations":[{"id":1,"caption":"a fox","image_prompt":"a tin fox"}]}\n```'
    )
    mocker.patch.object(service, "_ask_completion", return_value=payload)
    out = service._generate_story_struct(
        "spin a tale", channel="#c", persona="voice", conversation=[]
    )
    assert out["title"] == "The Tin Fox"
    assert out["illustrations"][0]["id"] == 1
    assert "[[illustration:1]]" in out["story_markdown"]


def test_generate_story_struct_retries_then_fails(make_service, mocker):
    service, plugin = make_service(verseStorybookMaxImages=3, verseStorybookMaxChars=6000)
    m = mocker.patch.object(service, "_ask_completion", return_value="no json here")
    out = service._generate_story_struct("x", channel="#c", persona="v", conversation=[])
    assert out is None
    assert m.call_count >= 3  # initial + >=2 retries


def test_validate_story_obj_drops_bad_illustrations():
    from llm.service import LLMService as S

    obj = {
        "title": "T",
        "story_markdown": "body",
        "illustrations": [
            {"id": 1, "caption": "ok", "image_prompt": "p"},
            {"id": "bad", "caption": "x", "image_prompt": "y"},
            {"id": 2, "caption": "no prompt", "image_prompt": "  "},
        ],
    }
    out = S._validate_story_obj(obj)
    assert [i["id"] for i in out["illustrations"]] == [1]


def test_validate_story_obj_requires_title_and_story():
    from llm.service import LLMService as S

    assert S._validate_story_obj({"title": "", "story_markdown": "x"}) is None
    assert S._validate_story_obj({"title": "T", "story_markdown": "  "}) is None
    assert S._validate_story_obj("not a dict") is None

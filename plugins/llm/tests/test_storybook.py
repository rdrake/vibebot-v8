"""Tests for the verse storybook tool."""

from __future__ import annotations


def test_storybook_config_defaults():
    import llm.config as cfg

    group = cfg.LLM
    assert group.verseStorybookEnabled is not None
    assert int(group.verseStorybookMaxImages()) == 5
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


def test_generate_storybook_embeds_and_saves(make_service, mocker, tmp_path):
    service, plugin = make_service(httpRoot=str(tmp_path), httpUrlBase="http://h/llm")
    mocker.patch.object(
        service,
        "_generate_story_struct",
        return_value={
            "title": "The Tin Fox",
            "story_markdown": "Once [[illustration:1]] upon a time.",
            "illustrations": [{"id": 1, "caption": "a fox", "image_prompt": "a tin fox"}],
        },
    )
    from llm.service import ImageResult

    mocker.patch.object(
        service,
        "_attempt_image_generation",
        return_value=ImageResult(content="ok", url="http://h/llm/img_fox.jpg"),
    )
    res = service.generate_storybook("brief", channel="#c", persona="v", conversation=[])
    assert res is not None and res.title == "The Tin Fox" and res.image_count == 1
    page = (tmp_path / res.url.split("/")[-1]).read_text()
    assert "<img" in page and 'src="img_fox.jpg"' in page  # bare filename embedded


def test_generate_storybook_image_failure_drops_marker(make_service, mocker, tmp_path):
    service, plugin = make_service(httpRoot=str(tmp_path), httpUrlBase="http://h/llm")
    mocker.patch.object(
        service,
        "_generate_story_struct",
        return_value={
            "title": "T",
            "story_markdown": "a [[illustration:1]] b",
            "illustrations": [{"id": 1, "caption": "c", "image_prompt": "p"}],
        },
    )
    from llm.service import ImageResult

    mocker.patch.object(
        service,
        "_attempt_image_generation",
        return_value=ImageResult(content="blocked", url=None, error="safety"),
    )
    res = service.generate_storybook("b", channel="#c", persona="v", conversation=[])
    assert res is not None and res.image_count == 0
    page = (tmp_path / res.url.split("/")[-1]).read_text()
    assert "[[illustration:1]]" not in page


def test_generate_storybook_image_none_drops_marker(make_service, mocker, tmp_path):
    service, plugin = make_service(httpRoot=str(tmp_path), httpUrlBase="http://h/llm")
    mocker.patch.object(
        service,
        "_generate_story_struct",
        return_value={
            "title": "T",
            "story_markdown": "a [[illustration:1]] b",
            "illustrations": [{"id": 1, "caption": "c", "image_prompt": "p"}],
        },
    )
    mocker.patch.object(service, "_attempt_image_generation", return_value=None)
    res = service.generate_storybook("b", channel="#c", persona="v", conversation=[])
    assert res is not None and res.image_count == 0


def test_generate_storybook_caps_images(make_service, mocker, tmp_path):
    service, plugin = make_service(
        httpRoot=str(tmp_path), httpUrlBase="http://h/llm", verseStorybookMaxImages=3
    )
    illos = [{"id": i, "caption": f"c{i}", "image_prompt": f"p{i}"} for i in range(1, 6)]
    markers = " ".join(f"[[illustration:{i}]]" for i in range(1, 6))
    mocker.patch.object(
        service,
        "_generate_story_struct",
        return_value={"title": "T", "story_markdown": markers, "illustrations": illos},
    )
    from llm.service import ImageResult

    gen = mocker.patch.object(
        service,
        "_attempt_image_generation",
        return_value=ImageResult(content="ok", url="http://h/llm/i.jpg"),
    )
    service.generate_storybook("b", channel="#c", persona="v", conversation=[])
    assert gen.call_count == 3  # capped at verseStorybookMaxImages


def test_generate_storybook_none_when_story_fails(make_service, mocker):
    service, plugin = make_service()
    mocker.patch.object(service, "_generate_story_struct", return_value=None)
    assert service.generate_storybook("b", channel="#c", persona="v", conversation=[]) is None


def test_resolves_to_public(make_service, mocker):
    service, plugin = make_service()
    mocker.patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("127.0.0.1", 443))])
    assert service._resolves_to_public("http://rebind.example/x.png") is False
    mocker.patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.216.34", 443))])
    assert service._resolves_to_public("http://example.com/x.png") is True
    # no host / unresolvable → False
    assert service._resolves_to_public("not-a-url") is False


def test_make_verse_tool_specs_includes_storybook_when_enabled():
    from llm.verse.avatar import make_verse_tool_specs

    on = make_verse_tool_specs(max_actors=2, storybook=True)
    off = make_verse_tool_specs(max_actors=2, storybook=False)

    def names(specs):
        out = []
        for s in specs:
            # support either {"function":{"name":...}} or {"name":...}
            out.append(s.get("function", s).get("name"))
        return out

    assert "verse_storybook" in names(on)
    assert "verse_storybook" not in names(off)


def test_storybook_job_records_canon_event(plugin_env, tmp_path, mocker):
    from llm.verse.store import VerseStore

    plugin, irc, msg = plugin_env
    store = VerseStore(tmp_path / "verse", "#afnet")
    avatar_id = store.opt_in_avatar(nick="alice", account="alice-acct", instruct_text="").entity_id
    mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)
    # Neutralize the async render/post job; the canon-record runs synchronously before it.
    mocker.patch.object(plugin, "_llm_executor")

    plugin._submit_storybook_job(
        channel="#afnet", nick="alice", persona="", brief="a tale of dragons", account="alice-acct"
    )

    events = store.recent_events(limit=10)
    assert len(events) == 1
    assert "dragons" in events[0].summary
    assert avatar_id in events[0].entity_ids


def test_storybook_job_no_avatar_skips_record(plugin_env, tmp_path, mocker):
    from llm.verse.store import VerseStore

    plugin, irc, msg = plugin_env
    store = VerseStore(tmp_path / "verse", "#afnet")  # nobody opted in
    mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)
    mocker.patch.object(plugin, "_llm_executor")

    plugin._submit_storybook_job(
        channel="#afnet", nick="ghost", persona="", brief="x", account=None
    )

    assert store.recent_events(limit=10) == []

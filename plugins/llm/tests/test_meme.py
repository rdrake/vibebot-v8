"""Meme templates: parsing, caption encoding, and template resolution.

The bot never picks a template. The user names one ("drake", "distracted
boyfriend"), the catalog resolves that name deterministically, and memegen.link
does the rendering. Every rule in here is the difference between a meme and a
404, so the encoding table follows memegen's README exactly.
"""

from __future__ import annotations

import json

import pytest
from llm import meme

_TEMPLATES_JSON = [
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
        "keywords": ["girlfriend", "cheating"],
        "example": {"text": ["me", "new thing", "old thing"]},
    },
    {
        "id": "fry",
        "name": "Futurama Fry",
        "lines": 2,
        "keywords": ["not sure if"],
        "example": {"text": ["not sure if trolling", "or just stupid"]},
    },
    {
        "id": "fine",
        "name": "This is Fine",
        "lines": 2,
        "keywords": ["dog", "fire"],
        "example": {"text": ["", "this is fine"]},
    },
]


@pytest.fixture
def catalog() -> meme.MemeCatalog:
    return meme.MemeCatalog(meme.parse_templates(_TEMPLATES_JSON))


class TestParseMemeRequest:
    def test_pipe_separated_template_then_lines(self):
        assert meme.parse_meme_request("drake | left on unread | left on read") == (
            "drake",
            ["left on unread", "left on read"],
        )

    def test_template_alone_has_no_lines(self):
        assert meme.parse_meme_request("drake") == ("drake", [])

    def test_blank_is_none(self):
        assert meme.parse_meme_request("   ") is None

    def test_leading_pipe_is_a_blank_template(self):
        assert meme.parse_meme_request("| a | b") is None


class TestEncodeCaption:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("left on read", "left_on_read"),
            ("snake_case", "snake__case"),
            ("well-known", "well--known"),
            ("why?", "why~q"),
            ("a & b", "a_~a_b"),
            ("100%", "100~p"),
            ("#irc", "~hirc"),
            ("a/b", "a~sb"),
            ("a\\b", "a~bb"),
            ("<html>", "~lhtml~g"),
            ('say "hi"', "say_''hi''"),
            ("two\nlines", "two~nlines"),
            ("", "_"),
            ("   ", "_"),
        ],
    )
    def test_memegen_special_characters(self, text: str, expected: str):
        assert meme.encode_caption(text) == expected

    def test_non_ascii_is_percent_encoded(self):
        assert meme.encode_caption("café") == "caf%C3%A9"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("y’all", "y'all"),
            ("‘quoted’", "'quoted'"),
            ("“hi”", "''hi''"),
            ("a–b", "a--b"),
        ],
    )
    def test_phone_punctuation_is_normalised_first(self, text: str, expected: str):
        """memegen answers a curly quote or an en dash with a 301 to the ASCII
        form, and the downloader refuses redirects — prod 2026-09-18 00:04Z,
        'y’all got any more of them tokens' from an iPhone keyboard."""
        assert meme.encode_caption(text) == expected


class TestBuildMemeUrl:
    def test_lines_become_path_segments(self):
        url = meme.build_meme_url("https://api.memegen.link", "drake", ["a b", "c?"])
        assert url == "https://api.memegen.link/images/drake/a_b/c~q.png"

    def test_base_trailing_slash_is_tolerated(self):
        url = meme.build_meme_url("https://api.memegen.link/", "drake", ["a", "b"])
        assert url == "https://api.memegen.link/images/drake/a/b.png"


class TestParseTemplates:
    def test_reads_memegen_shape(self):
        templates = meme.parse_templates(_TEMPLATES_JSON)
        db = next(t for t in templates if t.id == "db")
        assert db.name == "Distracted Boyfriend"
        assert db.lines == 3
        assert db.keywords == ("girlfriend", "cheating")
        assert db.example == ("me", "new thing", "old thing")

    def test_skips_malformed_entries(self):
        templates = meme.parse_templates([{"id": "x"}, {"name": "no id", "lines": 2}, "junk"])
        assert templates == []


class TestResolve:
    def test_exact_id(self, catalog):
        assert catalog.resolve("drake").id == "drake"

    def test_id_is_case_insensitive(self, catalog):
        assert catalog.resolve("DRAKE").id == "drake"

    def test_name_substring(self, catalog):
        assert catalog.resolve("distracted boyfriend").id == "db"
        assert catalog.resolve("boyfriend").id == "db"

    def test_keyword(self, catalog):
        assert catalog.resolve("cheating").id == "db"

    def test_articles_and_punctuation_do_not_matter(self, catalog):
        assert catalog.resolve("the this-is-fine").id == "fine"

    def test_no_match_is_none(self, catalog):
        assert catalog.resolve("loss") is None

    def test_ambiguous_substring_is_none(self):
        cat = meme.MemeCatalog(
            meme.parse_templates(
                [
                    {"id": "a", "name": "Grumpy Cat", "lines": 2, "keywords": []},
                    {"id": "b", "name": "Business Cat", "lines": 2, "keywords": []},
                ]
            )
        )
        assert cat.resolve("cat") is None

    def test_suggest_ranks_closest_first(self, catalog):
        ids = [t.id for t in catalog.suggest("fin", limit=5)]
        assert ids[0] == "fine"


class TestPlanMeme:
    """One entry point for both @meme and the chat tool."""

    def test_happy_path(self, catalog):
        plan = meme.plan_meme(catalog, "https://api.memegen.link", "drake", ["a", "b"])
        assert plan.error is None
        assert plan.url == "https://api.memegen.link/images/drake/a/b.png"
        assert plan.template.id == "drake"

    def test_fewer_lines_are_padded_blank(self, catalog):
        plan = meme.plan_meme(catalog, "https://api.memegen.link", "fine", ["this is fine"])
        assert plan.url == "https://api.memegen.link/images/fine/this_is_fine/_.png"

    def test_no_lines_is_an_error(self, catalog):
        plan = meme.plan_meme(catalog, "https://api.memegen.link", "drake", [])
        assert plan.url is None
        assert "2 captions" in plan.error
        assert "left on unread" in plan.error

    def test_too_many_lines_names_the_limit(self, catalog):
        plan = meme.plan_meme(catalog, "https://api.memegen.link", "drake", ["a", "b", "c"])
        assert plan.url is None
        assert "takes 2 captions" in plan.error
        assert "you gave 3" in plan.error

    def test_tied_template_suggests_the_candidates(self, catalog):
        plan = meme.plan_meme(catalog, "https://api.memegen.link", "this fry", ["a"])
        assert plan.url is None
        assert "this fry" in plan.error
        assert "fine" in plan.error and "fry" in plan.error

    def test_caption_too_long_is_an_error(self, catalog):
        plan = meme.plan_meme(catalog, "https://api.memegen.link", "drake", ["x" * 300, "b"])
        assert plan.url is None
        assert "too long" in plan.error


class TestFetchTemplates:
    def test_parses_the_templates_endpoint(self, mocker):
        body = json.dumps(_TEMPLATES_JSON).encode()
        resp = mocker.MagicMock()
        resp.read.return_value = body
        resp.__enter__.return_value = resp
        opener = mocker.MagicMock()
        opener.open.return_value = resp
        templates = meme.fetch_templates("https://api.memegen.link", timeout=5, opener=opener)
        assert [t.id for t in templates] == ["drake", "db", "fry", "fine"]
        called_url = opener.open.call_args.args[0]
        assert called_url.full_url == "https://api.memegen.link/templates/"

    def test_refuses_unsafe_base(self):
        with pytest.raises(ValueError):
            meme.fetch_templates("http://127.0.0.1/", timeout=5)


class TestCachedCatalog:
    def test_first_failure_leaves_no_catalog_and_records_why(self, mocker):
        mocker.patch.object(meme, "fetch_templates", side_effect=OSError("down"))
        cached = meme.CachedCatalog("https://api.memegen.link", timeout=5)
        assert cached.get(now=1000.0) is None
        assert "down" in (cached.last_error or "")

    def test_failed_refresh_keeps_the_last_good_list(self, mocker):
        good = meme.parse_templates(_TEMPLATES_JSON)
        fetch = mocker.patch.object(meme, "fetch_templates", side_effect=[good, OSError("down")])
        cached = meme.CachedCatalog("https://api.memegen.link", timeout=5, ttl=100.0)
        first = cached.get(now=1000.0)
        second = cached.get(now=1200.0)
        assert first is not None and len(first) == 4
        assert second is first
        assert fetch.call_count == 2

    def test_within_ttl_does_not_refetch(self, mocker):
        fetch = mocker.patch.object(
            meme, "fetch_templates", return_value=meme.parse_templates(_TEMPLATES_JSON)
        )
        cached = meme.CachedCatalog("https://api.memegen.link", timeout=5, ttl=100.0)
        cached.get(now=1000.0)
        cached.get(now=1050.0)
        assert fetch.call_count == 1

    def test_failure_is_not_retried_until_ttl(self, mocker):
        """A dead memegen must not cost a GET per @meme while it is down."""
        fetch = mocker.patch.object(meme, "fetch_templates", side_effect=OSError("down"))
        cached = meme.CachedCatalog("https://api.memegen.link", timeout=5, ttl=100.0)
        cached.get(now=1000.0)
        cached.get(now=1001.0)
        assert fetch.call_count == 1


_WONKA_JSON = [
    {"id": "wonka", "name": "Condescending Wonka", "lines": 2, "keywords": []},
    {"id": "gb", "name": "Galaxy Brain", "lines": 4, "keywords": []},
    {"id": "sb", "name": "Scumbag Brain", "lines": 2, "keywords": []},
    {"id": "elmo", "name": "Elmo Choosing Cocaine", "lines": 5, "keywords": []},
    {"id": "yallgot", "name": "Y'all Got Any More of Them", "lines": 2, "keywords": []},
    {"id": "pigeon", "name": "Is This a Pigeon?", "lines": 3, "keywords": []},
    {"id": "fine", "name": "This is Fine", "lines": 2, "keywords": []},
]


@pytest.fixture
def wonka() -> meme.MemeCatalog:
    return meme.MemeCatalog(meme.parse_templates(_WONKA_JSON))


class TestResolveByTokenOverlap:
    """memegen names templates by quote, users name them by character."""

    def test_half_the_words_hitting_one_template_is_enough(self, wonka):
        assert wonka.resolve("willy wonka").id == "wonka"
        assert wonka.resolve("elmo fire").id == "elmo"

    def test_a_tie_is_still_a_miss(self, wonka):
        assert wonka.resolve("expanding brain") is None

    def test_one_word_of_five_is_not_enough(self, wonka):
        assert wonka.resolve("what if i told you them") is None

    def test_more_words_in_common_wins(self, wonka):
        assert wonka.resolve("is this a pigeon").id == "pigeon"
        assert wonka.resolve("this is fine").id == "fine"


class TestSuggestHonesty:
    def test_nothing_in_common_suggests_nothing(self, wonka):
        assert wonka.suggest("tyrone biggums") == []

    def test_partial_overlap_still_suggests(self, wonka):
        assert [t.id for t in wonka.suggest("brain")] == ["gb", "sb"]

    def test_unknown_with_no_suggestions_points_at_list_and_urls(self, wonka):
        plan = meme.plan_meme(wonka, "https://api.memegen.link", "tyrone", ["a", "b"])
        assert plan.url is None
        assert "@meme list" in plan.error and "image URL" in plan.error


class TestAliases:
    def test_parse_aliases_reads_name_equals_target(self):
        parsed = meme.parse_aliases(
            ["tyrone=yallgot", "junk", "=x", "y=", "wojak=https://i.example.com/w.png"]
        )
        assert parsed == {"tyrone": "yallgot", "wojak": "https://i.example.com/w.png"}

    def test_id_alias_becomes_a_keyword_on_the_target(self, wonka):
        cat = wonka.with_aliases({"tyrone": "yallgot", "biggums": "yallgot"})
        assert cat.resolve("tyrone").id == "yallgot"
        assert cat.resolve("tyrone biggums").id == "yallgot"
        assert "tyrone" in cat.resolve("yallgot").keywords

    def test_url_alias_becomes_a_two_box_custom_template(self, wonka):
        cat = wonka.with_aliases({"wojak": "https://i.example.com/wojak.png"})
        t = cat.resolve("wojak")
        assert t.lines == 2 and t.background == "https://i.example.com/wojak.png"
        assert len(cat) == len(wonka) + 1
        plan = meme.plan_meme(cat, "https://api.memegen.link", "wojak", ["a", "b"])
        assert plan.url == (
            "https://api.memegen.link/images/custom/a/b.png"
            "?background=https%3A%2F%2Fi.example.com%2Fwojak.png"
        )

    def test_alias_to_unknown_id_or_unsafe_url_is_dropped(self, wonka):
        cat = wonka.with_aliases({"a": "nosuch", "b": "http://127.0.0.1/x.png"})
        assert cat.resolve("a") is None and cat.resolve("b") is None
        assert len(cat) == len(wonka)


class TestCustomBackground:
    _BG = "https://i.imgflip.com/1c1uej.jpg"

    def test_url_template_renders_through_custom(self, wonka):
        plan = meme.plan_meme(wonka, "https://api.memegen.link", self._BG, ["y'all got", "more?"])
        assert plan.url == (
            "https://api.memegen.link/images/custom/y'all_got/more~q.png"
            "?background=https%3A%2F%2Fi.imgflip.com%2F1c1uej.jpg"
        )
        assert plan.template.id == "custom"

    def test_custom_takes_two_captions(self, wonka):
        plan = meme.plan_meme(wonka, "https://api.memegen.link", self._BG, ["a", "b", "c"])
        assert plan.url is None and "takes 2 captions" in plan.error

    def test_private_background_is_refused(self, wonka):
        plan = meme.plan_meme(wonka, "https://api.memegen.link", "http://10.0.0.1/x.png", ["a"])
        assert plan.url is None and "URL" in plan.error

    def test_parse_request_keeps_the_url_intact(self):
        assert meme.parse_meme_request(f"{self._BG} | a | b") == (self._BG, ["a", "b"])


_STYLED_JSON = [
    {
        "id": "fine",
        "name": "This is Fine",
        "lines": 2,
        "keywords": [],
        "styles": ["animated", "default"],
    },
    {
        "id": "doge",
        "name": "Doge",
        "lines": 2,
        "keywords": [],
        "styles": ["bark", "bite", "default"],
    },
    {"id": "drake", "name": "Drakeposting", "lines": 2, "keywords": [], "styles": []},
]


@pytest.fixture
def styled() -> meme.MemeCatalog:
    return meme.MemeCatalog(meme.parse_templates(_STYLED_JSON))


class TestOptions:
    """The rest of memegen's query string: gif, style, font, layout."""

    B = "https://api.memegen.link"

    def test_parse_templates_reads_styles(self, styled):
        assert styled.resolve("doge").styles == ("bark", "bite")
        assert styled.resolve("fine").animated is True
        assert styled.resolve("drake").animated is False

    def test_animated_renders_a_gif(self, styled):
        plan = meme.plan_meme(styled, self.B, "fine", ["", "ok"], meme.MemeOptions(animated=True))
        assert plan.url == f"{self.B}/images/fine/_/ok.gif?style=animated"

    def test_animated_on_a_still_template_lists_the_animated_ones(self, styled):
        plan = meme.plan_meme(styled, self.B, "drake", ["a", "b"], meme.MemeOptions(animated=True))
        assert plan.url is None
        assert "drake" in plan.error and "fine" in plan.error

    def test_style_rides_as_a_query(self, styled):
        plan = meme.plan_meme(styled, self.B, "doge", ["a", "b"], meme.MemeOptions(style="bark"))
        assert plan.url == f"{self.B}/images/doge/a/b.png?style=bark"

    def test_unknown_style_lists_the_template_styles(self, styled):
        plan = meme.plan_meme(styled, self.B, "doge", ["a", "b"], meme.MemeOptions(style="meow"))
        assert plan.url is None
        assert "bark" in plan.error and "bite" in plan.error

    def test_style_on_a_template_without_any(self, styled):
        plan = meme.plan_meme(styled, self.B, "drake", ["a", "b"], meme.MemeOptions(style="yes"))
        assert plan.url is None and "no styles" in plan.error

    def test_font_and_top_layout(self, styled):
        opts = meme.MemeOptions(font="impact", layout_top=True)
        plan = meme.plan_meme(styled, self.B, "drake", ["a", "b"], opts)
        assert plan.url == f"{self.B}/images/drake/a/b.png?font=impact&layout=top"

    def test_unknown_font_lists_the_fonts(self, styled):
        plan = meme.plan_meme(styled, self.B, "drake", ["a", "b"], meme.MemeOptions(font="comic"))
        assert plan.url is None and "impact" in plan.error

    def test_custom_background_keeps_its_query_alongside(self, styled):
        opts = meme.MemeOptions(font="impact")
        plan = meme.plan_meme(styled, self.B, "https://i.example.com/x.png", ["a", "b"], opts)
        assert plan.url.endswith(".png?font=impact&background=https%3A%2F%2Fi.example.com%2Fx.png")

    def test_custom_cannot_animate(self, styled):
        opts = meme.MemeOptions(animated=True)
        plan = meme.plan_meme(styled, self.B, "https://i.example.com/x.png", ["a"], opts)
        assert plan.url is None and "animated" in plan.error

    def test_describe_mentions_gif_and_styles(self, styled):
        assert meme.describe_short(styled.resolve("fine")) == "fine (This is Fine, 2, gif)"
        assert meme.describe_short(styled.resolve("doge")) == "doge (Doge, 2, styles: bark/bite)"
        assert meme.describe_short(styled.resolve("drake")) == "drake (Drakeposting, 2)"


class TestCanonicalUrl:
    """memegen 301s to its canonical spelling; the downloader refuses redirects.

    So ask memegen first (HEAD, no render) and take its answer only when it
    stays on memegen under /images/ — anywhere else is somebody's redirect.
    """

    B = "https://api.memegen.link"

    def _opener(self, mocker, responses):
        """responses: list of (status, location) per call; status None = 200."""
        import urllib.error

        opener = mocker.MagicMock()
        calls: list[str] = []

        def open_(req, timeout=None):
            calls.append(req.full_url)
            status, location = responses[min(len(calls) - 1, len(responses) - 1)]
            if status is None:
                resp = mocker.MagicMock()
                resp.__enter__.return_value = resp
                resp.status = 200
                return resp
            headers = {"Location": location} if location else {}
            raise urllib.error.HTTPError(req.full_url, status, "x", headers, None)

        opener.open.side_effect = open_
        return opener, calls

    def test_no_redirect_keeps_the_url(self, mocker):
        opener, calls = self._opener(mocker, [(None, None)])
        url = f"{self.B}/images/drake/a/b.png"
        assert meme.canonical_url(url, timeout=5, opener=opener) == url
        assert calls == [url]

    def test_same_origin_images_redirect_is_followed(self, mocker):
        opener, _ = self._opener(mocker, [(301, "/images/drake/a----b/c.png"), (None, None)])
        got = meme.canonical_url(f"{self.B}/images/drake/a_--_b/c.png", timeout=5, opener=opener)
        assert got == f"{self.B}/images/drake/a----b/c.png"

    def test_query_string_survives_the_hop(self, mocker):
        opener, _ = self._opener(mocker, [(301, "/images/doge/a_b/c.png"), (None, None)])
        got = meme.canonical_url(
            f"{self.B}/images/doge/a-b/c.png?style=bark", timeout=5, opener=opener
        )
        assert got == f"{self.B}/images/doge/a_b/c.png?style=bark"

    def test_other_host_redirect_is_ignored(self, mocker):
        opener, _ = self._opener(mocker, [(301, "https://evil.example/x.png")])
        url = f"{self.B}/images/drake/a/b.png"
        assert meme.canonical_url(url, timeout=5, opener=opener) == url

    def test_redirect_outside_images_is_ignored(self, mocker):
        opener, _ = self._opener(mocker, [(301, "/templates/drake")])
        url = f"{self.B}/images/drake/a/b.png"
        assert meme.canonical_url(url, timeout=5, opener=opener) == url

    def test_head_error_keeps_the_url(self, mocker):
        """HEAD on a gif answers 422; that is not a redirect."""
        opener, _ = self._opener(mocker, [(422, None)])
        url = f"{self.B}/images/fine/_/ok.gif?style=animated"
        assert meme.canonical_url(url, timeout=5, opener=opener) == url

    def test_network_failure_keeps_the_url(self, mocker):
        opener = mocker.MagicMock()
        opener.open.side_effect = OSError("down")
        url = f"{self.B}/images/drake/a/b.png"
        assert meme.canonical_url(url, timeout=5, opener=opener) == url

    def test_hops_are_capped(self, mocker):
        opener, calls = self._opener(mocker, [(301, "/images/drake/loop/x.png")])
        meme.canonical_url(f"{self.B}/images/drake/a/b.png", timeout=5, opener=opener)
        assert len(calls) <= 4


class TestTopics:
    """The checked-in keyword column: memegen's own keywords cover a quarter
    of the templates, so "gym" found nothing until meme_topics tagged bd."""

    def test_with_keywords_adds_tags_and_ignores_unknown_ids(self, catalog):
        cat = catalog.with_keywords({"fine": ("burning", "coping"), "nosuch": ("x",)})
        assert cat.resolve("burning").id == "fine"
        assert cat.resolve("coping meme").id == "fine"
        assert cat.resolve("fine").keywords == ("dog", "fire", "burning", "coping")
        assert len(cat) == len(catalog)

    def test_shared_tag_is_ambiguous_not_a_guess(self, catalog):
        cat = catalog.with_keywords({"fine": ("sad",), "fry": ("sad",)})
        assert cat.resolve("sad") is None
        assert {t.id for t in cat.suggest("sad")} == {"fine", "fry"}

    def test_every_topic_id_and_tag_is_well_formed(self):
        from llm import meme_topics

        for template_id, tags in meme_topics.TOPICS.items():
            assert template_id == template_id.lower().strip(), template_id
            assert tags, template_id
            for tag in tags:
                assert meme._normalise(tag), (template_id, tag)

    def test_list_and_suggest_search_the_example_captions(self, catalog):
        assert meme.matches(catalog.resolve("fry"), "trolling")
        assert [t.id for t in catalog.suggest("trolling")] == ["fry"]

    def test_resolve_does_not_read_the_example_captions(self, catalog):
        assert catalog.resolve("trolling") is None

    def test_a_name_word_outranks_topic_tags(self, catalog):
        elmo = meme.MemeTemplate("elmo", "Elmo Choosing Cocaine", 2)
        cat = meme.MemeCatalog([*catalog.templates, elmo]).with_keywords(
            {"fine": ("fire", "burning"), "elmo": ("sesame street",)}
        )
        assert cat.resolve("elmo fire").id == "elmo"

    def test_suggest_ignores_two_letter_words_in_a_longer_query(self):
        cat = meme.MemeCatalog(
            meme.parse_templates(
                [
                    {"id": "money", "name": "Shut Up and Take My Money!", "lines": 2},
                    {"id": "yuno", "name": "Y U NO Guy", "lines": 2},
                ]
            )
        )
        assert cat.suggest("push-up") == []
        assert [t.id for t in cat.suggest("y u no")] == ["yuno"]


class TestPicker:
    """The model's answer is a proposal; the catalog decides whether it stands."""

    def test_catalog_brief_is_one_line_per_template(self, catalog):
        brief = meme.catalog_brief(catalog.with_keywords({"fine": ("coping",)}))
        rows = brief.splitlines()
        assert len(rows) == len(catalog)
        assert "fine | This is Fine | 2 | _ / this is fine | dog, fire, coping" in rows
        assert rows[0].startswith("drake | Drakeposting | 2 | left on unread / left on read")

    def test_valid_pick_resolves_and_keeps_lines(self, catalog):
        choice = meme.parse_pick('{"template": "drake", "lines": ["a", "b"]}', catalog)
        assert choice.template.id == "drake" and choice.lines == ["a", "b"]

    def test_pick_tolerates_fences_and_names(self, catalog):
        text = '```json\n{"template": "Distracted Boyfriend", "lines": ["me", "x", "y"]}\n```'
        assert meme.parse_pick(text, catalog).template.id == "db"

    def test_extra_lines_are_cut_not_refused(self, catalog):
        choice = meme.parse_pick('{"template": "drake", "lines": ["a", "b", "c"]}', catalog)
        assert choice.lines == ["a", "b"]

    def test_null_template_returns_the_reason(self, catalog):
        assert meme.parse_pick('{"template": null, "reason": "No gym meme."}', catalog) == (
            "No gym meme."
        )

    def test_invented_id_is_a_miss(self, catalog):
        result = meme.parse_pick('{"template": "gigachad", "lines": ["a"]}', catalog)
        assert isinstance(result, str) and "gigachad" in result

    @pytest.mark.parametrize(
        "content",
        [None, "", "sure!", '{"template": "drake"}', '{"template": "drake", "lines": []}'],
    )
    def test_unusable_answers_are_misses(self, catalog, content):
        assert isinstance(meme.parse_pick(content, catalog), str)

    def test_pick_carries_an_optional_draw(self, catalog):
        text = '{"template": "drake", "lines": ["a", "b"], "draw": "a tired dad"}'
        assert meme.parse_pick(text, catalog).draw == "a tired dad"
        assert meme.parse_pick('{"template": "drake", "lines": ["a", "b"]}', catalog).draw is None

    def test_pick_animated_only_when_the_template_has_a_gif(self):
        cat = meme.MemeCatalog(
            meme.parse_templates(
                [
                    {"id": "fine", "name": "This is Fine", "lines": 2, "styles": ["animated"]},
                    {"id": "drake", "name": "Drakeposting", "lines": 2},
                ]
            )
        )
        assert meme.parse_pick(
            '{"template": "fine", "lines": ["a"], "animated": true}', cat
        ).animated
        assert not meme.parse_pick(
            '{"template": "drake", "lines": ["a"], "animated": true}', cat
        ).animated
        assert "fine | This is Fine | 2 |  | gif" in meme.catalog_brief(cat)

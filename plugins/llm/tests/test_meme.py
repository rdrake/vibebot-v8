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

    def test_suggest_falls_back_to_popular_when_nothing_matches(self, catalog):
        assert len(catalog.suggest("zzzz", limit=3)) == 3


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

    def test_unknown_template_suggests(self, catalog):
        plan = meme.plan_meme(catalog, "https://api.memegen.link", "loss", ["a"])
        assert plan.url is None
        assert "loss" in plan.error
        assert "drake" in plan.error

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

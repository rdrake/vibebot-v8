import re as _re

from llm.verse.taste_mine import Msg, classify_praise, classify_repaste, iter_messages


class _Ent:
    def __init__(self, name):
        self.name = name


class FakeStore:
    def __init__(self, names):
        self._names = names

    def match_entities_in_text(self, text, limit=12):
        low = text.lower()
        return [
            _Ent(n)
            for n in self._names
            if _re.search(r"(?<!\w)" + _re.escape(n.lower()) + r"(?!\w)", low)
        ][:limit]


def test_iter_messages_parses_privmsg_action_skips_rest():
    lines = [
        "2026-06-22T00:01:16  <fc42> a normal message",
        "2026-06-22T00:02:00  * fc42 does a thing",  # CTCP ACTION / /me
        "2026-06-22T00:03:00  *** vibebot has joined #afternet",  # system -> skip
        "2026-06-22T00:04:00  -ChanServ- a notice",  # notice -> skip
        "2026-06-22T00:05:00  <fc42> ",  # empty body -> skip
        "garbage no double-space sep",  # malformed -> skip
        "2026-06-22T00:06:00  <rdrake> hi th�ere",  # garbled char -> kept
    ]
    assert list(iter_messages(lines)) == [
        Msg("fc42", "a normal message"),
        Msg("fc42", "does a thing"),
        Msg("rdrake", "hi th�ere"),
    ]


def test_repaste_long_prose_naming_entity_is_autotrusted():
    store = FakeStore(["stinky lads", "Ripping Robert"])
    text = (
        "the stinky lads marched into the assembly hall and ripping robert let "
        "off a perfectly timed duet that turned the leaves yellow indeed"
    )
    c = classify_repaste(text, store)
    assert c is not None and c.kind == "repaste" and c.needs_review is False  # multiword


def test_repaste_short_lowercase_only_match_flags_review():
    store = FakeStore(["Ghost"])
    text = (
        "i didn't have a ghost of a chance against that lot in the second half today "
        "mate it was a proper disaster from start to finish honestly yeah"
    )
    c = classify_repaste(text, store)
    assert c is not None and c.needs_review is True


def test_repaste_rejects_short_url_and_addressed():
    store = FakeStore(["stinky lads"])
    assert classify_repaste("stinky lads", store) is None  # < 120
    assert classify_repaste("grok " + "the stinky lads are great " * 6, store) is None  # addressed
    assert classify_repaste("look https://x.com/" + "a" * 120, store) is None  # URL


def test_repaste_keeps_name_led_prose():
    # addressed filter is narrowed to grok|vibebot, so name-led prose survives
    store = FakeStore(["stinky lads", "Larry"])
    text = (
        "Larry marched into the assembly hall with the stinky lads and let off a guff "
        "cloud that lingered for several minutes much to everyone's dismay yeah"
    )
    assert classify_repaste(text, store) is not None


def test_praise_inline_keeps_leading_article_starts_at_entity():
    store = FakeStore(["stinky lads"])
    line = (
        "i love it when it said earlier that the stinky lads will either rule "
        "the country or set it on fire"
    )
    c = classify_praise(line, store, prev_line="(some bot line)")
    assert c is not None and c.needs_review is True
    assert c.text.startswith(
        "the stinky lads will either rule"
    )  # 'earlier that' stripped, 'the' kept
    assert "earlier that" not in c.text


def test_praise_bare_attaches_source_line():
    store = FakeStore(["stinky lads"])
    c = classify_praise(
        "haha this is a good one", store, prev_line="the stinky lads stormed the chippy"
    )
    assert c is not None and c.needs_review is True
    assert c.text == "the stinky lads stormed the chippy"


def test_praise_wordlist_is_word_bounded():
    store = FakeStore(["stinky lads"])
    # "classroom" must NOT trigger the "class" praise word
    assert classify_praise("the classroom was loud", store, prev_line="x") is None


def test_non_praise_returns_none():
    store = FakeStore(["stinky lads"])
    assert classify_praise("what time is the match", store, prev_line="x") is None

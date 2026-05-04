"""Property-based round-trip tests for conversation persistence.

Locks down the JSON serialize/deserialize round trip and the
case-insensitive (nick, channel) keying for ``save_conversation``,
``load_conversations``, and ``delete_conversation``.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis.strategies import (
    characters,
    fixed_dictionaries,
    floats,
    lists,
    sampled_from,
    text,
)
from llm.persistence import LLMDatabase

# Lone surrogates are safe here: persistence.py:489 calls json.dumps with the
# default ensure_ascii=True, which escapes them as \uXXXX. If that ever
# changes, switch to text(alphabet=characters(blacklist_categories=["Cs"]),...)
_message_strategy = lists(
    fixed_dictionaries(
        {
            "role": sampled_from(["user", "assistant", "system"]),
            "content": text(max_size=500),
        }
    ),
    max_size=20,
)

_nick_strategy = text(
    alphabet=characters(min_codepoint=0x21, max_codepoint=0x7E, blacklist_characters=" "),
    min_size=1,
    max_size=15,
)

_channel_strategy = sampled_from(["#a", "#b", "#priv1"])

_last_activity_strategy = floats(
    min_value=0, max_value=2_000_000_000, allow_nan=False, allow_infinity=False
)


@given(
    nick=_nick_strategy,
    channel=_channel_strategy,
    messages=_message_strategy,
    last_activity=_last_activity_strategy,
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_save_load_round_trip(
    tmp_path_factory,
    nick: str,
    channel: str,
    messages: list[dict[str, str]],
    last_activity: float,
) -> None:
    """``save → load`` returns ``messages`` unchanged at the canonicalized key."""
    db_path = tmp_path_factory.mktemp("conv-roundtrip") / "c.db"
    db = LLMDatabase(str(db_path))
    try:
        db.save_conversation(nick, channel, messages, last_activity)
        rows = db.load_conversations()
        matching = [
            (n, c, m, t) for (n, c, m, t) in rows if (n, c) == (nick.lower(), channel.lower())
        ]
        assert len(matching) == 1
        assert matching[0][2] == messages
        assert matching[0][3] == pytest.approx(last_activity)
    finally:
        db.close()


@given(
    nick=_nick_strategy,
    channel=_channel_strategy,
    messages_a=_message_strategy,
    messages_b=_message_strategy,
    last_activity=_last_activity_strategy,
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_save_save_load_no_duplicates(
    tmp_path_factory,
    nick: str,
    channel: str,
    messages_a: list[dict[str, str]],
    messages_b: list[dict[str, str]],
    last_activity: float,
) -> None:
    """Two saves at the same key replace, never duplicate."""
    db_path = tmp_path_factory.mktemp("conv-noduplicates") / "c.db"
    db = LLMDatabase(str(db_path))
    try:
        db.save_conversation(nick, channel, messages_a, last_activity)
        db.save_conversation(nick, channel, messages_b, last_activity + 1)
        rows = db.load_conversations()
        matching = [
            (n, c, m, t) for (n, c, m, t) in rows if (n, c) == (nick.lower(), channel.lower())
        ]
        assert len(matching) == 1
        assert matching[0][2] == messages_b
    finally:
        db.close()


@given(
    nick=_nick_strategy,
    channel=_channel_strategy,
    messages=_message_strategy,
    last_activity=_last_activity_strategy,
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_delete_is_case_insensitive(
    tmp_path_factory,
    nick: str,
    channel: str,
    messages: list[dict[str, str]],
    last_activity: float,
) -> None:
    """``save("Alice","#X") → delete("ALICE","#x") → load`` returns no row.

    Save uses lowercase, delete uses uppercase (or vice versa) so the
    property catches a regression that drops ``.lower()`` from either path.
    """
    db_path = tmp_path_factory.mktemp("conv-delete") / "c.db"
    db = LLMDatabase(str(db_path))
    try:
        # Save lowercases everything; delete must also lowercase to match.
        db.save_conversation(nick.lower(), channel.lower(), messages, last_activity)
        db.delete_conversation(nick.upper(), channel.upper())
        rows = db.load_conversations()
        matching = [(n, c) for (n, c, _m, _t) in rows if (n, c) == (nick.lower(), channel.lower())]
        assert matching == []
    finally:
        db.close()

"""Property-based equivalence tests for ``Identity.matches`` and ``Identity.key``.

``Identity.matches`` uses ``ircutils.toLower`` (``plugin.py:120-121``),
which implements RFC 1459 casemapping where ``[]\\^`` are the
upper-case partners of ``{}|~``. Python's ``str.lower()`` does not
bridge these characters (``'['.lower() == '['``), so a strategy that
constructs case-equivalent pairs via ``str.upper()`` / ``str.lower()``
cannot exercise the IRC-specific behavior. We build pairs with
``ircutils.toLower`` directly.
"""

from __future__ import annotations

from string import ascii_letters, digits

from hypothesis import given
from hypothesis.strategies import builds, none, one_of, text
from llm.plugin import Identity
from supybot import ircutils

# Alphabet includes the IRC special-case characters []{}\\^| so the
# RFC 1459 casemapping in ircutils.toLower is exercised, not just str.lower.
_NICK_ALPHABET = ascii_letters + digits + "[]{}\\^_-|"
nicks = text(alphabet=_NICK_ALPHABET, min_size=1, max_size=15)
accounts = one_of(none(), nicks)


@given(raw=nicks, account=accounts)
def test_matches_is_reflexive(raw: str, account: str | None) -> None:
    ident = Identity(raw_nick=raw, account=account)
    assert ident.matches(ident)


@given(raw_a=nicks, raw_b=nicks, acct_a=accounts, acct_b=accounts)
def test_matches_is_symmetric(
    raw_a: str, raw_b: str, acct_a: str | None, acct_b: str | None
) -> None:
    a = Identity(raw_nick=raw_a, account=acct_a)
    b = Identity(raw_nick=raw_b, account=acct_b)
    assert a.matches(b) == b.matches(a)


@given(
    pair=builds(lambda r: (r, ircutils.toLower(r)), nicks),
    account=accounts,
)
def test_matches_uses_irc_casemapping_on_raw_nick(
    pair: tuple[str, str],
    account: str | None,  # noqa: ARG001 -- account unused; reserved for future
) -> None:
    """``[, ], \\, ^`` should be case-equivalent to ``{, }, |, ~`` via toLower."""
    raw_upper, raw_lower = pair
    a = Identity(raw_nick=raw_upper, account=None)
    b = Identity(raw_nick=raw_lower, account=None)
    assert a.matches(b)


@given(pair=builds(lambda r: (r, ircutils.toLower(r)), nicks))
def test_matches_uses_irc_casemapping_on_account(pair: tuple[str, str]) -> None:
    acct_upper, acct_lower = pair
    a = Identity(raw_nick="x", account=acct_upper)
    b = Identity(raw_nick="y", account=acct_lower)
    # When both have an account, raw_nick is irrelevant.
    assert a.matches(b)


@given(raw=nicks, acct=nicks)
def test_account_overrides_raw_nick_in_matches(raw: str, acct: str) -> None:
    """When both have the same account, mismatched raw_nicks still match."""
    a = Identity(raw_nick=raw + "_distinct_a", account=acct)
    b = Identity(raw_nick=raw + "_distinct_b", account=acct)
    assert a.matches(b)


@given(raw=nicks, acct=nicks)
def test_key_equals_account_when_present(raw: str, acct: str) -> None:
    """Two identities with the same account share a storage key
    regardless of raw nick."""
    assert (
        Identity(raw_nick=raw, account=acct).key
        == Identity(raw_nick=raw + "_other", account=acct).key
    )


@given(raw=nicks)
def test_key_falls_back_to_raw_nick_when_unidentified(raw: str) -> None:
    assert Identity(raw_nick=raw, account=None).key == raw

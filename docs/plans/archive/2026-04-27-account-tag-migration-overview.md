---
status: revised-after-review
date: 2026-04-27
---

# Account-tag identity migration — high-level overview

## Why

Channel join sync is slow because Limnoria auto-queries `WHO`, `MODE`, and `MODE +b` per channel, gated by a 1.0s outbound throttle. The slowest piece is the ban-list query, which we don't use. We want to skip it, and ideally skip the WHO too.

The blocker for skipping WHO is identity gating: `irc.state.nickToAccount(nick)` is currently the only thing populating account names for users already in-channel at bot startup. `account-notify` only fires on state *changes*; idle identified users would be invisible to the bot indefinitely.

The portable IRCv3-correct fix is to read `msg.server_tags['account']` (the `account-tag` capability, already requested by Limnoria). It rides on every PRIVMSG/NOTICE/TAGMSG from an identified user — the account arrives *with* the message, so there's no "idle user" gap for any gating decision tied to an incoming command.

## Plan in one paragraph

Introduce a **two-layer** resolver `_account_from_msg(irc, msg)`: `msg.server_tags['account']` → `irc.state.nickToAccount(nick)` → `None`. (No ircdb hostmask layer — that path silently promotes unidentified users to the `registered` tier; owner/admin/trusted already use `ircdb.checkCapability(prefix, …)` separately and stay untouched.) Migrate the 5 sender-side call sites. Capture the requesting account onto task rows so delivery-time logging doesn't need a late lookup. Leave `%usage <nick>` on `nickToAccount` as best-effort (it's already best-effort today). Then we can strip `MODE +b` unconditionally, and strip auto-WHO conditionally on both `account-tag` AND `extended-join` being ACK'd.

## Resolver semantics (precise)

```
def _account_from_msg(irc, msg) -> str | None:
    # Layer 1: account-tag (rides on every PRIVMSG/NOTICE/TAGMSG from identified user)
    tag = msg.server_tags.get('account')
    if tag:
        return tag
    # Layer 2: Limnoria's session cache (populated by account-tag ingest, account-notify, extended-join, WHO)
    nick = ircutils.nickFromHostmask(msg.prefix)
    try:
        cached = irc.state.nickToAccount(nick)
    except (KeyError, AttributeError):
        return None
    return cached  # may be None if user logged out (account-notify '*' sets it None) — terminal, not a fall-through
```

`None` is a terminal answer in both layers. A user known-logged-out is *not* the same as unknown, but for our gating purposes both are "unregistered" — that matches the current `_resolve_tier` semantics at `plugin.py:1471-1474`.

## What changes (rough)

- **`plugin.py`**: new `_account_from_msg` helper; refactor `_require_account` (1213), `_run_preflight` both branches (1255, 1274), `_resolve_tier` (1471), and the `_get_identity`/`_resolve_nick_to_identity` chain (1083, 1203). Two callers without a `msg` in scope (line 620 task delivery, line 2342 `%usage <nick>`) keep using `nickToAccount` and are documented as best-effort.
- **DB**: add nullable `account TEXT` column on the tasks table. Capture account at request submission. Delivery-time logging reads it directly. **NULL means "user wasn't identified at submission time"** — at delivery, fall back to nick (existing behavior). No backfill needed.
- **Tests**: extend the existing `mock_irc` fixture in `plugins/llm/tests/conftest.py:103` to provide a single account-resolution shape testable with one parameter override. Avoids per-test `server_tags` stubs across ~50 sites; future tests inherit it.
- **Speed change** (separate PRs): patch `supybot.irclib.Irc.doJoin` from the LLM plugin's `__init__`. Two distinct gates — see Sequencing.

## Risks / things that scare me

1. **Servers without `extended-join`.** `account-tag` covers PRIVMSG-class messages but NOT JOINs. JOINs need `extended-join` to carry account info. So the WHO drop must be gated on **both** caps being ACK'd, not just `account-tag`. AfterNet's UnrealIRCd advertises both; older servers may not.
2. **`account-tag` populates `nicksToAccounts` as a side effect.** Per `irclib.py:789-790`, every tagged message updates the cache. So once a user sends one tagged message, layer 2 (`nickToAccount`) works for them too — including from non-message contexts (timers, `%usage`). This makes the "user idle since bot start" gap narrower than it sounds: it only persists until their first message.
3. **Tests.** Conftest-level fixture refactor is mechanical but touches many files. Sequence the resolver PR first behind a stable mock signature; test churn becomes additive.
4. **Schema migration.** Nullable column, no backfill. Trivial. Standard migration tooling.
5. **Casemap nit.** `_maybe_migrate_nick` at `plugin.py:1102` does `.lower()` for nick comparison. Wrong for RFC1459 (`{}|` are lower of `[]\\`), happens to work on AfterNet because no users have those characters. **Decision:** defer to a separate cleanup; not in scope for this migration.
6. **Rollback.** The auto-WHO drop needs a config flag, not a code revert. Add `supybot.plugins.LLM.skipAutoWhoOnJoin` (default True when both caps ACK'd, settable False for emergency disable). The `MODE +b` drop is unconditional — no flag needed; nothing reads ban state.

## Open questions for the human (not the reviewer)

1. Are you OK with three PRs (resolver, `MODE +b` drop, auto-WHO drop) sequenced over a few sessions, or want it bundled?
2. Migration tooling — is there an existing pattern for adding a column to the tasks table, or do I need to wire one up?
3. Casemap fix in `_maybe_migrate_nick` — defer or include?

## Sequencing (three PRs)

1. **PR 1: Identity resolver + tasks.account column.** Adds `_account_from_msg`, migrates the 5 sender-side call sites, adds nullable `account` column, captures at task submission. Shared `mock_irc` fixture extension. **No behavior change visible to users** — both old and new code paths produce the same result while the resolver is wired in. Tests and gating still work the same.
2. **PR 2: Drop `MODE +b` ban-list query.** Tiny `__init__` monkey-patch on `supybot.irclib.Irc.doJoin`. No IRCv3 cap replaces ban-list lookup; we just don't use it. Verify nothing reads `state.channels[c].bans` first (`grep -rn '\.bans' plugins/`). Unconditional. Saves ~1s × channel count.
3. **PR 3: Drop auto-WHO conditionally.** Same monkey-patch site. Gate on `'account-tag' in capabilities_ack AND 'extended-join' in capabilities_ack`. Otherwise leave WHO in place. Add `skipAutoWhoOnJoin` config flag for emergency disable. Saves another ~1s × channel count.

PRs 2 and 3 are independent and can land in either order, but PR 1 must land first.

## Non-goals

- Don't touch owner/admin/trusted gating — uses `ircdb.checkCapability(prefix, …)`, hostmask-based and unrelated.
- Don't add ircdb hostmask matching to the identity resolver. (Reviewer caught this — would silently promote unidentified users to `registered` tier.)
- Don't try to auto-populate `nicksToAccounts` for idle users via WHOIS-on-demand. Account-tag obsoletes that need.
- Don't parse hostnames for account names. AfterNet's `<account>.users.afternet.org` cloak is context, not a portable signal.
- Don't fix the casemap nit in `_maybe_migrate_nick` here. Separate cleanup.
- Don't change JOIN-handling code paths. We have none today; `extended-join` is forward-looking only.

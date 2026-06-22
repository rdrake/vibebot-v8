# fc42 Taste-Tuned Verse Exemplars — Design

**Date:** 2026-06-21
**Status:** Approved + red-teamed (34 confirmed findings folded; ready for implementation plan)
**Topic:** Mine fc42's positive-taste signal from channel logs and inject curated
style exemplars into the verse system prompt.

> **Red-team note (2026-06-21):** An 8-dimension adversarial review (find →
> verify-against-code) raised 39 findings, 34 confirmed (1 HIGH, 15 MED, 18 LOW).
> All are folded below. Headline corrections: storage moved from a
> delimiter-joined `registry.String` to a `registry.Json` list (kills
> quote-stripping/delimiter-collision); a new `store.all_active_entity_names()`
> roster accessor (canon-only would match almost nothing); injection-time
> exemplar sanitization (prevents scene-marker forgery); honest poisoning framing
> (curated exemplars *are* prior output, just human-gated); and bot.conf-edit as
> the primary deploy path (IRC `@config` truncates at 512 bytes).

---

## Goal

Bias `#afternet` verse generation toward the prose that the channel's sharpest
critic (fc42) actually likes, by injecting a small set of **curated style
exemplars** — lines fc42 demonstrably approved of — into the verse system
prompt. fc42 benchmarks verse output ~5:1 against raw grok, so his taste is the
target signal worth optimizing for. This is a quality/“voice” improvement to
verse, not a new subsystem.

## Decisions (locked during brainstorm + red-team)

1. **Offline mine → human-curated.** A batch tool produces candidates; a human
   approves what becomes live. No live capture, no auto-injection, no self-tuning
   loop. Cheapest, safest, reversible.
2. **Verse only.** Exemplars tune the verse path (`build_verse_system_prompt`).
   The `@ask` chat path and `assistantSystemPrompt` overlay are untouched.
3. **Mining key = re-pastes + explicit praise; `lol` is NOT used.** `lol` is
   overwhelmingly football/TV/banter noise and sometimes *critical* of the bot.
4. **Storage = a new per-channel `registry.Json` key** (`verseStyleExemplars`,
   a JSON list of strings). No schema migration; round-trips quote/punctuation-laden
   prose cleanly (unlike a delimiter-joined `String`). Deployed by editing
   `bot.conf` while stopped.

## Log-reality findings (motivation for decision 3)

From 9,460 fc42 lines across `#afternet` ChannelLogger logs:
- **748** lines contain `lol`/`lmao`, mostly real-world reactions (*"lol the ref
  is uzbekistan"*); when aimed at the bot a `lol` can be a complaint (*"lol rdrake
  its gone back to one sentence replies"*). → poor key.
- fc42 **verbatim-reposts verse lines he loves** (gold, self-contained) — e.g.
  *"…flatulent finn and ripping robert… lettin' off a perfectly timed duet that
  makes the leaves on the nearest tree turn yellow."*
- fc42 gives **explicit praise** (~63 lines) — *"haha this is a good one"*, *"i
  love it when it said earlier that the stinky lads will either rule the country
  or set it on fire"*, *"amazing"*. NB these also fire on football (*"croatia are
  amazing"*, *"he is literally the perfect striker"*) → praise candidates are
  **always** `needs_review=True`.

Long verse output is posted in-channel as a teaser + pastebin link, so fc42's
full re-pastes come from the pastebin — fine, the pasted text *is* the exemplar.

---

## Architecture

```
ChannelLogger logs ──► offline miner ──► candidate review file ──► human curates
   (+ store.all_active_entity_names())                                    │  edits
                                                                          ▼  bot.conf
build_verse_system_prompt(..., style_exemplars=[...]) ◄── reads ── verseStyleExemplars (Json)
   └─► sanitizes + caps + injects a "match this energy" block in the verse prompt's
       byte-stable region (after persona/roster, before VERSE_SCENE_MARKER)
       ─► taste-biased verse output
```

Three isolated units:

| Unit | Responsibility | Depends on |
|------|----------------|------------|
| **Miner** | `log files + entity-name roster → ranked candidate file` (offline, read-only) | log files; `store.all_active_entity_names()` |
| **Config key** | Stores curated exemplars per channel as a JSON list | supybot `registry.Json` |
| **Injection** | Reads the list, sanitizes + caps + renders an exemplar block into the verse prompt | config key; `build_verse_system_prompt` |

Nothing runs in the live bot except the prompt read. With the key empty (default
`[]`), the new param defaults to `()` and the render is skipped, so the verse
prompt is **byte-for-byte identical to today**. (Note: `build_verse_system_prompt`'s
output is *not* the leading bytes of the final system content — the verse caller
at `plugin.py` prepends the channel overlay; see Component 3.)

## Component 1 — The miner

**Location:** a new standalone module `plugins/llm/src/llm/verse/taste_mine.py`
with a thin `if __name__ == "__main__"` CLI. (NB: `verse/purge.py` is an
operational *library* with **no** CLI — it is a structural precedent for "offline
read-mostly verse tool," not for a CLI; the miner adds its own argparse entry.)

**Core is a pure function** for testability:
`extract_candidates(log_lines, entity_names, *, min_repaste_chars=120) -> list[Candidate]`,
`Candidate(text, kind, source_date, source_line, needs_review)`. The CLI wires
live inputs: reads log files; pulls `entity_names` from
**`store.all_active_entity_names()`** (read-only; see Component 1a); writes the
review file.

**Log parsing.** ChannelLogger lines are `ISO_TS<2 spaces><body>`. The body is
one of: `<nick> message` (privmsg), `* nick message` (CTCP ACTION / `/me`), or a
system line `*** …`. Parsing rules:
- Recognize **both** `<nick> …` and `* nick …` as fc42 utterances (fc42 re-pastes
  and reacts via `/me` too); skip `*** …` system lines and any `-nick- …` notices.
- Handle empty/degenerate bodies (`<fc42>` with no trailing text) without error.
- Read with `encoding="utf-8", errors="replace"` (logs contain stray `�`); never
  crash on a malformed line — skip and continue.

**Re-paste detector (primary).** An fc42 message is a re-paste candidate when:
- length ≥ `min_repaste_chars` (default 120), AND
- it is narrative prose (not an addressed command like `grok …`, not predominantly
  a URL), AND
- it mentions ≥1 active entity name (see entity-matching below).

The matched line text becomes the exemplar (`needs_review=False`). *Accepted
limitation:* re-pastes of a brand-new character not yet in the roster, and short
gems (<120 chars), are missed; the human may add those by hand during curation.

**Praise detector (secondary).** An fc42 message matching the praise wordlist
(`good one`, `amazing`, `brilliant`, `genius`, `love it`, `so good`, `this is
gold`, …; tunable) is processed as follows — and **every praise-derived candidate
is `needs_review=True`** (the wordlist also fires on football/banter):
- Inline form: a locator regex such as `r"when it said (?:earlier |that |it |the )*(.+)"`
  captures `<X>`, then a **leading stopword run** (`earlier, that, it, the, when,
  said, a, an`) is stripped from `<X>`. The kept span must itself contain ≥1
  roster entity (same matcher as re-paste); otherwise fall through to bare form.
  *(Test: the flagship line "…love it when it said earlier that the stinky lads
  will either rule the country…" must yield a candidate STARTING at "the stinky
  lads", not "earlier that…".)*
- Bare form (no inline content): attach the nearest preceding non-fc42,
  non-system line as the candidate; `needs_review=True`.

**Entity matching.** The miner **reuses `store.match_entities_in_text(text)`** for
parity with prod retrieval (single source of truth), needing only its truthiness
("≥1 entity?") — it must NOT depend on which entities or their order. Documented
inherited quirks: names/aliases ≤2 chars are skipped; a name equal to a stoplist
word matches only when capitalized; internal whitespace/punctuation is matched
literally. **Pre-normalize** whitespace runs to single spaces in both the line and
names (`re.sub(r"\s+", " ", s)`) so `"stinky   lads"` matches `"stinky lads"`.
*Short common-word proper-noun guard:* for a short single-token name not in the
store stoplist (e.g. `Ghost`, `Harry`), require capitalized whole-word usage **or**
≥2 roster names co-occurring before counting it a hit (so `"ghost of a chance"` /
`"harry kane"` don't false-match). The `"Mr. Pringle"` vs `"mr pringle"` period
case is accepted-as-miss (a human curates).

**Post-processing.** Dedup with **strong normalization** (lowercase, collapse
whitespace, strip surrounding/most punctuation and trailing ellipses) so pastebin
re-paste variants collapse; rank by (entity-count, length); cap candidate count.
**Reject any candidate whose text contains the storage delimiter/control sentinel**
(see Component 2) so storage can never be corrupted. Emit a **review file**
(markdown) grouped by kind, each candidate with provenance (`date`, `kind`,
`needs_review`, source line) AND a ready-to-paste JSON array of the auto-trusted
(non-`needs_review`) candidates, so curation is copy-edit-then-paste.

The miner is **strictly read-only**: reads log files + the verse store; writes
only the review file.

### Component 1a — new store accessor

Add to `plugins/llm/src/llm/verse/store.py` near the other `list_*` accessors:

```python
def all_active_entity_names(self) -> list[str]:
    """Every active entity's name (any kind) — the miner's match roster.
    Read-only, deterministic order."""
    with self.read_connection() as conn:
        return [r[0] for r in conn.execute(
            "SELECT name FROM entities WHERE status='active' ORDER BY name COLLATE NOCASE"
        )]
```

The miner MUST use this (full active set). It MUST NOT use `list_canon_entities()`
— that returns only pinned/`author_locked` rows (~16 on prod; `author_locked=0`),
which would silently exclude the auto-created NPCs the miner targets (stinky lads,
kacky kyle, ripping robert) and produce a near-empty candidate list.

## Component 2 — Storage (`verseStyleExemplars`)

A new **per-channel** key, `registerChannelValue(LLM, "verseStyleExemplars",
registry.Json([], _(...)))` (verified available in this Limnoria; `registry.Text`
is not), default `[]` — a JSON list of exemplar strings. `registry.Json`
round-trips quote/apostrophe/punctuation-laden prose cleanly through the
`bot.conf` serialize/load cycle, eliminating the `registry.String` quote-stripping
and delimiter-collision failure modes entirely.

**Deploy path (primary): edit `bot.conf` while the bot is stopped** (per the
bot.conf-edit-order rule), pasting the miner's ready-made JSON array into
`supybot.plugins.LLM.verseStyleExemplars.#afternet`, then start. The miner emits
exactly this array. The `@config channel #afternet plugin.llm.verseStyleExemplars
…` IRC command is a **fallback for tiny edits only**: a single IRC line is capped
at 512 bytes including the `:nick!user@host PRIVMSG …` prefix (~98 bytes on
AfterNet), so a near-`MAX_EXEMPLAR_CHARS` value is silently truncated server-side
with a misleading success reply. (Curation is a channel-admin action — the trust
boundary; miner candidates are advisory.)

## Component 3 — Injection into the verse prompt

`build_verse_system_prompt` (`avatar.py:471`, called from `plugin.py:2566`) gains a
**keyword-only** parameter `style_exemplars: list[str] = ()` (keyword-only so the
~existing call sites stay green; default `()` ⇒ render skipped ⇒ byte-identical
output). The verse caller reads `verseStyleExemplars` for the channel and passes
the list in.

**Render (only when non-empty):**
1. **Sanitize each exemplar** (at render, the trust boundary — not only at
   curation): collapse all interior whitespace incl. `\n\r\t\v\f` to single spaces
   via `" ".join(ex.split())`; then **drop or neutralize** any exemplar that still
   contains the literal `VERSE_SCENE_MARKER` ("In play right now:") or begins with
   `"Scene:"` / `"- "`. This structurally prevents a newline-bearing exemplar from
   forging the scene marker or a fake Scene line inside the prefix.
2. **Cap:** at most `MAX_EXEMPLARS` (5) and `MAX_EXEMPLAR_CHARS` (~600) total;
   truncate the list to satisfy both.
3. **Render** a short labeled block placed in the **byte-stable region of the verse
   prompt — after persona/roster, immediately before `VERSE_SCENE_MARKER`**:
   ```
   The channel's sharpest critic singled these lines out as the good stuff —
   match this voice and energy; never copy them verbatim:
   - <exemplar 1>
   - <exemplar 2>
   ```

**Caching:** the exemplar block is static per-turn (persona, roster, exemplars all
change only on curation/canon edits, never per message), so the provider
prefix-cache stays warm turn-to-turn exactly as today; the only varying content
(the scene/message) still follows the marker. **Setting/changing the key is a
one-time cold cache-invalidation for that channel** — acceptable and expected.

The block **layers with** the verse prefix and channel-energy guidance; it never
replaces them (verse inherits, not overrides, the overlay energy). Short by design
(keep-the-verse-prompt-short).

## Poisoning / quality — honest framing

Curated exemplars **are prior model output fed back into the prompt** — the very
pattern behind the historical verse quality-collapse. The mitigation is not
"this isn't feedback" (it is); it is that the feedback is **human-selected,
hard-capped, static, and screened**, versus the collapse mode which was *automatic
self-imitation of recent raw output*. Concretely:
- The exemplar block lives in the **system prompt**, which is **invisible to both
  runtime self-poisoning guards** (they scan conversation history / recent output,
  not the system prompt). Curation is therefore the *only* gate — keep it tight.
- **Curation MUST screen out** any line that matches the denial-regex or the
  degraded-output patterns: a "liked" vivid line that happens to phrase like a
  refusal could, if imitated, trip `_is_verse_denial` and burn the denial-retry
  budget (stripping the good reply from history). The review file flags candidates;
  the curator drops anything denial-shaped.
- `"never copy them verbatim"` is a weak instruction in an imitation-prone slot;
  the real defense is the **tight cap + curation**, not the instruction. Keep
  ≤5 short exemplars.

## Error handling & guards (summary)

- **Miner:** tolerant parsing (`errors="replace"`, skip system/notice/malformed);
  read-only; empty roster or logs → empty candidate list (not an error); reject
  candidates containing the storage sentinel.
- **Injection:** empty list → no-op (byte-identical); sanitize + cap so a
  mis-curated/oversized/newline-bearing value cannot bloat or restructure the
  prompt.
- **Poisoning:** curated-only + capped + static + denial/degraded-screened.

## Testing

- **Miner (pure-function), fixtures covering:** (a) a `<fc42>` verbatim re-paste
  naming a roster entity → kept, `needs_review=False`; (a2) the same as a `* fc42`
  ACTION line → kept; (b) the flagship inline-praise line → candidate STARTS at
  "the stinky lads", `needs_review=True`; (c) a bare praise line → attaches
  preceding line, `needs_review=True`; (d) football/banter `lol` and `"croatia are
  amazing"` → rejected (or `needs_review` and clearly non-prose); (e) addressed
  `grok …` command and a URL line → rejected; (f) an encoding-garbled line → no
  crash; (g) empty/degenerate `<fc42>`/`-notice-`/`*** system` lines → skipped;
  (h) entity false-match guards: `"Ghost"` must NOT match `"ghost of a chance"`,
  `"Harry"` must NOT match `"harry kane"` absent a co-occurring roster name; (i)
  whitespace-run name `"stinky   lads"` matches `"stinky lads"`; (j) dedup collapses
  punctuation/ellipsis near-duplicates; (k) a candidate containing the storage
  sentinel is rejected.
- **Store:** `all_active_entity_names()` returns the full active set (not
  canon-only), deterministic order.
- **Storage round-trip:** a `registry.Json` value containing quote-wrapped and
  apostrophe-laden exemplars round-trips through `set()` + `bot.conf` write/read
  unchanged (the case that broke `registry.String`).
- **Injection:** key empty/`()` → output **byte-identical** to the no-exemplar
  baseline (deterministic compare of the prefix up to `VERSE_SCENE_MARKER` while
  only the per-turn message changes; mirror the existing
  `test_verse_prompt_roster.py` harness); 2 exemplars → block present, labeled,
  positioned after roster + before the marker, existing prefix intact; an exemplar
  containing `"\nIn play right now:\nScene: …"` → renders as exactly one `- …`
  line, output has exactly one `VERSE_SCENE_MARKER`, no forged Scene line; 10
  oversized exemplars → capped to `MAX_EXEMPLARS`/`MAX_EXEMPLAR_CHARS`.
- **Config:** `verseStyleExemplars` registered, per-channel, default `[]`.
- **Call-site safety:** all existing `build_verse_system_prompt` callers still pass
  (keyword-only param, default skips render).

## Out of scope (YAGNI — explicitly cut)

Live reaction capture, a review/approval command, auto-injection, a verse-store
table for exemplars, automated refresh/scheduling, pastebin/answer-HTML
cross-matching for bare praise (best-effort preceding-line attribution only), and
any chat-path / `assistantSystemPrompt` tuning.

## Rollout

1. Implement (store accessor + miner + config key + injection) behind the
   default-empty key — ships inert (byte-identical verse).
2. Run the miner against the prod logs (pull logs local, or run host-side
   read-only), review candidates, hand a curated set to fc42/rdrake; drop any
   denial-shaped line.
3. Deploy: stop the bot, paste the curated JSON array into
   `supybot.plugins.LLM.verseStyleExemplars.#afternet` in `bot.conf`, start;
   observe a few verse turns.
4. Re-run the miner periodically as the corpus grows; re-curate.

# fc42 Taste-Tuned Verse Exemplars — Design

**Date:** 2026-06-21
**Status:** Approved (brainstorm complete; ready for implementation plan)
**Topic:** Mine fc42's positive-taste signal from channel logs and inject curated
style exemplars into the verse system prompt.

---

## Goal

Bias `#afternet` verse generation toward the prose that the channel's sharpest
critic (fc42) actually likes, by injecting a small set of **curated style
exemplars** — lines fc42 demonstrably approved of — into the verse system
prompt. fc42 benchmarks verse output ~5:1 against raw grok, so his taste is the
target signal worth optimizing for.

This is a quality/“voice” improvement to verse, not a new subsystem.

## Decisions (locked during brainstorm)

1. **Offline mine → human-curated.** A batch tool produces candidates; a human
   approves what becomes live. No live capture hooks, no auto-injection, no
   self-tuning loop. Cheapest, safest, fully reversible; sidesteps the
   self-poisoning failure mode seen historically with feeding model output back
   into prompts.
2. **Verse only.** Exemplars tune the verse path (`build_verse_system_prompt`).
   The normal `@ask` chat path and its `assistantSystemPrompt` overlay are left
   untouched. Smallest blast radius, highest-value target.
3. **Mining key = re-pastes + explicit praise; `lol` is NOT used.** Log analysis
   (below) showed `lol` is overwhelmingly noise (football/TV/banter) and
   occasionally *critical* of the bot. The real, clean signals are: fc42
   reposting verse prose verbatim, and explicit praise lines.
4. **Storage = a new per-channel config key** (`verseStyleExemplars`). No schema
   migration; matches the existing `verseModel` / `assistantSystemPrompt`
   registry pattern; trivially editable and reversible via `@config`.

## Log-reality findings (motivation for decision 3)

From 9,460 fc42 lines across `#afternet` ChannelLogger logs:

- **748** lines contain `lol`/`lmao`, but they are mostly real-world reactions
  (`"lol the ref is uzbekistan"`, `"lol saudi arabia are about to get DICKED"`).
  When a `lol` *is* aimed at the bot it can be a complaint
  (`"lol rdrake its gone back to one sentence replies"`). → `lol` is a poor key.
- fc42 **verbatim-reposts verse lines he loves** (gold, self-contained — the
  pasted text *is* the liked line), e.g. *"…flatulent finn and ripping robert,
  the year 7 duo, protest loudly… lettin' off a perfectly timed duet that makes
  the leaves on the nearest tree turn yellow."*
- fc42 gives **explicit praise** (~63 lines), often naming the bot output:
  *"haha this is a good one"*, *"i love it when it said earlier that the stinky
  lads will either rule the country or set it on fire"*, *"amazing"*.

Note: long verse output is posted in-channel as a teaser + pastebin link, so
fc42's full re-pastes are read from the pastebin. This is fine — the pasted text
itself is the exemplar; no attribution back to a logged bot line is required.

---

## Architecture

```
ChannelLogger logs ──► offline miner ──► candidate review file ──► human curates
   (+ verse entity roster)                                              │
                                                                        ▼
build_verse_system_prompt ◄── reads ── verseStyleExemplars (channel config key)
   └─► injects a short, capped "match this energy" exemplar block into the
       stable prefix (before VERSE_SCENE_MARKER) ─► taste-biased verse output
```

Three isolated units with clean interfaces:

| Unit | Responsibility | Depends on |
|------|----------------|------------|
| **Miner** | `logs + entity roster → ranked candidate file` (offline, read-only) | log files; entity-name list |
| **Config key** | Stores curated exemplars per channel | supybot registry |
| **Injection** | Reads the key, renders a capped exemplar block into the verse prompt | config key; `build_verse_system_prompt` |

Nothing runs inside the live bot except the prompt read. With the key empty
(the default), behavior is byte-for-byte identical to today.

## Component 1 — The miner

**Location:** `plugins/llm/src/llm/verse/taste_mine.py` (importable module +
`__main__` CLI, mirroring the operational `verse/purge.py` precedent). Tested.

**Core is a pure function** for testability:
`extract_candidates(log_lines, entity_names, *, min_repaste_chars=120) -> list[Candidate]`
where `Candidate` carries `(text, kind, source_date, source_line, needs_review)`.
The CLI wires the live inputs (reads log files; pulls `entity_names` from the
verse store, read-only) and writes the review file.

**Log parsing.** ChannelLogger format is `ISO_TS<2 spaces><nick> message`, with
system lines like `*** … sets mode` / `*** … has joined`. Parsing:
- Skip system (`***`) lines and non-`<nick>` lines.
- Read files with `encoding="utf-8", errors="replace"` (logs contain stray `�`
  from a latin-1/utf-8 mix); never crash on a bad line.

**Re-paste detector (primary).** An fc42 message is a re-paste candidate when:
- length ≥ `min_repaste_chars` (default 120), AND
- it is narrative prose: not an addressed command/URL (excludes lines starting
  with a bot name like `grok …`, lines that are predominantly a URL), AND
- it mentions ≥1 known verse entity name (case-insensitive, word-boundary match
  against the roster: `stinky lads`, `kacky kyle`, `ripping robert`, …).

The matched line text becomes the exemplar verbatim (`needs_review=False`).

**Praise detector (secondary).** An fc42 message matches a praise wordlist
(`good one`, `amazing`, `brilliant`, `genius`, `love it`, `so good`,
`this is gold`, …; tunable constant). Then:
- If praise is inline (`love it when it said <X>` / `when it said that <X>`),
  extract `<X>` as the exemplar (`needs_review=False`).
- Otherwise, attach the nearest preceding non-fc42, non-system, non-noise line as
  the candidate and set `needs_review=True` (best-effort; the human decides).

**Post-processing.** Normalize whitespace/case for dedup; drop near-duplicates;
rank by (entity-count, length); cap to a configurable max. Emit a **review file**
(markdown) grouping candidates by kind, each with provenance
(`date`, `kind`, `needs_review`, original source line) so the human can curate
confidently.

The miner is **strictly read-only**: it reads log files and the verse store; it
writes only the review file. It never mutates prod data.

## Component 2 — Storage (`verseStyleExemplars`)

A new **per-channel** registry key, `registerChannelValue(LLM,
"verseStyleExemplars", registry.String("", _(...)))`, default `""` (added in
`plugins/llm/src/llm/config.py` alongside the other verse keys).

The value holds the curated exemplars joined by a **single-line-safe delimiter**
(` ||| `), so it survives the registry/`bot.conf` round-trip regardless of
newline handling. The injection code splits on the delimiter, trims, and drops
empties.

Curated exemplars are set via `@config channel #afternet
plugin.llm.verseStyleExemplars <value>` (or offline while the bot is stopped, per
the bot.conf-edit-order rule). The miner's output is a review aid, **not** wired
directly to the key — a human always picks.

## Component 3 — Injection into the verse prompt

In `build_verse_system_prompt` (`plugins/llm/src/llm/verse/avatar.py`, currently
~471–530; `VERSE_SCENE_MARKER` at avatar.py:22):

- Read `verseStyleExemplars` for the channel. If empty → **no-op** (prompt
  byte-identical to today).
- If non-empty: split on the delimiter, trim, then **hard-cap**: at most
  `MAX_EXEMPLARS` (default 5) and at most `MAX_EXEMPLAR_CHARS` (default ~600)
  total; truncate the list to stay under both. Render a short labeled block,
  placed in the **stable prefix, before `VERSE_SCENE_MARKER`**:

  ```
  The channel's sharpest critic singled these lines out as the good stuff —
  match this voice and energy; never copy them verbatim:
  - <exemplar 1>
  - <exemplar 2>
  ```

The block **layers with** the existing verse prefix and channel-energy guidance;
it never replaces them (consistent with the rule that verse inherits, not
overrides, the channel overlay energy). Keeping it short respects the
keep-the-verse-prompt-short constraint and avoids prompt bloat.

`build_verse_system_prompt`'s signature gains the exemplars input (passed by its
caller, which already has channel context for the existing `registryValue`
reads), keeping the function itself free of plugin/registry coupling for
testability.

## Error handling & guards

- **Miner:** tolerant parsing (`errors="replace"`, skip malformed/system lines);
  read-only; empty roster or empty logs → empty candidate list, not an error.
- **Injection:** empty key → no-op; cap count + total length so a
  mis-curated/oversized value can't bloat or destabilize the prompt.
- **Poisoning:** curated-only + capped + static. No raw model output is ever fed
  back automatically; this is the structural defense against the self-imitation
  quality-collapse seen previously.
- **Overlay coexistence:** additive to the verse stable prefix; does not touch
  `assistantSystemPrompt` or the chat path.

## Testing

- **Miner (pure-function):** fixture log lines containing (a) a verbatim verse
  re-paste mentioning a roster entity, (b) an inline-praise line, (c) a bare
  praise line (→ `needs_review=True`, grabs preceding line), (d) football/TV
  `lol` noise, (e) an addressed `grok …` command, (f) a URL line, (g) an
  encoding-garbled line. Assert: re-paste + inline-praise kept; noise/command/URL
  rejected; dedup works; entity matching is word-boundary; no crash on garbled
  input.
- **Injection:** `build_verse_system_prompt` with the key (i) empty → output
  byte-identical to the no-exemplar baseline; (ii) set with 2 exemplars → block
  present, correctly labeled, positioned before `VERSE_SCENE_MARKER`, existing
  prefix intact; (iii) set with 10 oversized exemplars → capped to
  `MAX_EXEMPLARS`/`MAX_EXEMPLAR_CHARS`.
- **Config:** `verseStyleExemplars` registered, per-channel, default `""`.

## Out of scope (YAGNI — explicitly cut)

- Live reaction capture, a review/approval command, auto-injection.
- A verse-store table for exemplars (avoids a schema migration).
- Automated refresh / scheduling of the miner.
- Pastebin/answer-HTML cross-matching to attribute bare praise (best-effort
  preceding-line attribution only, flagged for human review).
- Tuning the chat path or `assistantSystemPrompt`.

## Rollout

1. Implement (miner + config key + injection) behind the default-empty key — ships
   inert.
2. Run the miner against the prod logs (pull logs local, or run host-side
   read-only), review candidates, hand a curated set to fc42/rdrake.
3. Set `verseStyleExemplars` for `#afternet`; observe a few verse turns.
4. Re-run the miner periodically as the corpus grows; re-curate.

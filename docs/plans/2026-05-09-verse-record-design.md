# Forest-verse: member-driven worldbuilding (`verse_record` + auto-entity aging) — Design

**Goal:** Let any opted-in verse member narrate canon involving entities other than themselves — "stinky dan threw a guff grenade at Andrew" gets recorded immediately, no proposal queue, with auto-creation of unknown actors. Junk entities age out so casual mentions self-clean.

**Architecture:**
- One new assistant tool `verse_record` (added to `make_verse_tool_specs()` in `verse/avatar.py`). Members get it whenever the four existing verse tools are routed.
- Dispatch branch in `dispatch_verse_tool_call` resolves names → entity ids (auto-create npcs as needed, in a single write transaction) and writes one `events` row through the existing `add_event` path.
- One new pure helper module `verse/aging.py` with `age_auto_created_entities(store, *, retire_after_days, now)` — soft-retires `auto_created='1'` entities whose `last_seen_ts` is older than the cutoff. Hooks into the existing `_run_compaction_pass` per channel; no new timer.
- Existing `verse/store.py` grows two query methods (`list_entities_with_attribute`, `find_entity_by_name_active_first`) and gains a `last_seen_ts` bump on every code path that references an entity (verse_record, loom apply, compaction digest).
- `plugin.py` extends the compaction outcome string and registers two registry keys (`verseAutoEntityRetireDays`, default 14; `verseAutoEntityMaxNamesPerCall`, default 8).

**Tech stack:**
- Python 3.13+ via `uv`, `pytest`, `ruff`, `ty`.
- Real SQLite in `tmp_path` for tests. **No DB mocks.**
- Tool-call dispatch via the existing `dispatch_verse_tool_call` envelope; no new LiteLLM wiring.

**Reference design:** `docs/plans/2026-05-07-forest-verse-design.md` (esp. §"Verse store schema", §"Avatar tool surface"); `docs/plans/2026-05-08-forest-verse-pr3.md` (compaction interaction). Codex review of the v0 sketch is summarised in §"Adversarial review notes" below.

**Working directory:** `.worktrees/verse-record` (branch `feat/verse-record`). Push to `main` is fine; CI + Docker are separate workflows — wait for both before restarting prod.

---

## Revisions

- **2026-05-09 v1** — initial draft after Codex review of the inline sketch in chat.
  Codex flagged one fatal (schema CHECK constraint on `events.source`) and seven significant issues (race on find-then-add, lookup precedence not implemented, avatar-first risk, `min_keep_references` × compaction interaction, loom doesn't bump `last_seen_ts`, `find_entity_by_name` doesn't filter retired, dangling refs after digest truncation). All addressed below; specifics in §6.

---

## 1. Tool surface

`make_verse_tool_specs()` (`verse/avatar.py:16`) gains a fifth entry:

```jsonc
{
  "type": "function",
  "function": {
    "name": "verse_record",
    "description":
      "Record an in-world event involving one or more named actors. "
      "Use whenever a member narrates events that aren't strictly about "
      "their own avatar (e.g. \"stinky dan threw a guff grenade at "
      "Andrew\"). Names that don't match an existing entity are "
      "auto-created as kind=npc. Only include actors central to the "
      "narrated event — do not capture every noun in the sentence.",
    "parameters": {
      "type": "object",
      "properties": {
        "summary": {
          "type": "string",
          "description":
            "What happened, in past tense, ≤200 chars. e.g. "
            "'stinky dan threw a guff grenade at Andrew'."
        },
        "subjects": {
          "type": "array",
          "items": { "type": "string" },
          "maxItems": 8,
          "description":
            "Names of actors central to the event. Up to 8."
        }
      },
      "required": ["summary"]
    }
  }
}
```

**Dropped from the v0 sketch (per Codex):**

- `objects: string[]` — items and places have downstream behavioural differences (movement, scene). v1 doesn't auto-create either; loom + manual `@verseopt`/proposals remain the only path. If members want to mention an item-like noun, they put it in the `summary` text and skip linking.
- `subjects` defaulting to multi-kind auto-create — v1 only auto-creates `npc`. An existing entity of any kind matches first via the lookup policy below; only when nothing matches do we mint an npc.

`maxItems: 8` matches `verseAutoEntityMaxNamesPerCall` (registry-tunable). The user has signalled "lots of entities" — 8 is generous for a single sentence and bounds high-cardinality flooding from runaway model output. Operators can lower; raising past 16 should be discouraged in the doc.

## 2. Lookup policy & precedence

Codex's most important call-out: today's `find_entity_by_name(name)` returns the first id ASC across **all kinds and statuses**, contradicting the spec'd `avatars > npcs > items > places` precedence. Either the spec is wrong or the impl is. We're doing both: a new helper, and an explicit decision on precedence.

**New method:** `VerseStore.find_entity_by_name_active_first(name) -> Entity | None`

```python
def find_entity_by_name_active_first(self, name: str) -> Entity | None:
    """Resolve a name with precedence avatar > npc > item > place,
    only over status='active' entities. Case-insensitive.

    Used by verse_record to bind narration to a real avatar when the
    name matches one, and to reuse npcs the verse has already met."""
```

Implemented as a single SQL with `CASE` ordering on `kind` so it stays one round-trip. Filters `status='active'` — so retired entities never get silently rehydrated by a new mention (Codex SIG #5).

The legacy `find_entity_by_name(name, kind=...)` stays as-is for `verse_act`'s movement/item lookups (those genuinely want a kind filter and don't care about precedence).

**Precedence decision: avatar-first stays.** Codex flagged the risk that "Andrew did X" silently links a real player's avatar to junk. Counterweight:
- Members deliberately opt in via `@verseopt in`; their avatar identity is durable via `avatar_link`. Ambient narration referencing them by nick is the *desired* behaviour ("Andrew laughed" should attach to Andrew's avatar so `verse_recall` and `@verseproposals` reflect what the verse says about Andrew).
- Avatar-first is what makes `verse_record` work for the Forest mode case where multiple humans co-narrate.
- Mitigation: when a name resolves to an avatar, **don't bump `last_seen_ts`** and **don't tag `auto_created`** (the avatar wasn't created by this call and isn't aging out). The avatar's identity is unaffected; only the event row links.

A per-channel registry key `verseRecordAvatarPrecedence` (default `True`) lets an operator disable avatar-first if a verse turns out to suffer from name collisions. Out of scope for v1 unless the test pass surfaces a real failure mode.

## 3. Dispatch

New branch in `dispatch_verse_tool_call` (`verse/avatar.py:383`):

```python
elif name == "verse_record":
    summary = (args.get("summary") or "").strip()[:200]
    if not summary:
        return ToolResult(error="summary required")
    raw_subjects = args.get("subjects") or []
    if not isinstance(raw_subjects, list):
        return ToolResult(error="subjects must be an array")
    max_names = ...  # passed in via dispatch context, see §5
    subjects = [s.strip() for s in raw_subjects[:max_names] if isinstance(s, str)]
    event_id = store.record_user_event(
        actor_avatar_id=avatar_id,
        summary=summary,
        subject_names=subjects,
        now=now,
    )
    return ToolResult(content=json.dumps({"event_id": event_id}))
```

The DB work is pushed into a new store method `record_user_event` so the **find-or-create-then-link** is atomic in one `write_transaction`:

```python
def record_user_event(self, *, actor_avatar_id, summary, subject_names, now):
    """Resolve subject_names to entity ids (auto-create as npc if
    unknown), bump last_seen_ts on each, and write the event row in
    one transaction. Returns the new event id.

    Race-safe: holds the write lock across find-or-create. Two
    concurrent verse_record calls for the same npc name will not
    create duplicates."""
    with self.write_transaction() as conn:
        ids = [actor_avatar_id]
        for raw in subject_names:
            entity = self._lookup_in_tx(conn, raw)        # active-first, precedence
            if entity is None:
                eid = self._add_entity_in_tx(conn, "npc", raw, "")
                self._set_attr_in_tx(conn, eid, "auto_created", "1")
            else:
                eid = entity.id
                if entity.kind != "avatar":               # don't tag avatars
                    self._set_attr_in_tx(conn, eid, "last_seen_ts", str(now()))
            ids.append(eid)
        return self._add_event_in_tx(
            conn, summary=summary, entity_ids=ids, source="avatar"
        )
```

**`source='avatar'` (not `'user_record'`).** Codex FATAL: `events.source` has a CHECK constraint to `('avatar','loom','crosspoll')` (`verse/schema.sql:43`). Adding a fourth value requires a migration; SQLite's `ALTER TABLE` doesn't drop a CHECK so the migration is a table-rebuild. **v1 reuses `'avatar'`** since the actor *is* the caller's avatar acting on the world. The provenance loss (can't audit user-record vs verse_act) is documented as a known v1 limitation; a follow-up PR can add a fifth source via a one-shot rebuild migration.

`_lookup_in_tx` calls the same SQL as `find_entity_by_name_active_first` but on the open connection. `_add_entity_in_tx` and `_set_attr_in_tx` are the existing private TX helpers.

## 4. Aging

New module `verse/aging.py`:

```python
def age_auto_created_entities(
    store, *, retire_after_days, now
) -> AgingOutcome:
    """Soft-retire auto_created='1' entities whose last_seen_ts is
    older than now - retire_after_days*86400. Returns
    AgingOutcome(scanned, retired, kept). Skips kind='avatar' entities
    defensively even if somehow tagged. retire_after_days<=0 disables."""
```

Single SQL: `SELECT id, last_seen_ts attribute FROM entities JOIN attributes WHERE auto_created='1' AND status='active' AND kind!='avatar'`. Iterate in Python; flip status with `set_status` (existing).

**`min_keep_references` is dropped from v1.** Codex SIG #2.1: lifetime event-counts are unreliable post-compaction (compaction deletes raw events). `last_seen_ts` is sufficient — entities that get re-mentioned stay alive; entities that don't, age out. Simpler, no compaction-interaction bugs.

**Heartbeat sources** (everywhere `last_seen_ts` is bumped):

1. `record_user_event` — for non-avatar subjects.
2. `apply_or_queue` (loom apply path, `verse/loom.py`) — when a loom proposal references an existing entity, bump it. Closes Codex SIG #2.2.
3. `replace_events_with_lore_digest` (`verse/compaction.py`) — when an entity survives the truncated `entity_ids` list (≤32) of a freshly-written digest event, bump its `last_seen_ts` to "now". Closes Codex SIG #4.2: the entity may have lost raw event references in compaction, but the digest IS the heartbeat. Entities truncated out of the digest are exactly the ones we want to consider for aging — they carry no recent signal.

Heartbeat-on-lookup is the simpler property than the v0 sketch's "count references over lifetime" rule, and it falls out cleanly from "every code path that touches an entity bumps its timestamp."

## 5. Configuration

Two new registry keys in `config.py`:

| Key | Type | Default | Scope | Description |
|---|---|---|---|---|
| `verseAutoEntityRetireDays` | int | `14` | per-channel | Days of no reference before auto-created entities retire. `0` disables sweep. |
| `verseAutoEntityMaxNamesPerCall` | int | `8` | per-channel | Hard cap on `subjects` array length. Tool spec advertises this; dispatch enforces. Increase past 16 only if you have a verse with regularly-cited large casts. |

The dispatcher needs `verseAutoEntityMaxNamesPerCall` at call time. Plumb it via the existing dispatch context (the closure built around `dispatch_verse_tool_call` in `plugin.py:_register_verse_tools`). One more parameter on the closure; no new injection point.

## 6. Adversarial review notes (Codex pass, 2026-05-09)

Each numbered item maps to a Codex finding from the v0 inline sketch.

| Finding | Severity | Resolution |
|---|---|---|
| `source='user_record'` violates CHECK | **FATAL** | §3: reuse `source='avatar'`. Document provenance-loss; follow-up to extend CHECK. |
| Find-then-add race | SIG | §3: single `write_transaction` in `record_user_event`. Test #5 (concurrent calls) covers. |
| `find_entity_by_name` doesn't implement precedence | SIG | §2: new `find_entity_by_name_active_first` with explicit `CASE WHEN kind='avatar' THEN 0 …` ordering. |
| Avatar-first risk | SIG | §2: kept, with mitigations (no `auto_created` tag, no `last_seen_ts` bump on avatars). Disable knob deferred. |
| `min_keep_references` × compaction | SIG | §4: dropped entirely. Last-seen-only policy. |
| Loom doesn't bump `last_seen_ts` | SIG | §4 heartbeat #2: wire `apply_or_queue` to bump on referenced entities. |
| `find_entity_by_name` doesn't filter active | SIG | §2: new helper filters `status='active'`. |
| `subjects`/`objects` can't distinguish item vs place | SIG | §1: dropped `objects` from v1; only npcs auto-create. |
| No `maxItems` flooding guard | SIG | §1, §5: `maxItems: 8` in spec, registry-tunable cap, dispatch enforces. |
| Soft-retire creates dangling refs in `who()`/`snapshot()` | SIG | Accepted as-is. Digest events keep the id; `who()`/`snapshot()` already filter `status='active'`. The id dangles only in JSON event payloads, which is the existing post-`@verseopt out` behaviour. Documented in operator guide. |
| Compaction `entity_ids` truncated to 32 → false retire | SIG | §4 heartbeat #3: `replace_events_with_lore_digest` bumps `last_seen_ts` on entities included in the digest. Entities truncated out *correctly* age (they're not load-bearing). Test #9 covers. |
| Digest `source='loom'` loses user_record provenance | MINOR | Accepted. v1 reuses `'avatar'` source so the loss is symmetric across paths. |
| Partial-failure swallowing in dispatcher | MINOR | `record_user_event` is one TX — either all subjects link or the whole call rolls back. No partial state. |

## 7. Tests

`tests/test_verse_record.py` (new):

1. `verse_record` with all-new names creates `kind='npc'` entities, links event with caller's avatar id first.
2. Existing avatar with the same name as a `subject` is linked instead of inventing an npc; the avatar gets neither `auto_created='1'` nor a `last_seen_ts` bump.
3. Existing npc with the same name is reused; `last_seen_ts` is bumped.
4. Repeated `verse_record` for the same npc keeps one entity row; `last_seen_ts` is the most recent value.
5. **Race**: two `verse_record` calls in parallel for the same new name produce one entity. (Use `threading` + the real SQLite write lock; two threads race the find-or-create.)
6. Retired entity with the same name as a subject is **not** rehydrated — a new active npc is created instead. Closes the active-filter gap.
7. `subjects` longer than `verseAutoEntityMaxNamesPerCall` is truncated, not rejected. The first N are processed.
8. Empty `summary` returns a tool error; no DB writes.

`tests/test_verse_aging.py` (new):

1. Auto-created npc with `last_seen_ts` past cutoff is retired.
2. Auto-created npc with `last_seen_ts` recent is kept.
3. Manually-created entity (no `auto_created='1'` attribute) is never touched, even past cutoff.
4. Avatar with `auto_created='1'` (defensive: shouldn't happen, but if it does) is never retired.
5. `retire_after_days=0` makes the helper a no-op.
6. **Compaction interaction**: auto-created npc with 3 raw event references runs through `replace_events_with_lore_digest`, the digest event includes its id, aging then runs — entity stays active because the digest bumped `last_seen_ts`. Closes the trickiest Codex finding.
7. **Compaction-truncation interaction**: same setup as #6 but with 40 entities so the digest truncates to 32 and excludes our npc; aging then retires it. Documents and verifies the intentional behaviour.

`tests/test_verse_record_wiring.py` (new, plugin-level):

8. `_run_compaction_pass` calls `age_auto_created_entities` once per `_verse_enabled_channels()` entry.
9. Aging exception in one channel doesn't abort the pass for others (mirror the existing `_run_compaction_pass` failure-isolation test).
10. The verse-tool dispatch closure passes `verseAutoEntityMaxNamesPerCall` from the registry to `dispatch_verse_tool_call`.

Existing test that's likely to break and needs update: `test_dispatch_verse_tool_call_unknown_op` — add `verse_record` to the known-ops list.

## 8. Operator UX

The compaction-pass reply (`@versecompact #foo` and the daily timer's log line) gets aging counts:

```
compaction outcome for #foo: compacted 12 events; aged 2 entities (kept 5)
compaction outcome for #foo: skipped (only 7 events; floor is 20); aged 0 entities (kept 0)
```

(Also fixes the cryptic `skipped_below_floor` from the chat — replaces enum-string with human text.)

Document `verse_record` in `docs/guide/operator/forest-verse.md`:
- "What members can record" section — example `vibebot stinky dan threw a guff grenade at Andrew` → resulting event row.
- "Auto-created npcs" section — visible in `@versedump`, reused on subsequent mentions, retire after `verseAutoEntityRetireDays` days of silence.
- "Why some entities disappear" troubleshooting — explains aging and how to bump `verseAutoEntityRetireDays` if a verse loses cast members the operator wanted to keep.

## 9. Out of scope (deliberately)

- **No `verse_unrecord`/edit-event tool.** Members can't delete or rewrite events. Aging is the only auto-cleanup; operators can do hand-surgery via `@versedump` + manual SQL if desperate.
- **No bulk un-retire UI.** Reanimating a retired npc on accident is high-cost; force the operator path.
- **No NER / kind inference.** All auto-creates are `npc`. Adding place/item inference is a follow-up if real verses develop noun-confusion patterns.
- **No rate-limit additions.** Existing tier rate-limits throttle the assistant flow; abuse is bounded there.
- **No new `events.source` value.** Reuse `'avatar'`. Provenance follow-up is a separate one-PR migration if it ever becomes load-bearing.
- **No deletion.** Soft-retire only. Audit trail preserved.

## 10. Estimated change

| Path | Lines |
|---|---|
| `plugins/llm/src/llm/verse/avatar.py` (tool spec + dispatch branch) | +35 |
| `plugins/llm/src/llm/verse/store.py` (record_user_event + lookup + 1 attr query) | +60 |
| `plugins/llm/src/llm/verse/aging.py` (new) | +50 |
| `plugins/llm/src/llm/verse/loom.py` (heartbeat in apply_or_queue) | +5 |
| `plugins/llm/src/llm/verse/compaction.py` (heartbeat in digest write) | +5 |
| `plugins/llm/src/llm/plugin.py` (compaction-pass hook + outcome string + dispatch closure) | +25 |
| `plugins/llm/src/llm/config.py` (two registry keys) | +20 |
| `plugins/llm/tests/test_verse_record.py` (new) | +200 |
| `plugins/llm/tests/test_verse_aging.py` (new) | +180 |
| `plugins/llm/tests/test_verse_record_wiring.py` (new) | +120 |
| `docs/guide/operator/forest-verse.md` (3 new sections) | +60 |
| `CHANGELOG.md` | +10 |

Total ~770 lines. One PR. No schema migration.

## 11. Implementation order

For a follow-up PR plan doc (`2026-05-09-verse-record-pr1.md`):

1. Store: `find_entity_by_name_active_first`, `list_entities_with_attribute`, `record_user_event` — TDD red-green for each.
2. Aging helper + tests.
3. Heartbeat wiring in loom + compaction (3 lines × 2 sites; tests already in test_verse_aging.py #6 and #7).
4. Tool spec + dispatch branch + tests.
5. Compaction-pass hook in plugin + outcome string + wiring tests.
6. Operator guide update.
7. CHANGELOG.

Each step is independently red-green-commit-able; the final wiring test is the integration gate.

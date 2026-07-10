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

- **2026-05-09 v2.4** — first-deploy smoke test (`vibebot, stinky dan threw a guff grenade at Andrew` in `#afternet`) revealed a prompt-engineering gap: `tools=23 tool_calls=0`. The verse system prompt frames the model as the avatar (`"You are <name>. Persona: …"`) and is silent on tool use; the model interprets in-channel narration prompts as creative-writing briefs and emits text directly. The tool description alone is not loud enough to overcome the persona framing.
  - Fix: extend `build_verse_system_prompt` to add a final "Tools" paragraph that names `verse_record` and tells the model to call it whenever the user narrates an in-world event involving named characters (other than the avatar's own action — those go through `verse_act`). Items / weapons / places stay in the summary as prose. Implemented in §1, tested in §7.
  - Out of scope: tuning the description text on the four legacy tools (`verse_act` / `verse_move` / `verse_look` / `verse_recall`) — they continue to rely on their own descriptions, unchanged.

- **2026-05-09 v2.3** — adversarial review (Codex pass on actual code) surfaced two heartbeat-scope drifts:
  1. **Loom heartbeat payload-key scope.** v2.0–v2.2 §4.2 #2 said "every entity_id referenced by the proposal payload" but the plan's snippet only scanned `payload.get("entity_ids")`. Applied `set_attribute` proposals reference an entity via `payload["entity_id"]` (singular); `add_relation` via `payload["from_id"]` and `payload["to_id"]`. Implementations must scan ALL entity-id-bearing payload keys per op:
     - `add_event` → `payload["entity_ids"]` (list[int])
     - `set_attribute` → `[payload["entity_id"]]`
     - `add_relation` → `[payload["from_id"], payload["to_id"]]`
     - `add_entity` → `[]` (creating a new row; no refs to bump)
     - `crosspoll_seed` → `payload["entity_ids"]` (list[int])
     The `_proposal_entity_refs_resolve` helper in `loom.py:218-242` already encodes this dispatch shape — the heartbeat code reuses the same dispatch via a sibling helper `_referenced_entity_ids(prop)`. Both `applied` and `crosspoll_emitted` heartbeat sites use it. §7 #8 / #9 gain `set_attribute`-applied and `add_relation`-applied coverage so the gap can't regress.
  2. **Race-test scope clarification.** v2.0–v2.2 §6 SIG #4 framed Test #5 as "proves the SQLite-level race window with `time.sleep` injection between find and insert." That's wrong: the sleep lands INSIDE `write_transaction()`, after the Python `threading.Lock` is already held, so the test exercises lock-held serialization, not SQLite-level contention. The Python lock IS the actual safety mechanism (intentional design choice — write_transaction's lock serializes all writers within one process), so the test's value is in pinning the contract "two threads racing for the same unknown actor produce one entity row" regardless of mechanism. The test docstring is updated to say so. We do not attempt a true SQLite-level race test (would require multi-process or moving the sleep outside `write_transaction()`, neither of which models real prod call sites).

- **2026-05-09 v2.2** — implementation-time clarifications surfaced during Phase 4 of the PR plan:
  1. **FK-defensive heartbeat skip.** §4.2 now documents that both `_replace_events_with_source` and `bump_last_seen_ts` silently skip ids that don't resolve to a real `entities` row. `events.entity_ids` is a JSON blob with no FK, but `attributes.entity_id` does have an FK; the existing `test_entity_ids_truncation_logs_when_capped` compaction test passes synthetic ids. Defensive `SELECT 1 FROM entities WHERE id = ?` guard added to both heartbeat paths. Production effect: zero (digest `union_ids` and proposal `entity_ids` always reference real rows). Test effect: `test_entity_ids_truncation_logs_when_capped` stays green without coupling to FK enforcement details.
  2. **§7 Aging Test #7 timing math.** v2.1's snippet (`digest_ts=1000.0`, `now=digest_ts + 30 * SECONDS_PER_DAY`, 14-day cutoff) would have retired all 40 entities (survivors got bumped to 1000.0, then `now ≈ 2.6M` aged them past cutoff). Implementation uses `digest_ts = 30 * SECONDS_PER_DAY`, `now = digest_ts + 5 * SECONDS_PER_DAY` so survivors are 5 days old (kept) and truncated-out NPCs are 35 days old (retired). Test still pins the design's intent: in-digest bumps protect, truncated-out ones age.

- **2026-05-09 v1** — initial draft after Codex review of the inline sketch in chat.
  Codex flagged one fatal (schema CHECK constraint on `events.source`) and seven significant issues (race on find-then-add, lookup precedence not implemented, avatar-first risk, `min_keep_references` × compaction interaction, loom doesn't bump `last_seen_ts`, `find_entity_by_name` doesn't filter retired, dangling refs after digest truncation). All addressed below; specifics in §6.
- **2026-05-09 v2.1** — line-citation correctness pass while drafting the PR plan. Two factual errors in v2 that don't affect architecture but would mislead the implementer:
  1. **Bogus precedent.** v2 cited `_apply_op_inline` in `verse/loom.py` as the working precedent for the inline-helper extraction (§3 FATAL fix; §11 Step 0b). That symbol does not exist — `grep -n '_inline' plugins/llm/src/llm/verse/loom.py` returns nothing. Replaced with the actual inline-pattern reference (`opt_in_avatar` in `store.py:465-560`, which already does multi-step work directly on a single `write_transaction() as conn` block); no change to the extraction strategy itself.
  2. **Test-migration count was 4, actual is 8.** v2 said Step 5b migrates "the four assertion-on-string tests in `tests/verse/test_compaction.py:47, 66, 92, 192`". `grep -n 'assert out' plugins/llm/tests/verse/test_compaction.py` shows eight: 47, 66, 92, 130, 176, 192, 225, 278. v2 caught every `skipped_*` site but missed every `"compacted"` site — those still equality-check the string and will break under the NamedTuple migration. Updated §4.3, §7 #15, §10 to enumerate all eight. The +20 line estimate in §10 is doubled to +40.

- **2026-05-09 v2** — second adversarial pass (Codex) + senior code-review pass (general-purpose). Both surfaced the same two FATAL claims-vs-code mismatches; eight more SIG gaps:
  1. **FATAL — `dispatch_verse_tool_call` returns `None`**, not a `ToolResult`. The wrapper `make_verse_extra_handlers` (`avatar.py:438-462`) always returns `{"status":"ok"}`. Validation errors and `event_id` payloads from §3 are *not observable by the model* under the current contract. Resolved by adding **Step 0a** to §11: retrofit `dispatch_verse_tool_call` to return a structured result and update `make_verse_extra_handlers` to propagate it. Touches the four existing tool branches.
  2. **FATAL — `write_transaction` is non-reentrant** (uses a `threading.Lock`, see `opt_in_avatar`'s warning at `store.py:471-475`). The §3 pseudocode that calls public `add_entity` / `set_attribute` / `add_event` from inside `record_user_event`'s `write_transaction` will *deadlock*. The `_lookup_in_tx` / `_add_entity_in_tx` / `_set_attr_in_tx` / `_add_event_in_tx` helpers the v1 doc cited *do not exist*. Resolved by adding **Step 0b** to §11: refactor existing public mutators to expose `_inline(conn, …)` helpers (modelled on `_apply_op_inline` in `loom.py`); public methods become thin wrappers. No behaviour change, all existing tests stay green; precondition for §3.
  3. **SIG — `compact_verse` returns a string**, so §8's UX text `compaction outcome for #foo: skipped (only 7 events; floor is 20); aged 2 entities (kept 5)` is unproducible without a contract change. Resolved: §4 now specifies `compact_verse` returns a `CompactionOutcome` NamedTuple `(state: str, total_events: int, kept_in_digest: int)`; the eight assertion-on-string tests in `tests/verse/test_compaction.py` (lines 47, 66, 92, 130, 176, 192, 225, 278 — every `assert out == "..."` site, both `skipped_*` and `"compacted"`) are migrated as part of Step 5b.
  4. **SIG — Test #5 (race) tests Python lock, not the actual race window.** Both threads serialise behind `_lock` before any SQLite contention. Resolved: §7 #5 now specifies the test mocks `time.sleep` between lookup and insert, modelled on `test_concurrent_opt_in_distinct_nicks_one_place` (`test_store.py:611-629`) and the crosspoll barrier pattern (`test_crosspoll_store.py:84-108`). Also reframes the concurrency *claim*: §3 now scopes safety to "one cached `VerseStore` instance per channel within one process" (the loom and `verse_record` share that cached instance via `_get_or_create_verse_store`, `plugin.py:4952-4964`).
  5. **SIG — Test #6 doesn't pin the heartbeat call site.** Resolved: §4 now specifies the bump is added inside `_replace_events_with_source` (`store.py:391-434`) — not in `compaction.py` — so it executes on the same `conn` that wrote the digest, atomically. Test #6 asserts `get_attribute(entity_id, "last_seen_ts")` equals the digest's `now()`.
  6. **SIG — Test #7 hard-codes 40 entities** while `_MAX_DIGEST_ENTITY_IDS` is a private 32. Bumping the constant silently inverts the test. Resolved: §7 #7 imports `_MAX_DIGEST_ENTITY_IDS` from `verse.compaction` and uses `_MAX_DIGEST_ENTITY_IDS + 8`.
  7. **SIG — Loom heartbeat (§4 #2) had no test.** Resolved: §7 gains tests #8 (loom `applied` proposal bumps), #9 (loom `queued`/`rejected_invalid_refs` proposals do NOT bump — keeps junk from being kept alive by low-confidence model output).
  8. **SIG — `objects` was dropped, but the flagship example "stinky dan threw a guff grenade at Andrew" implies item tracking.** Resolved: §1 now states explicitly that v1 records the grenade as *prose only* in `summary`. The model is instructed not to put items into `actors`. Documented in §1 example block + operator guide.
  9. **SIG — Avatar opt-out → auto-NPC → opt-in produces three same-name rows.** The orphan NPC outlives its usefulness if it ever heartbeated. Resolved: §2 documents the explicit state machine and §7 #10 covers it. Aging will retire the orphan after `verseAutoEntityRetireDays`. Acceptable.
  10. **SIG — Heartbeat-on-touch claim was overbroad.** `verse_act`, `verse_move`, `verse_look`, `verse_recall`, `add_relation` don't bump. Resolved: §4 narrows scope to *exactly* `record_user_event`, loom `apply_or_queue` (applied/crosspoll-emitted only), and `_replace_events_with_source`. Other read/write paths stay heartbeat-free; auto-NPCs are *only* kept alive by being re-mentioned via `verse_record` or surviving compaction's digest. Documented in §4.
  11. **SIG — `AgingOutcome(scanned, retired, kept)` is invariant-violating.** `scanned == retired + kept`. Resolved: dropped `kept`; now `(scanned: int, retired: int)`. Operator string derives `kept = scanned - retired` if needed.
  12. **SIG — `maxItems: 8` in spec is static while registry is dynamic.** The registry is a *floor*, not a ceiling. Resolved: §1 now generates the tool spec dynamically per channel — `make_verse_tool_specs(*, max_actors)` takes the per-channel registry value at call time. `make_verse_tool_specs()` is already called per assistant request, so the per-channel reach exists.
  13. **MINOR cleanups** integrated inline:
      - Renames: `actor_avatar_id` → `actor_id`, `subjects` → `actors`, `find_entity_by_name_active_first` → `find_active_entity_by_name`.
      - Summary too long: returns `ToolResult(error=...)` rather than silent truncation.
      - Subjects with empty/whitespace-only or non-string entries: filter before slicing (filter-then-slice fixes the integer-eats-a-slot footgun).
      - Wrong test name in v1 §7 (`test_dispatch_verse_tool_call_unknown_op`) corrected to actual impacted assertions: `tests/verse/test_avatar.py:617` (the literal 4-set `assert set(handlers.keys()) == {…}`) and the `_verse_names` literal at `avatar.py:452` (must grow to 5-set).
      - `_register_verse_tools` → real call site is `plugin.py:3281` `make_verse_extra_handlers(verse_route.store, verse_route.avatar_id)`.
      - Wiring test (§7 #11) verifies per-channel registry scope of `verseAutoEntityRetireDays`.
      - `verseRecordAvatarPrecedence` reference removed from §2 (out of scope = nowhere mentioned).
      - Operator guide §8 lists the three new H2 anchors so a reviewer can spot collisions.
      - `last_seen_ts` as attribute row is fine for v1; documented as a known scale concern in §9 ("revisit if aging-pass latency exceeds ~50ms").
      - Realistic line estimate: ~900, not 770. §10 updated.

---

## 1. Tool surface

`make_verse_tool_specs(*, max_actors: int = 8)` (`verse/avatar.py:16`) gains a fifth entry. The function signature gains a keyword arg so the per-channel registry value `verseAutoEntityMaxNamesPerCall` flows into the advertised `maxItems` — without this, the registry would be a floor (the model refuses to send more than the static cap) instead of a ceiling. `make_verse_tool_specs` is already called per assistant request, so the per-channel reach is free.

```jsonc
{
  "type": "function",
  "function": {
    "name": "verse_record",
    "description":
      "Record an in-world event involving one or more named actors. "
      "Use whenever a member narrates events that aren't strictly about "
      "their own avatar (e.g. \"stinky dan threw a guff grenade at "
      "Andrew\" — record actors=[\"stinky dan\",\"Andrew\"], the grenade "
      "stays in the summary as prose). Names that don't match an existing "
      "entity are auto-created as kind=npc. Items, places, and weapons "
      "are NOT actors — only put characters/people in the actors list.",
    "parameters": {
      "type": "object",
      "properties": {
        "summary": {
          "type": "string",
          "description":
            "What happened, in past tense, ≤200 chars. The full prose "
            "narration including any items, places, or weapons mentioned. "
            "e.g. 'stinky dan threw a guff grenade at Andrew'."
        },
        "actors": {
          "type": "array",
          "items": { "type": "string" },
          "maxItems": "<max_actors>",   // ← injected from registry per channel
          "description":
            "Names of CHARACTERS (people/npcs) central to the event. "
            "Do NOT include items, weapons, places, or abstractions."
        }
      },
      "required": ["summary"]
    }
  }
}
```

**Decisions vs v0 sketch:**

- `subjects` → `actors` — the v0 name implied grammatical subjects (actors *doing*), but the description includes patients ("threw at Andrew"). `actors` is honest and cuts model hesitation.
- `objects: string[]` dropped from v1. Items and places have downstream behavioural differences (movement, scene snapshots, item-take/drop verbs in `verse_act`). The flagship example records *Dan* and *Andrew* as entity links; the *grenade* is prose only. Documented in the tool description and operator guide so the model behaves predictably. Item/place auto-create is a follow-up if real verses surface the need.
- Auto-create kind is fixed to `npc`. An existing entity of any kind matches first via the lookup policy below; only when nothing matches do we mint an npc.
- `maxItems` is dynamic, defaulting to 8 (matches `verseAutoEntityMaxNamesPerCall` default). The user signalled "lots of entities" — 8 is generous for a single sentence and bounds high-cardinality flooding. Operators can raise/lower per channel; raising past 16 should be discouraged in the doc.

## 2. Lookup policy & precedence

Codex's most important call-out: today's `find_entity_by_name(name)` returns the first id ASC across **all kinds and statuses**, contradicting the spec'd `avatars > npcs > items > places` precedence. Either the spec is wrong or the impl is. We're doing both: a new helper, and an explicit decision on precedence.

**New method:** `VerseStore.find_active_entity_by_name(name) -> Entity | None`

```python
def find_active_entity_by_name(self, name: str) -> Entity | None:
    """Resolve a name with precedence avatar > npc > item > place,
    case-insensitive, restricted to status='active'. Single SQL via
    CASE on kind. Used by verse_record to bind narration to a real
    avatar when the name matches one, and to reuse active npcs the
    verse has already met. Retired entities are never returned (so
    new mentions create a fresh row instead of silently rehydrating)."""
```

The legacy `find_entity_by_name(name, kind=...)` stays as-is for `verse_act`'s movement/item lookups (those genuinely want a kind filter and don't care about precedence).

**Precedence decision: avatar-first stays.** Codex's senior-review pass flagged the risk that "Andrew did X" silently links a real player's avatar to junk. Counterweight:
- Members deliberately opt in via `@verseopt in`; their avatar identity is durable via `avatar_link`. Ambient narration referencing them by nick is the *desired* behaviour ("Andrew laughed" should attach to Andrew's avatar so `verse_recall` and `@verseproposals` reflect what the verse says about Andrew).
- Avatar-first is what makes `verse_record` work for the Forest mode case where multiple humans co-narrate.
- Mitigation: when a name resolves to an avatar, **don't bump `last_seen_ts`** and **don't tag `auto_created`** (the avatar wasn't created by this call and isn't aging out). The avatar's identity is unaffected; only the event row links.

**Avatar opt-out / re-opt-in edge case (Codex v2 SIG #6).** The unlink path deletes `avatar_link` and retires the entity (`store.py:345-356`). State machine after a member opts out, gets mentioned (auto-NPC created), then opts back in:

| Step | entities table | avatar_link |
|---|---|---|
| Initial: opted in | `(id=1, kind=avatar, name=Andrew, status=active)` | `(entity_id=1, nick=andrew, account=…)` |
| Opt out | `(id=1, status=retired)` | _deleted_ |
| Member mentioned by verse_record | `(id=1, retired)`, `(id=2, kind=npc, name=Andrew, status=active, auto_created=1)` | _none_ |
| Opt back in | `(id=1, retired)`, `(id=2, npc active)`, `(id=3, kind=avatar, name=Andrew, status=active)` | `(entity_id=3, …)` |

After re-opt-in, `find_active_entity_by_name("Andrew")` returns `id=3` (avatar wins by precedence). `id=2` is an orphan NPC that aging will retire after `verseAutoEntityRetireDays` of no further mentions — provided no future `verse_record` mentions "Andrew" before then (each mention now resolves to id=3 and bumps id=3's nothing — avatars don't bump). Worst case the orphan lingers `verseAutoEntityRetireDays`. Documented behaviour; not blocking.

## 3. Dispatch

**Pre-requisite: dispatch contract retrofit (Step 0a).** Today `dispatch_verse_tool_call` returns `None` and `make_verse_extra_handlers` always returns `{"status":"ok","tool":name}` — so any error or success payload `verse_record` tries to surface is invisible to the model. v2 retrofits both:

```python
@dataclass
class VerseDispatchResult:
    ok: bool
    payload: dict[str, Any] | None = None   # serialised to tool result on success
    error: str | None = None                # surfaced as tool error on failure

def dispatch_verse_tool_call(store, avatar_id, name, args, *, logger, now=time.time):
    # returns VerseDispatchResult; existing four branches return ok=True with
    # payload={'status': 'ok'} so the JSON the model sees is unchanged.
```

The four existing branches change from side-effect-only to returning `VerseDispatchResult(ok=True, payload={"status": "ok"})`. `make_verse_extra_handlers._handler._call` consumes the result and either emits the payload as the tool content or sets a tool error. No behaviour change for existing tools; new contract for `verse_record`.

**New branch in `dispatch_verse_tool_call` (`verse/avatar.py:383`):**

```python
elif name == "verse_record":
    summary = (args.get("summary") or "").strip()
    if not summary:
        return VerseDispatchResult(ok=False, error="summary required")
    if len(summary) > 200:
        return VerseDispatchResult(
            ok=False, error=f"summary too long: {len(summary)} chars (max 200)"
        )
    raw = args.get("actors") or []
    if not isinstance(raw, list):
        return VerseDispatchResult(ok=False, error="actors must be an array")
    # Filter THEN slice — order matters. raw=["alice", 42, "bob"] with
    # max=2 must yield ["alice","bob"], not ["alice"] (the 42 ate a slot).
    max_actors = args.get("_max_actors", 8)        # closure-injected
    cleaned = [s.strip() for s in raw if isinstance(s, str) and s.strip()]
    actors = cleaned[:max_actors]
    event_id = store.record_user_event(
        actor_id=avatar_id,
        summary=summary,
        actor_names=actors,
        now=now,
    )
    return VerseDispatchResult(
        ok=True, payload={"status": "ok", "event_id": event_id}
    )
```

**The DB work is pushed into a new store method `record_user_event`** so find-or-create-then-link is atomic in one `write_transaction`:

```python
def record_user_event(self, *, actor_id, summary, actor_names, now):
    """Resolve actor_names to entity ids (auto-create as npc if unknown),
    bump last_seen_ts on each non-avatar, and write the event row — all
    in one write_transaction. Returns the new event id.

    Concurrency scope: safe across callers sharing one cached VerseStore
    instance per channel within one process (the loom and verse_record
    both go through _get_or_create_verse_store in plugin.py). Multiple
    processes touching the same DB or multiple VerseStore instances for
    the same channel are NOT defended against; out of scope for v1."""
    with self.write_transaction() as conn:
        ids = [actor_id]
        for name in actor_names:
            entity = self._find_active_entity_by_name_inline(conn, name)
            if entity is None:
                eid = self._add_entity_inline(conn, "npc", name, "")
                self._set_attribute_inline(conn, eid, "auto_created", "1")
                self._set_attribute_inline(conn, eid, "last_seen_ts", str(now()))
            else:
                eid = entity.id
                if entity.kind != "avatar":
                    self._set_attribute_inline(conn, eid, "last_seen_ts", str(now()))
            ids.append(eid)
        return self._add_event_inline(
            conn, summary=summary, entity_ids=ids, source="avatar"
        )
```

**FATAL fix: `_*_inline(conn, …)` helpers are NEW.** Codex v2 caught that `write_transaction` is non-reentrant (uses `threading.Lock`, see `opt_in_avatar`'s warning at `store.py:471-475`). Calling public `add_entity` / `set_attribute` / `add_event` from inside `record_user_event`'s transaction would *deadlock*. **Step 0b** in §11 refactors those public mutators to expose `_inline` private helpers that take an open `conn` and skip the lock. The public methods become thin wrappers that open their own transaction and delegate. The closest existing precedent for "do many things on a single open `conn`" is `opt_in_avatar` (`store.py:465-560`), which inlines all of its DB work directly inside one `write_transaction() as conn` block. We're generalising that to be reusable across mutators rather than duplicated per call site. No behaviour change to public API; all existing tests stay green.

**`source='avatar'` (not `'user_record'`).** Codex v1 FATAL: `events.source` has a CHECK constraint to `('avatar','loom','crosspoll')` (`verse/schema.sql:43`). Adding a fourth value requires a table-rebuild migration. **v1 reuses `'avatar'`** since the actor *is* the caller's avatar acting on the world. The provenance loss (can't audit user-record vs verse_act, loom can't tell the difference in transcript ingest) is documented as a known v1 limitation; a follow-up PR can extend the CHECK with a one-shot rebuild migration if the loom turns out to riff badly on user-recorded events.

## 4. Aging & compaction outcome

### 4.1 Aging helper

New module `verse/aging.py`:

```python
class AgingOutcome(NamedTuple):
    scanned: int
    retired: int
    # `kept` derived as `scanned - retired` if needed; storing it
    # would invite invariant-violation bugs (Codex v2 SIG #11).

def age_auto_created_entities(
    store, *, retire_after_days, now
) -> AgingOutcome:
    """Soft-retire auto_created='1' entities whose last_seen_ts is
    older than now - retire_after_days*86400. Skips kind='avatar'
    defensively. retire_after_days<=0 disables (returns (0,0))."""
```

Single SQL: `SELECT entities.id, attributes.value AS last_seen FROM entities JOIN attributes ON … WHERE attributes.key='auto_created' AND attributes.value='1' AND entities.status='active' AND entities.kind != 'avatar'`. Iterate in Python; flip status with `set_status` (existing).

**`min_keep_references` is dropped.** Codex v1 SIG #2.1: lifetime event-counts are unreliable post-compaction (compaction deletes raw events). `last_seen_ts` is sufficient — entities re-mentioned stay alive; quiet entities age out. Simpler, no compaction-interaction bugs.

### 4.2 Heartbeat scope (narrowed per Codex v2 SIG #10)

`last_seen_ts` is bumped at exactly three sites — *not* "every code path that touches an entity" (the v1 doc overclaimed):

1. **`record_user_event`** — every non-avatar entity referenced by an `actors` list, on every call.
2. **`apply_or_queue` in `verse/loom.py`** — only when the proposal lands as `applied` or `crosspoll_emitted`. Codex v2 SIG: `queued`/`rejected_invalid_refs` proposals must NOT bump (low-confidence model output shouldn't keep junk alive). Tested as a negative case in §7 #9. The set of bumped ids is op-dispatched (v2.3): `add_event`/`crosspoll_seed` → `entity_ids`; `set_attribute` → `[entity_id]`; `add_relation` → `[from_id, to_id]`; `add_entity` → `[]`. Encoded in a `_referenced_entity_ids(prop)` helper that mirrors `_proposal_entity_refs_resolve`.
3. **`_replace_events_with_source` in `verse/store.py`** — *not* in `compaction.py`. The bump runs on the same `conn` that wrote the digest, atomically. Each entity in the digest's truncated `union_ids[:32]` list gets its `last_seen_ts` set to the digest's `now()`. Codex v2 SIG #5: this is the heartbeat call site, not `compaction.py`.

Other paths that touch entities — `verse_act`, `verse_move`, `verse_look`, `verse_recall`, `add_relation`, `opt_in_avatar` — do **not** bump. Auto-NPCs are kept alive *only* by re-mention via `verse_record` or by surviving compaction's digest. Documented in the operator guide.

**Best-effort heartbeat semantics.** `events.entity_ids` is a JSON blob with no FK to `entities`, but `attributes.entity_id` does have an FK. Both heartbeat paths (`_replace_events_with_source` inline bump and the public `bump_last_seen_ts`) silently skip ids that don't resolve to a real `entities` row, via a `SELECT 1 FROM entities WHERE id = ?` guard. In production this is a no-op (digest `union_ids` and proposal `entity_ids` always reference real rows); the guard exists to keep the existing `test_entity_ids_truncation_logs_when_capped` compaction test green without coupling tests to FK enforcement details.

### 4.3 `compact_verse` return shape change

§8's outcome string requires counts that the current `compact_verse` doesn't surface. v2 changes the return type:

```python
class CompactionOutcome(NamedTuple):
    state: str           # 'compacted' | 'skipped_disabled' | 'skipped_below_floor' | 'skipped_no_events'
    total_events: int    # COUNT(*) FROM events at pass entry
    kept_in_digest: int  # len(union_ids[:32]) when state=='compacted', else 0
```

This breaks the eight assertion-on-string tests in `tests/verse/test_compaction.py` — every `assert out == "..."` site at lines 47, 66, 92, 130, 176, 192, 225, 278 (both the `skipped_*` and `"compacted"` cases) — migrated as part of Step 5b (split). The plugin-side caller already has `min_keep_events` (it's an input), so the friendlier outcome string can render without a second query.

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

### `tests/verse/test_verse_record.py` (new)

1. `verse_record` with all-new names creates `kind='npc'` entities, links event with caller's avatar id first in `entity_ids`.
2. Existing avatar with the same name as an `actor` is linked (avatar-first precedence); the avatar gets neither `auto_created='1'` nor a `last_seen_ts` bump (verify via `get_attribute` returning None).
3. Existing npc with the same name is reused (single entity row, not a duplicate); `last_seen_ts` is bumped to `now()`.
4. Repeated `verse_record` for the same npc keeps one entity row; `last_seen_ts` is the most recent timestamp.
5. **Race (real)** — pattern modelled on `test_concurrent_opt_in_distinct_nicks_one_place` (`test_store.py:611-629`) and `test_crosspoll_store.py:84-108`. Mock `time.sleep` between `_find_active_entity_by_name_inline` and `_add_entity_inline` so the contention window is real; two threads race the find-or-create on the same `VerseStore` instance and only one entity row results. Without the sleep injection the test passes trivially (Python lock serialises) — **the sleep IS the test**.
6. Retired entity with the same name as an actor is **not** rehydrated — a new active npc is created instead. Verifies the active-filter in `find_active_entity_by_name`.
7. `actors` longer than `verseAutoEntityMaxNamesPerCall` is truncated to the first N after non-string filtering; mixed-type input `["alice", 42, "bob"]` with `max_actors=2` yields actors `["alice","bob"]` (filter-then-slice).
8. Empty `summary` returns `VerseDispatchResult(ok=False, error="summary required")`; no DB writes.
9. `summary` longer than 200 chars returns `VerseDispatchResult(ok=False, error="summary too long: …")`; no truncation, no DB writes (model is expected to retry).
10. **Avatar opt-out → re-opt-in three-row state** (Codex v2 SIG #6): opt in alice, opt out, `verse_record actors=[alice]` creates auto-NPC, opt back in. Subsequent `verse_record actors=[alice]` resolves to the NEW avatar id (not the orphan NPC). The orphan NPC ages out after `retire_after_days`.
11. Subject names case-collide with avatars — `actors=["ANDREW"]` resolves to avatar "andrew" (case-insensitive `LOWER(name) = LOWER(?)` already in `find_entity_by_name`).
12. Empty/whitespace-only actor strings (`actors=["", "  ", "alice"]`) are filtered before slicing — only "alice" is processed; no empty-name entities created.
13. `actor_id` pointing at a *retired* avatar raises (matching `verse_act`'s "avatar retired" guard).

### `tests/verse/test_verse_aging.py` (new)

1. Auto-created npc with `last_seen_ts` past cutoff is retired.
2. Auto-created npc with `last_seen_ts` recent is kept.
3. Manually-created entity (no `auto_created='1'` attribute) is never touched, even past cutoff.
4. Avatar with `auto_created='1'` (defensive: shouldn't happen) is never retired (the `kind != 'avatar'` guard).
5. `retire_after_days=0` makes the helper a no-op (returns `AgingOutcome(0,0)`).
6. **Compaction interaction (heartbeat fires)**: auto-created npc with 3 raw event references; run `replace_events_with_lore_digest`; aging runs immediately after. Assert `get_attribute(entity_id, "last_seen_ts")` equals the digest's `now()` *and* entity status remains `active`. Heartbeat call site is `_replace_events_with_source` (verified by reading the test target).
7. **Compaction-truncation interaction (intentional retire)**: same setup as #6 but with `_MAX_DIGEST_ENTITY_IDS + 8` entities (imported from `verse.compaction`, not hardcoded — bumping the constant must not silently invert this test). The truncation drops our npc from the digest's `union_ids`; aging then correctly retires it.
8. **Loom heartbeat (positive)** — Codex v2 SIG #7: an `apply_or_queue` call landing as `applied` or `crosspoll_emitted` bumps `last_seen_ts` on every referenced entity.
9. **Loom heartbeat (negative)** — Codex v2 SIG #7: an `apply_or_queue` call landing as `queued` (low confidence) or `rejected_invalid_refs` does NOT bump. Critical: low-confidence model output must not keep aging junk alive.

### `tests/test_plugin.py` additions (plugin-level wiring)

10. `_run_compaction_pass` calls `age_auto_created_entities` once per channel returned by `_verse_enabled_channels()`.
11. The aging call reads `verseAutoEntityRetireDays` at the *channel* scope, not the global scope (verify via `registryValue` mock with `channel=` kwarg assertion).
12. An aging exception in one channel doesn't abort the pass for others (mirror existing `_run_compaction_pass` failure-isolation test).
13. `make_verse_tool_specs(max_actors=N)` with N from `verseAutoEntityMaxNamesPerCall` flows into the dispatch closure built at `plugin.py:3281`.

### `tests/verse/test_avatar.py` updates

14. `test_unknown_tool_name_logged_and_skipped` (`tests/verse/test_avatar.py:596`) is unaffected; the actually-impacted assertions are:
    - The 4-set assertion at `tests/verse/test_avatar.py:617` (`assert set(handlers.keys()) == {…}`) becomes a 5-set.
    - `_verse_names` literal at `avatar.py:452` grows from a 4-set to a 5-set.
    - Any test asserting `len(make_verse_tool_specs()) == 4` becomes `== 5`.

### `tests/verse/test_compaction.py` updates (Step 5b)

15. Migrate every `assert out == "..."` (and `assert out1 ==`, `assert out2 ==`) site to NamedTuple `.state` lookups: `out.state == "compacted"`, `out.state == "skipped_below_floor"`, etc. Eight sites total: lines 47, 66, 92, 130, 176, 192, 225, 278. v2 listed only the four `skipped_*` sites; the four `"compacted"` sites would equally fail under the NamedTuple migration. Also assert the new NamedTuple fields (`out.total_events`, `out.kept_in_digest`) carry sensible values in the `"compacted"` cases — exact numbers depend on the test's seed pattern.

### `tests/verse/test_avatar.py` dispatch-contract updates (Step 0a)

16. Existing tests verifying `dispatch_verse_tool_call` returns `None` are updated to assert `VerseDispatchResult(ok=True, payload={"status":"ok"})` for the four existing tools (no behaviour change observable via the wrapper's JSON output).

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
| `plugins/llm/src/llm/verse/avatar.py` (tool spec + dispatch branch + `VerseDispatchResult` + 4-branch return migration) | +60 |
| `plugins/llm/src/llm/verse/store.py` (Step 0b inline-helper extraction + record_user_event + lookup + attr query + heartbeat in `_replace_events_with_source`) | +120 |
| `plugins/llm/src/llm/verse/aging.py` (new) | +60 |
| `plugins/llm/src/llm/verse/loom.py` (heartbeat in apply_or_queue, applied/crosspoll-emitted only) | +10 |
| `plugins/llm/src/llm/verse/compaction.py` (`CompactionOutcome` NamedTuple + return-shape migration) | +20 |
| `plugins/llm/src/llm/plugin.py` (compaction-pass hook + new outcome string + max_actors plumbing at `:3281`) | +35 |
| `plugins/llm/src/llm/config.py` (two registry keys) | +20 |
| `plugins/llm/tests/verse/test_verse_record.py` (new, tests #1-13) | +280 |
| `plugins/llm/tests/verse/test_verse_aging.py` (new, tests #1-9) | +220 |
| `plugins/llm/tests/test_plugin.py` additions (tests #10-13) | +120 |
| `plugins/llm/tests/verse/test_compaction.py` migration (Test #15, eight sites) | +40 |
| `plugins/llm/tests/verse/test_avatar.py` 5-set + dispatch-contract updates (Tests #14, #16) | +30 |
| `docs/guide/operator/forest-verse.md` (3 new H2 sections) | +80 |
| `CHANGELOG.md` | +15 |

Total **~1110 lines**. One PR. No schema migration. v1's 770-line estimate undercounted Step 0a (dispatch retrofit), Step 0b (inline-helper extraction across 4+ public methods), `CompactionOutcome` migration, and the loom heartbeat tests. v2.1 corrected Step 5b's test-migration count from 4 sites to 8 (+20 lines).

## 11. Implementation order

For the follow-up PR plan doc (`2026-05-09-verse-record-pr1.md`). Each step is independently red-green-commit-able; integration gates are noted.

**Step 0a — Dispatch contract retrofit.** Introduce `VerseDispatchResult`. Change `dispatch_verse_tool_call` to return it; update `make_verse_extra_handlers._handler._call` to consume it. The four existing branches return `VerseDispatchResult(ok=True, payload={"status":"ok"})`; observable JSON to the model is unchanged. Migrate the dispatch-call tests (Test #16 above). **Gate**: `tests/verse/test_avatar.py` green; no other tests should change.

**Step 0b — Store mutator inline-helper extraction.** Refactor `add_entity`, `set_attribute`, `add_event`, `set_status` to delegate to private `_*_inline(conn, …)` helpers. Public methods become thin wrappers that open `write_transaction` and call the inline helper. The closest existing precedent for "many DB ops on one open `conn`" is `opt_in_avatar` (`store.py:465-560`); we generalise that pattern across the four mutators. **No behaviour change**, all existing store tests stay green. **Gate**: full `pytest plugins/llm/tests/verse/` green.

**Step 1 — New store queries.** TDD red-green for `find_active_entity_by_name(name)`, `list_entities_with_attribute(key, value, *, status)`. Inline variants (`_find_active_entity_by_name_inline`, `_set_attribute_inline`) come from Step 0b.

**Step 2 — `record_user_event`.** TDD red-green using the inline helpers from Steps 0b + 1. Race test (#5) with sleep injection. **Gate**: tests #1-13 from §7 green.

**Step 3 — Aging helper + tests #1-5.** New `verse/aging.py`; pure helper. **Gate**: tests #1-5 from §7 green.

**Step 4 — Heartbeat wiring.** Three sites:
- `record_user_event` already does it (Step 2).
- `_replace_events_with_source` in `verse/store.py` — bump `last_seen_ts` for every entity in `union_ids[:32]` after the digest insert, on the same `conn`. Tests #6, #7.
- `apply_or_queue` in `verse/loom.py` — bump only when the result is `applied` or `crosspoll_emitted`. Tests #8, #9.

**Gate**: tests #6-9 green.

**Step 5a — Wire aging into the compaction pass.** Plugin's `_run_compaction_pass` calls `age_auto_created_entities` per channel after `compact_verse`. **Old enum-string outcome unchanged at this step.** Tests #10-12.

**Step 5b — `compact_verse` returns `CompactionOutcome` NamedTuple + new outcome string.** Migrate **all eight** assertion-on-string sites in `tests/verse/test_compaction.py` (lines 47, 66, 92, 130, 176, 192, 225, 278 — every `assert out == ...` site, both `skipped_*` and `"compacted"`) to NamedTuple `.state` lookups. Test #15. Plugin renders the friendlier outcome including aging counts. **Gate**: full `pytest` green.

**Step 6 — Tool spec + dispatch branch.** Add `verse_record` to `make_verse_tool_specs(max_actors=…)`; add the dispatch branch from §3 using `VerseDispatchResult`. Plumb `verseAutoEntityMaxNamesPerCall` from registry through the call site at `plugin.py:3281` (`make_verse_extra_handlers`). Test #13.

**Step 7 — Operator guide + CHANGELOG.**
- New H2 anchors in `docs/guide/operator/forest-verse.md`:
  - `## Member-driven worldbuilding (verse_record)` — example `vibebot, stinky dan threw a guff grenade at Andrew` → resulting event row + linked entities.
  - `## Auto-created NPCs and aging` — `verseAutoEntityRetireDays`, heartbeat semantics, "why do entities disappear" troubleshooting.
  - `## Compaction outcome reference` — what each `CompactionOutcome.state` means in the friendlier string.
- CHANGELOG entry under "Unreleased".

**Step 8 — Re-review.** Before merging, dispatch:
- `superpowers:requesting-code-review` (or `general-purpose` agent in code-review mode) over the diff.
- `codex:codex-rescue` adversarial pass over the diff.
Integrate findings inline. **Gate**: both reviewers say "ready to merge."

**Step 9 — Wait for CI + Docker, restart prod, validate** per the standard protocol (CI green → wait for `Build and Push Docker Image` → `systemctl --user restart vibebot` → check logs for clean startup, no AssertionErrors, no 401s).

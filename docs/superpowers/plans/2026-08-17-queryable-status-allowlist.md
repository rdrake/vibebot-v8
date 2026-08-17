# Queryable status allowlist implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the bot answer about status pages it does not poll, selected by an enum-constrained `service` argument whose vocabulary comes only from operator config.

**Architecture:** Both config keys gain a shared `Name=url` grammar parsed in one place. A new `_status_named_pages()` is the single source of truth for the enum, for `service` resolution, and for pruning. Queryable pages get their own lazily-filled cache and never acquire lifecycle state. The name→source mapping is resolved once per completion and bound to the tool callback, so the schema and the dispatcher are one snapshot.

**Tech Stack:** Python 3.14, Limnoria plugin framework, pytest + pytest-mock, `uv`, ruff + ty via pre-commit.

**Spec:** `docs/superpowers/specs/2026-08-17-queryable-status-allowlist-design.md` — read its "What v1 got wrong" section before starting; it explains why the obvious design is unsafe.

## Global constraints

- **Nothing third-party may reach the enum.** Selector names come from `bot.conf` only. `page_name` keeps its existing job as the payload's `service` display field. This is not a preference: v1 was rejected because a compromised page returning `page.name = "Cloudflare"` would capture the operator's name for the real Cloudflare.
- **Tool schemas are part of the paid prompt-cache prefix** (`service.py:2798`). The enum must be a pure function of config so its bytes are stable across restarts and polls.
- **`_status_state`, `_status_read_cache` and `_status_last_fetch` belong to the poller** and are pruned against the **polled** set only. A queryable page must never acquire lifecycle state.
- **The history and query dicts are pruned against polled ∪ queryable.** Pruning history against the polled set alone deletes an allowlisted page's 4 MB history cache 120 seconds after it is fetched.
- **Schema injection copies four dict levels** — `tool`, `function`, `parameters`, `properties`. Two is what ships today and is sufficient only while just `description` changes.
- **Logging uses `%i`, never `%d`.** Config diagnostics fire on the poll path only (`warn=False` on request paths), or one typo logs once per chat message.
- **Do not push.** Commit locally; the orchestrator handles pushes.
- Suite is 3153 repo-wide (3130 in `plugins/llm/tests/`). Report exact counts.

## File structure

| File | Responsibility | Change |
|---|---|---|
| `plugins/llm/src/llm/config.py` | Registry keys | `statusPageUrls` default gains names; add `statusQueryablePages` |
| `plugins/llm/src/llm/plugin.py` | Parsing, caches, payload | Bulk of the work |
| `plugins/llm/src/llm/service.py` | Schema injection, wiring | `_with_status_context`, gate, frozen mapping |
| `plugins/llm/src/llm/assistant.py` | Tool schema + handler | `service` property; handler forwards it |
| `plugins/llm/tests/conftest.py` | Fixtures | New config, caches, constants, bindings |
| `docs/guide/operator/configuration.md`, `docs/guide/user/service-status.md` | Docs | New key, the ask-vs-announce split |

---

### Task 1: The shared grammar and name resolution

**Files:**
- Modify: `plugins/llm/src/llm/config.py:996-1020`
- Modify: `plugins/llm/src/llm/plugin.py:1077-1135` (`_status_sources`, `_status_prune_sources`)
- Test: `plugins/llm/tests/test_status_poller.py`, `plugins/llm/tests/test_config.py`

**Interfaces:**
- Produces: `LLM._status_parse_pages(raw_entries, *, key, cap, warn) -> dict[str, str]` — ordered name → canonical source, one parser for both keys.
- Produces: `LLM._status_named_pages(*, warn=True) -> dict[str, str]` — polled entries then queryable, deduped by canonical source.
- Produces: `LLM._status_sources(*, warn=True) -> list[str]` keeps its signature and return type; it now takes the values of the polled mapping.
- Produces: `LLM._STATUS_MAX_QUERYABLE = 20`.

- [ ] **Step 1: Write the failing tests**

Add to `plugins/llm/tests/test_status_poller.py`:

```python
class TestPageGrammar:
    """One grammar for both keys. A bare URL stays valid and takes its host as
    its name, which is what statusPageUrls entries did before names existed."""

    def test_named_and_bare_entries_both_parse(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [
            "Claude=https://status.claude.com",
            "https://www.githubstatus.com",
        ]
        plugin._registry["statusQueryablePages"] = []
        assert plugin._status_named_pages() == {
            "Claude": "https://status.claude.com",
            "www.githubstatus.com": "https://www.githubstatus.com",
        }

    @pytest.mark.parametrize(
        "entry",
        [
            "=https://status.claude.com",          # empty name
            "has space=https://x.example",         # impossible in a space list, but explicit
            "toolongname" * 5 + "=https://x.example",
            "bad!name=https://x.example",
            "Name=not a url",
            "Name=ftp://x.example",
            "Name=https://x.example/path",
        ],
    )
    def test_unusable_entries_are_dropped_not_fatal(self, status_plugin, entry):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [entry, "Good=https://status.claude.com"]
        plugin._registry["statusQueryablePages"] = []
        assert plugin._status_named_pages() == {"Good": "https://status.claude.com"}

    def test_duplicate_name_keeps_the_first(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [
            "Dup=https://status.claude.com",
            "dup=https://www.githubstatus.com",
        ]
        plugin._registry["statusQueryablePages"] = []
        assert plugin._status_named_pages() == {"Dup": "https://status.claude.com"}

    def test_two_names_one_canonical_source_drops_the_later(self, status_plugin):
        """A silent skip would show the operator a valid-looking entry in
        @config that never appears in the enum."""
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = ["Foo=https://status.claude.com"]
        plugin._registry["statusQueryablePages"] = ["Bar=https://status.claude.com/"]
        assert plugin._status_named_pages() == {"Foo": "https://status.claude.com"}

    def test_polled_entries_come_first_and_win_a_collision(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = ["X=https://status.claude.com"]
        plugin._registry["statusQueryablePages"] = ["Y=https://www.cloudflarestatus.com"]
        assert list(plugin._status_named_pages()) == ["X", "Y"]

    def test_queryable_is_capped(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = []
        plugin._registry["statusQueryablePages"] = [
            f"N{i}=https://status{i}.example.com" for i in range(25)
        ]
        assert len(plugin._status_named_pages()) == plugin._STATUS_MAX_QUERYABLE

    def test_sources_returns_polled_urls_only(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = ["Claude=https://status.claude.com"]
        plugin._registry["statusQueryablePages"] = ["CF=https://www.cloudflarestatus.com"]
        assert plugin._status_sources() == ["https://status.claude.com"]


class TestPruneKeepsQueryableHistory:
    """The subtlest interaction in this feature. Pruning history against the
    polled set alone deletes an allowlisted page's history — up to 4 MB, cached
    for an hour — on the very next poll, 120 seconds after it was fetched,
    along with its failure backoff. Every history question would refetch 4 MB."""

    def test_queryable_history_survives_a_poll(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = ["Claude=https://status.claude.com"]
        plugin._registry["statusQueryablePages"] = ["CF=https://www.cloudflarestatus.com"]
        plugin._status_history_cache = {CF: ()}
        plugin._status_history_at = {CF: plugin._now}
        plugin._status_history_failed_at = {CF: plugin._now}

        plugin._run_status_poll()

        assert CF in plugin._status_history_cache, "queryable history was pruned by the poll"
        assert CF in plugin._status_history_at
        assert CF in plugin._status_history_failed_at

    def test_lifecycle_state_is_still_pruned_against_polled_only(self, status_plugin):
        """The other half: a queryable page must never keep lifecycle state,
        or a question could consume an announcement it has no right to."""
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = ["Claude=https://status.claude.com"]
        plugin._registry["statusQueryablePages"] = ["CF=https://www.cloudflarestatus.com"]
        plugin._status_state = {CF: statuspage.StatusState(seeded=True)}
        plugin._status_read_cache = {CF: green_snapshot(plugin._now)}

        plugin._run_status_poll()

        assert CF not in plugin._status_state
        assert CF not in plugin._status_read_cache
```

`CF = "https://www.cloudflarestatus.com"` at module scope in this file.

In `plugins/llm/tests/conftest.py`, the `status_plugin` fixture needs the new key and bindings. Add to `_registry` and alongside the existing constant bindings:

```python
    obj._registry["statusQueryablePages"] = []
    obj._STATUS_MAX_QUERYABLE = LLM._STATUS_MAX_QUERYABLE
    obj._status_parse_pages = LLM._status_parse_pages.__get__(obj)
    obj._status_named_pages = LLM._status_named_pages.__get__(obj)
```

and change the existing `statusPageUrls` seed to the named form so the fixture exercises the new grammar:

```python
    obj._registry = {"statusPageUrls": ["Claude=https://status.claude.com"], ...}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest plugins/llm/tests/test_status_poller.py::TestPageGrammar -q`
Expected: FAIL with `AttributeError: type object 'LLM' has no attribute '_STATUS_MAX_QUERYABLE'`

- [ ] **Step 3: Add the config key and the default's names**

In `plugins/llm/src/llm/config.py`, change the `statusPageUrls` default to the named form and add the new key after it:

```python
conf.registerGlobalValue(
    LLM,
    "statusPageUrls",
    registry.SpaceSeparatedListOfStrings(
        [
            "Claude=https://status.claude.com",
            "GitHub=https://www.githubstatus.com",
            "OpenAI=https://status.openai.com",
        ],
        _("""Space-separated status pages to poll and announce, each written as
        Name=url or as a bare url (which takes its host as its name). Both
        Atlassian Statuspage and incident.io pages work. Name is 1-32 chars of
        [A-Za-z0-9._-] and is what the model uses to ask about one service. The
        url must be a bare scheme://host with no trailing path. Unusable
        entries are dropped with a warning, duplicates collapse, and at most 5
        are polled. Empty disables polling and announcements."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "statusQueryablePages",
    registry.SpaceSeparatedListOfStrings(
        [],
        _("""Space-separated status pages the bot can be ASKED about but never
        polls or announces, same Name=url grammar as statusPageUrls. These cost
        nothing until someone asks: they are fetched lazily and cached, with no
        incident lifecycle and no channel announcements. Use this for the long
        tail; use statusPageUrls for pages a channel should be told about. At
        most 20."""),
    ),
)
```

Keep `statusAnnounce` unchanged.

- [ ] **Step 4: Implement the parser and resolver**

In `plugins/llm/src/llm/plugin.py`, add `_STATUS_MAX_QUERYABLE = 20` to the `_STATUS_*` block, add a module-level name pattern near the other regexes, and replace `_status_sources`:

```python
_STATUS_PAGE_NAME_RE = re.compile(r"\A[A-Za-z0-9._-]{1,32}\Z")
```

```python
    def _status_parse_pages(
        self, raw_entries, *, key: str, cap: int, warn: bool
    ) -> dict[str, str]:
        """Parse ``Name=url`` (or bare ``url``) entries into name -> source.

        One grammar, one parser, both config keys. Bad entries are dropped
        rather than raising: one typo must not disable the feature.

        Split on the FIRST '=' — names forbid '=' and, being in a
        space-separated list, spaces, so this is unambiguous.
        """
        pages: dict[str, str] = {}
        lowered: set[str] = set()
        for raw in raw_entries or []:
            text = str(raw).strip()
            if not text:
                continue
            name, sep, url = text.partition("=")
            if not sep:
                name, url = "", text
            source = statuspage.canonical_source(url)
            if source is None:
                if warn:
                    self.log.warning("Ignoring unusable %s entry: %s", key, text[:100])
                continue
            if not name:
                name = self._status_host(source)
            if not _STATUS_PAGE_NAME_RE.match(name):
                if warn:
                    self.log.warning("Ignoring %s entry with an unusable name: %s", key, text[:100])
                continue
            if name.lower() in lowered:
                if warn:
                    self.log.warning("Ignoring duplicate %s name: %s", key, name)
                continue
            if source in pages.values():
                # Two names for one page: the later one would look valid in
                # @config while never appearing in the enum.
                existing = next(n for n, s in pages.items() if s == source)
                if warn:
                    self.log.warning(
                        "Ignoring %s entry %s: same page as %s", key, name, existing
                    )
                continue
            pages[name] = source
            lowered.add(name.lower())
        if len(pages) > cap:
            if warn:
                self.log.warning("%s lists %i usable pages; using the first %i", key, len(pages), cap)
            pages = dict(list(pages.items())[:cap])
        return pages

    def _status_polled_pages(self, *, warn: bool = True) -> dict[str, str]:
        return self._status_parse_pages(
            self.registryValue("statusPageUrls"),
            key="statusPageUrls",
            cap=self._STATUS_MAX_SOURCES,
            warn=warn,
        )

    def _status_named_pages(self, *, warn: bool = True) -> dict[str, str]:
        """Name -> canonical source for every configured page, polled first.

        The single source of truth for the tool's ``service`` enum, for
        resolving that argument, and for the prune sets. Purely a function of
        config: nothing here may come from a fetched payload, or a page could
        rename itself into another page's selector.
        """
        pages = self._status_polled_pages(warn=warn)
        polled_sources = set(pages.values())
        lowered = {n.lower() for n in pages}
        for name, source in self._status_parse_pages(
            self.registryValue("statusQueryablePages"),
            key="statusQueryablePages",
            cap=self._STATUS_MAX_QUERYABLE,
            warn=warn,
        ).items():
            if source in polled_sources or name.lower() in lowered:
                if warn:
                    self.log.warning("Ignoring queryable page %s: already configured", name)
                continue
            pages[name] = source
        return pages

    def _status_sources(self, *, warn: bool = True) -> list[str]:
        """Canonical, deduplicated, capped list of POLLED status pages.

        Order is the operator's. ``warn`` gates diagnostics and defaults to on
        for the poller's ~2-minute cadence; request-path callers pass False, or
        one typo'd entry logs once per chat message.
        """
        return list(self._status_polled_pages(warn=warn).values())
```

`re` is already imported in `plugin.py`; confirm before adding it.

- [ ] **Step 5: Prune against the right sets**

Replace `_status_prune_sources` (`plugin.py:1115-1133`):

```python
    def _status_prune_sources(
        self, sources: list[str], queryable: list[str] | None = None
    ) -> None:
        """Drop state for sources no longer configured.

        Two different keep-sets, deliberately. Lifecycle state is pruned
        against the POLLED set: a queryable page must never hold any. The
        history and query caches are pruned against polled UNION queryable —
        pruning history against the polled set alone deletes an allowlisted
        page's history (up to 4 MB, cached for an hour) on the very next poll,
        120 seconds after it was fetched, backoff and all.
        """
        polled = set(sources)
        both = polled | set(queryable or ())
        for holder, keep in (
            (self._status_state, polled),
            (self._status_read_cache, polled),
            (self._status_last_fetch, polled),
            (self._status_history_cache, both),
            (self._status_history_at, both),
            (self._status_history_failed_at, both),
        ):
            for stale in [k for k in holder if k not in keep]:
                del holder[stale]
```

The two query dicts join this tuple in Task 2. Update the poll's call site in `_run_status_poll` to pass the queryable sources:

```python
            sources = self._status_sources()
            queryable = [
                s for s in self._status_named_pages().values() if s not in set(sources)
            ]
            self._status_prune_sources(sources, queryable)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest plugins/llm/tests/test_status_poller.py -q`
Expected: PASS. Existing tests seeding `_registry["statusPageUrls"]` with bare URLs still work — bare URLs remain valid.

- [ ] **Step 7: Update the config test**

`plugins/llm/tests/test_config.py`'s default assertion now expects the named form:

```python
        assert list(conf.supybot.plugins.LLM.statusPageUrls()) == [
            "Claude=https://status.claude.com",
            "GitHub=https://www.githubstatus.com",
            "OpenAI=https://status.openai.com",
        ]

    def test_status_queryable_pages_defaults_empty(self) -> None:
        import llm.config  # noqa: F401
        import supybot.conf as conf

        assert list(conf.supybot.plugins.LLM.statusQueryablePages()) == []
```

Run: `uv run pytest plugins/llm/tests/test_config.py -q` — expected PASS.

- [ ] **Step 8: Commit**

```bash
git add plugins/llm/src/llm/config.py plugins/llm/src/llm/plugin.py \
        plugins/llm/tests/test_status_poller.py plugins/llm/tests/test_config.py \
        plugins/llm/tests/conftest.py
git commit -m "feat(status): one Name=url grammar for both page keys

Adds statusQueryablePages and gives statusPageUrls optional names, parsed by
a single function. Selector names must come from config alone — deriving them
from a page's own page_name would let a compromised page capture another
page's name. Pruning now uses two keep-sets: lifecycle against polled, history
against polled union queryable."
```

---

### Task 2: The query cache

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (state init ~880, `_status_fetch_snapshot` ~1184, prune)
- Test: `plugins/llm/tests/test_status_tool.py`

**Interfaces:**
- Consumes: `_status_named_pages`, `_status_parse_pages` from Task 1.
- Produces: `_status_query_cache: dict[str, statuspage.Snapshot]`, `_status_query_failed_at: dict[str, float]`.
- Produces: `LLM._status_query_snapshot(self, source: str, *, deadline: float | None = None) -> statuspage.Snapshot | None`.
- Produces: `_status_fetch_snapshot` gains `cached: statuspage.Snapshot | None = None`.
- Produces: `_STATUS_QUERY_TTL = 300`, `_STATUS_QUERY_CACHE_MAX = 20`.

- [ ] **Step 1: Write the failing tests**

Add to `plugins/llm/tests/test_status_tool.py`:

```python
CF = "https://www.cloudflarestatus.com"


class TestQueryCache:
    def test_first_call_fetches_and_caches(self, status_plugin):
        plugin = status_plugin
        plugin._status_query_snapshot = LLM._status_query_snapshot.__get__(plugin)
        assert plugin._status_query_snapshot(CF) is not None
        assert plugin._fetch_calls == 1
        assert CF in plugin._status_query_cache

    def test_second_call_inside_the_ttl_does_not_refetch(self, status_plugin):
        plugin = status_plugin
        plugin._status_query_snapshot = LLM._status_query_snapshot.__get__(plugin)
        plugin._status_query_snapshot(CF)
        plugin._now += plugin._STATUS_QUERY_TTL - 1
        plugin._status_query_snapshot(CF)
        assert plugin._fetch_calls == 1

    def test_past_the_ttl_refetches(self, status_plugin):
        plugin = status_plugin
        plugin._status_query_snapshot = LLM._status_query_snapshot.__get__(plugin)
        plugin._status_query_snapshot(CF)
        plugin._now += plugin._STATUS_QUERY_TTL + 1
        plugin._status_query_snapshot(CF)
        assert plugin._fetch_calls == 2

    def test_failure_is_backed_off(self, status_plugin):
        plugin = status_plugin
        plugin._status_query_snapshot = LLM._status_query_snapshot.__get__(plugin)
        plugin._fake_error = statuspage.FetchError("down")
        assert plugin._status_query_snapshot(CF) is None
        plugin._now += 1
        assert plugin._status_query_snapshot(CF) is None
        assert plugin._fetch_calls == 1, "backoff did not hold"

    def test_a_full_cycle_of_the_cap_does_not_thrash(self, status_plugin):
        """A cache smaller than the allowlist evicts every entry before it is
        reused, so every request fetches despite the TTL."""
        plugin = status_plugin
        plugin._status_query_snapshot = LLM._status_query_snapshot.__get__(plugin)
        pages = [f"https://status{i}.example.com" for i in range(plugin._STATUS_QUERY_CACHE_MAX)]
        for p in pages:
            plugin._status_query_snapshot(p)
        first_pass = plugin._fetch_calls
        for p in pages:
            plugin._status_query_snapshot(p)
        assert plugin._fetch_calls == first_pass, "second pass should be all cache hits"

    def test_cache_evicts_the_oldest_past_capacity(self, status_plugin):
        plugin = status_plugin
        plugin._status_query_snapshot = LLM._status_query_snapshot.__get__(plugin)
        for i in range(plugin._STATUS_QUERY_CACHE_MAX + 1):
            plugin._now += 1
            plugin._status_query_snapshot(f"https://status{i}.example.com")
        assert len(plugin._status_query_cache) == plugin._STATUS_QUERY_CACHE_MAX
        assert "https://status0.example.com" not in plugin._status_query_cache

    def test_conditional_get_uses_the_query_cache_validators(self, status_plugin, mocker):
        """_status_fetch_snapshot reads ETag from _status_read_cache, which a
        queryable page never populates — so without an explicit cached= the
        refresh is an unconditional full GET."""
        plugin = status_plugin
        plugin._status_query_snapshot = LLM._status_query_snapshot.__get__(plugin)
        plugin._status_fetch_snapshot = LLM._status_fetch_snapshot.__get__(plugin)
        fetch = mocker.patch(
            "llm.plugin.statuspage.fetch_summary",
            return_value=statuspage.FetchResult(
                payload={
                    "page": {"name": "CF", "url": CF},
                    "status": {"indicator": "none", "description": "ok"},
                    "components": [],
                    "incidents": [],
                    "scheduled_maintenances": [],
                },
                etag='W/"abc"',
                modified=None,
                not_modified=False,
            ),
        )
        plugin._status_query_snapshot(CF)
        plugin._now += plugin._STATUS_QUERY_TTL + 1
        plugin._status_query_snapshot(CF)
        assert fetch.call_args.kwargs["etag"] == 'W/"abc"'
```

Check `statuspage.FetchResult`'s actual field names before writing that construction; adjust to match.

Add to `conftest.py`'s `status_plugin`:

```python
    obj._status_query_cache = {}
    obj._status_query_failed_at = {}
    obj._STATUS_QUERY_TTL = LLM._STATUS_QUERY_TTL
    obj._STATUS_QUERY_CACHE_MAX = LLM._STATUS_QUERY_CACHE_MAX
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest plugins/llm/tests/test_status_tool.py::TestQueryCache -q`
Expected: FAIL with `AttributeError: type object 'LLM' has no attribute '_status_query_snapshot'`

- [ ] **Step 3: Add state and constants**

In `plugin.py`'s status state block (~line 880), after the history dicts:

```python
        # Queryable-only pages: lazily filled, never polled, never announced,
        # and never granted lifecycle state.
        self._status_query_cache: dict[str, statuspage.Snapshot] = {}
        self._status_query_failed_at: dict[str, float] = {}
```

In the `_STATUS_*` block:

```python
    # Queryable pages are refreshed only when asked for, so their TTL is
    # shorter than the 600s staleness line — nothing else refreshes them.
    _STATUS_QUERY_TTL = 300
    # Equal to _STATUS_MAX_QUERYABLE on purpose. A cache smaller than the
    # allowlist thrashes: cycling every page inside the TTL evicts each entry
    # before it is reused, so every request fetches despite the cache.
    _STATUS_QUERY_CACHE_MAX = 20
```

- [ ] **Step 4: Give `_status_fetch_snapshot` explicit validators**

Change its signature and the two lines that read the read cache:

```python
    def _status_fetch_snapshot(
        self,
        source: str,
        *,
        timeout_cap: float | None = None,
        cached: statuspage.Snapshot | None = None,
    ) -> statuspage.Snapshot:
```

```python
        if cached is None:
            cached = self._status_read_cache.get(source)
```

Everything else in the method is unchanged — the poller keeps its existing behaviour by passing nothing.

- [ ] **Step 5: Implement the query fetch**

```python
    def _status_query_snapshot(
        self, source: str, *, deadline: float | None = None
    ) -> statuspage.Snapshot | None:
        """Cached reading for a page we never poll.

        Writes only the two query dicts — never _status_state, never
        _status_read_cache. A queryable page has no lifecycle, so there is
        nothing to announce and nothing a question could consume.
        """
        now = self._status_now()
        cached = self._status_query_cache.get(source)
        if cached is not None and (now - cached.fetched_at) < self._STATUS_QUERY_TTL:
            return cached
        if now - self._status_query_failed_at.get(source, 0.0) < self._STATUS_HISTORY_RETRY:
            return cached
        timeout_cap = None
        if deadline is not None:
            timeout_cap = deadline - self._status_monotonic()
            if timeout_cap <= self._STATUS_MIN_FETCH_WINDOW:
                return cached
        try:
            snapshot = self._status_fetch_snapshot(
                source, timeout_cap=timeout_cap, cached=cached
            )
        except Exception as e:
            self.log.info("Status query fetch failed for %s: %s", source, e)
            self._status_query_failed_at[source] = now
            return cached
        self._status_query_cache[source] = snapshot
        self._status_query_failed_at.pop(source, None)
        self._status_evict_query_cache()
        return snapshot

    def _status_evict_query_cache(self) -> None:
        """Keep the newest _STATUS_QUERY_CACHE_MAX readings."""
        excess = len(self._status_query_cache) - self._STATUS_QUERY_CACHE_MAX
        if excess <= 0:
            return
        oldest = sorted(self._status_query_cache.items(), key=lambda kv: kv[1].fetched_at)
        for source, _snap in oldest[:excess]:
            del self._status_query_cache[source]
```

- [ ] **Step 6: Add the two dicts to the prune**

In `_status_prune_sources`, extend the tuple with `(self._status_query_cache, both)` and `(self._status_query_failed_at, both)`. Update its docstring to say eight structures.

Extend `test_pruning_clears_every_keyed_structure` in `test_status_poller.py` to enumerate all eight — it currently names six, so it cannot detect growth in the new dicts.

- [ ] **Step 7: Run the tests**

Run: `uv run pytest plugins/llm/tests/test_status_tool.py::TestQueryCache plugins/llm/tests/test_status_poller.py -q`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/conftest.py \
        plugins/llm/tests/test_status_tool.py plugins/llm/tests/test_status_poller.py
git commit -m "feat(status): lazy cache for pages we never poll

Queryable pages get their own TTL cache and failure backoff, bounded equal to
the allowlist cap so a full cycle inside the TTL cannot thrash. Both dicts
join the prune. _status_fetch_snapshot takes explicit validators so a query
refresh is still a conditional GET."
```

---

### Task 3: `service` selection in the payload

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py:1310+` (`_status_tool_payload`)
- Test: `plugins/llm/tests/test_status_tool.py`

**Interfaces:**
- Consumes: `_status_named_pages`, `_status_query_snapshot`.
- Produces: `_status_tool_payload(self, *, service: str | None = None, include_history: bool = False, pages: dict[str, str] | None = None) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
class TestServiceSelection:
    def _plugin(self, status_plugin):
        p = status_plugin
        p._registry["statusPageUrls"] = ["Claude=https://status.claude.com"]
        p._registry["statusQueryablePages"] = ["CF=https://www.cloudflarestatus.com"]
        p._status_tool_payload = LLM._status_tool_payload.__get__(p)
        p._status_query_snapshot = LLM._status_query_snapshot.__get__(p)
        return p

    def test_omitted_service_returns_every_polled_source(self, status_plugin):
        plugin = self._plugin(status_plugin)
        plugin._status_read_cache = {"https://status.claude.com": green_snapshot(plugin._now)}
        payload = plugin._status_tool_payload()
        assert [e["source"] for e in payload["services"]] == ["status.claude.com"]

    def test_named_polled_page_returns_only_that_one(self, status_plugin):
        plugin = self._plugin(status_plugin)
        plugin._status_read_cache = {"https://status.claude.com": green_snapshot(plugin._now)}
        payload = plugin._status_tool_payload(service="Claude")
        assert len(payload["services"]) == 1
        assert payload["services"][0]["source"] == "status.claude.com"

    def test_named_queryable_page_fetches_lazily(self, status_plugin):
        plugin = self._plugin(status_plugin)
        payload = plugin._status_tool_payload(service="CF")
        assert [e["source"] for e in payload["services"]] == ["www.cloudflarestatus.com"]
        assert plugin._fetch_calls == 1
        assert plugin._status_state == {}, "a queryable page must not gain lifecycle state"

    def test_unresolvable_service_errors_but_still_returns_polled_data(self, status_plugin):
        """service.py records any dict without "error" as a SUCCESSFUL tool
        call, so a bare note would let the model summarise unrelated healthy
        services as the requested one's state."""
        plugin = self._plugin(status_plugin)
        plugin._status_read_cache = {"https://status.claude.com": green_snapshot(plugin._now)}
        payload = plugin._status_tool_payload(service="Nope")
        assert "error" in payload
        assert "Nope" in payload["error"]
        assert payload["services"], "polled data should still ride along"

    def test_service_resolution_uses_the_frozen_mapping_when_given(self, status_plugin):
        """The enum and the dispatcher must be one snapshot; config changing
        mid-completion must not reroute an already-issued call."""
        plugin = self._plugin(status_plugin)
        frozen = {"CF": "https://www.cloudflarestatus.com"}
        plugin._registry["statusQueryablePages"] = ["CF=https://other.example.com"]
        payload = plugin._status_tool_payload(service="CF", pages=frozen)
        assert payload["services"][0]["source"] == "www.cloudflarestatus.com"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest plugins/llm/tests/test_status_tool.py::TestServiceSelection -q`
Expected: FAIL — `_status_tool_payload() got an unexpected keyword argument 'service'`

- [ ] **Step 3: Implement**

Change the signature and add the selection branch before the existing polled loop. Keep the existing polled path byte-for-byte in the `service is None` case — it is pinned by tests and by production behaviour.

```python
    def _status_tool_payload(
        self,
        *,
        service: str | None = None,
        include_history: bool = False,
        pages: dict[str, str] | None = None,
    ) -> dict[str, Any]:
```

Inside, before the polled loop:

```python
        # The mapping the tool schema was built from, when the caller froze one.
        # Resolving live here instead would let config churn between the
        # model's call and its dispatch route the call somewhere the enum
        # never advertised.
        named = pages if pages is not None else self._status_named_pages(warn=False)

        if service is not None:
            source = next(
                (s for n, s in named.items() if n.lower() == service.strip().lower()), None
            )
            if source is None:
                payload = self._status_tool_payload(include_history=include_history, pages=named)
                payload["error"] = (
                    f"No status page named {statuspage.sanitise_text(service, limit=40)!r} "
                    "is configured; the services listed are the ones that are."
                )
                return payload
            return self._status_single_payload(
                source, include_history=include_history, polled=source in set(self._status_sources(warn=False))
            )
```

Factor the per-source entry construction out of the existing loop into
`_status_single_payload(source, *, include_history, polled)` so both paths share it: a
polled source reads `_status_read_cache` under the staleness rule, a queryable one calls
`_status_query_snapshot`. Return `{"services": [entry], "note": statuspage.UNTRUSTED_NOTE}`,
plus a top-level `"error"` when the single entry could not be read.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest plugins/llm/tests/test_status_tool.py -q`
Expected: PASS, including every pre-existing payload test — the `service is None` path is unchanged.

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_status_tool.py
git commit -m "feat(status): answer about one named page on request

service omitted keeps returning every polled source. Named, it returns that
page alone, fetching a queryable one lazily. An unresolvable name returns the
polled set AND a top-level error — a bare note would be recorded as a
successful tool call and invite the model to answer from the wrong service."
```

---

### Task 4: Schema enum, gating, and the frozen mapping

**Files:**
- Modify: `plugins/llm/src/llm/service.py:126-155`, `:5197-5233`
- Modify: `plugins/llm/src/llm/assistant.py` (schema ~509-531, handler ~1102-1119)
- Test: `plugins/llm/tests/test_status_tool.py`

**Interfaces:**
- Produces: `service._with_status_context(tools, sources, pages) -> list[dict]`, replacing `_with_status_hosts`.
- Produces: `check_service_status` gains an optional `service` string property, injected per build.

- [ ] **Step 1: Write the failing tests**

```python
class TestServiceEnum:
    def test_enum_lists_configured_names(self):
        from llm.assistant import get_tools_for_profile
        from llm.service import _with_status_context

        patched = _with_status_context(
            get_tools_for_profile("chat"),
            ["https://status.claude.com"],
            {"Claude": "https://status.claude.com", "CF": "https://www.cloudflarestatus.com"},
        )
        fn = next(t["function"] for t in patched if t["function"]["name"] == "check_service_status")
        assert fn["parameters"]["properties"]["service"]["enum"] == ["Claude", "CF"]

    def test_module_schema_is_not_mutated_at_any_depth(self):
        """The shipped two-level copy shares `parameters` and `properties`, so
        writing a property into them would corrupt the process-wide schema."""
        from llm.assistant import get_tools_for_profile
        from llm.service import _with_status_context

        def props():
            fn = next(
                t["function"] for t in get_tools_for_profile("chat")
                if t["function"]["name"] == "check_service_status"
            )
            return dict(fn["parameters"]["properties"])

        before = props()
        for _ in range(3):
            _with_status_context(
                get_tools_for_profile("chat"),
                ["https://status.claude.com"],
                {"Claude": "https://status.claude.com"},
            )
        assert props() == before
        assert "service" not in props()

    def test_no_pages_omits_the_property_entirely(self):
        from llm.assistant import get_tools_for_profile
        from llm.service import _with_status_context

        patched = _with_status_context(get_tools_for_profile("chat"), [], {})
        fn = next(t["function"] for t in patched if t["function"]["name"] == "check_service_status")
        assert "service" not in fn["parameters"]["properties"]


class TestQueryableOnlyGate:
    def test_tool_is_wired_with_no_polled_pages(self, mocker, make_service):
        """The premise of the feature: polled and queryable are independent.
        The shipped gate keys on the polled list alone, so this config would
        expose no tool at all."""
        service, plugin = make_service(
            statusPageUrls=[], statusQueryablePages=["CF=https://www.cloudflarestatus.com"]
        )
        for attr in ("_STATUS_MAX_SOURCES", "_STATUS_MAX_QUERYABLE"):
            setattr(plugin, attr, getattr(LLM, attr))
        for meth in ("_status_parse_pages", "_status_named_pages", "_status_sources", "_status_host"):
            setattr(plugin, meth, getattr(LLM, meth).__get__(plugin))
        plugin._status_tool_payload = mocker.Mock(name="_status_tool_payload")
        executor_spy = mocker.spy(llm.service, "AssistantToolExecutor")
        mocker.patch(
            "llm.service.litellm.completion",
            return_value=make_completion_response("hi"),
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        service.assistant_completion(prompt="hi", nick="tester", channel="#test")

        assert executor_spy.call_args.kwargs["status_fn"] is not None
        tools = mocker.patch  # replaced below
```

Finish that test against the existing `TestToolWiringGate` / `TestToolSchemaGateOnConfig`
pattern in the same file — they already show how to spy the executor and how to read the
tool list off the captured `litellm.completion` call. Two assertions are required: that
`status_fn` is not `None`, and that `check_service_status` appears in the tool list the
completion was given.

Bind the **real** resolver methods, as above, rather than stubbing them with a lambda: a
prior review on this feature found that stubbing this exact seam made the gate
structurally unable to detect a deleted registry key, which is how a broken commit
reached production.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest plugins/llm/tests/test_status_tool.py::TestServiceEnum -q`
Expected: FAIL — `cannot import name '_with_status_context'`

- [ ] **Step 3: Add the schema property**

In `assistant.py`, add `service` to `check_service_status`'s properties with no `enum` (injection adds it) and extend the description to explain when to use it:

```python
                    "service": {
                        "type": "string",
                        "description": (
                            "Name of ONE configured service to report on, from the "
                            "enum. Omit it to get every monitored service at once — "
                            "that is the right choice for a general question like "
                            "'is anything down?' or one naming several services. Use "
                            "it only when the user asks about one specific service "
                            "that is not among the monitored ones."
                        ),
                    },
```

Update the handler at `assistant.py:1102` to forward it:

```python
            service = _arguments.get("service")
            service = service.strip() if isinstance(service, str) and service.strip() else None
            return json.dumps(
                self._status_fn(include_history=include_history, service=service)
            )
```

- [ ] **Step 4: Implement `_with_status_context`**

Replace `_with_status_hosts` in `service.py`:

```python
def _with_status_context(
    tools: list[dict], sources: list[str], pages: dict[str, str]
) -> list[dict]:
    """Name the configured pages in the description and constrain `service`.

    Copies FOUR levels — tool, function, parameters, properties. ToolSpec's
    `as_tool()` returns a fresh outer dict but shares the module-level schema
    as `function`, and `parameters`/`properties` beneath it are shared too.
    Writing a property into them would add `service` to the process-wide
    schema permanently, and a later build that should omit it would inherit it.

    The enum comes only from operator config: it is part of the cached prompt
    prefix, and a page that could name itself would both churn that cache and
    be able to capture another page's selector.
    """
    if not sources and not pages:
        return tools
    hosts = ", ".join(urlparse(s).hostname or s for s in sources)
    patched = []
    for tool in tools:
        fn = tool.get("function") or {}
        if fn.get("name") != "check_service_status":
            patched.append(tool)
            continue
        params = {**(fn.get("parameters") or {})}
        props = {**(params.get("properties") or {})}
        if pages:
            props["service"] = {**props.get("service", {}), "enum": list(pages)}
        else:
            props.pop("service", None)
        params["properties"] = props
        description = fn["description"]
        if hosts:
            description = f"{description} Monitored services: {hosts}."
        patched.append({**tool, "function": {**fn, "description": description, "parameters": params}})
    return patched
```

- [ ] **Step 5: Wire the gate and freeze the mapping**

At the profile-build site in `service.py`, replace the polled-only gate:

```python
            status_sources = self.plugin._status_sources(warn=False)
            # The whole point of the queryable allowlist is that it works with
            # no polled pages at all, so the gate is polled OR queryable.
            status_pages = self.plugin._status_named_pages(warn=False)
```

```python
                status_fn=(
                    functools.partial(self.plugin._status_tool_payload, pages=status_pages)
                    if status_pages
                    else None
                ),
```

```python
            if not status_pages:
                exclude_tools = exclude_tools | {"check_service_status"}
            profile_tools = get_tools_for_profile(profile.id, exclude=exclude_tools)
            profile_tools = _with_status_context(profile_tools, status_sources, status_pages)
```

Binding `pages` with `functools.partial` is what makes the schema and the dispatcher one
snapshot: `profile_tools` is built once and reused across every turn of the tool loop, and
the executor does no schema validation before dispatch. Confirm `functools` is imported.

- [ ] **Step 6: Run the tests**

Run: `uv run pytest plugins/llm/tests/test_status_tool.py -q`
Expected: PASS. Two pre-existing tests must be updated: one asserts `include_history` is the schema's only property, another asserts a `service` argument is ignored by the handler. Update both to the new contract rather than deleting them.

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest plugins/llm/tests/ -q`
Expected: PASS, at least 3130.

- [ ] **Step 8: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/src/llm/assistant.py \
        plugins/llm/tests/test_status_tool.py
git commit -m "feat(status): enum-constrained service argument

_with_status_context copies four dict levels so injecting a property cannot
corrupt the module-level schema, gates the tool on polled OR queryable, and
binds the resolved mapping to the callback so the schema and the dispatcher
are one snapshot."
```

---

### Task 5: Documentation and verification

**Files:**
- Modify: `docs/guide/operator/configuration.md`, `docs/guide/user/service-status.md`
- Verify: whole suite, strict docs build

- [ ] **Step 1: Operator docs**

Add `statusQueryablePages` to the status table with its default (empty), the `Name=url` grammar, the 20 cap, and the distinction that matters: pages here are answered about but never announced, and cost nothing until asked for. Note that `statusPageUrls` now accepts names and that a bare URL still works, taking its host as its name.

- [ ] **Step 2: User docs**

State that the bot can be asked about services it does not announce, and that asking about one service by name returns just that one. Do not promise the model always picks the right entry — the enum constrains the value, not the model's choice of when to use it.

Repo doc style: `@` prefix, en-CA, sentence-case headings, "authenticated" not NickServ terminology.

- [ ] **Step 3: Verify**

Run: `uv run pytest plugins/llm/tests/ -q` — expected PASS, at least 3130.
Run: `uv run mkdocs build --strict` — expected no warnings.

- [ ] **Step 4: Commit**

```bash
git add docs/
git commit -m "docs(status): document the queryable allowlist

statusQueryablePages holds pages the bot answers about but never announces,
and statusPageUrls now accepts optional names."
```

---

## Deployment

Auto-deploy: CI green → Docker build → the 15-minute updater timer restarts the service.

`statusQueryablePages` defaults to empty, so the deploy changes nothing observable.

`statusPageUrls`' default gains names, but **that default will not reach prod** — `bot.conf` holds the persisted three-URL line and overrides it. Bare URLs remain valid and take their host as their name, so the deploy is safe with no config change; the enum simply reads `status.claude.com` rather than `Claude` until the line is rewritten. To adopt names or add queryable pages, use `@config` or stop-edit-start.

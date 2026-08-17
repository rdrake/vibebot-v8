# incident.io support implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the status feature read incident.io pages, specifically `status.openai.com`, by relaxing three structural guards and one vocabulary check in `parse_summary`.

**Architecture:** incident.io serves an Atlassian-compatible shim at the same `/api/v2/*` paths, so there is no second provider and no parse branch. `parse_summary` stops requiring optional collections to be present, and stops rejecting an entire page over one unrecognised component status. Everything downstream — `classify`, the poller, the announcer, the tool payload — is untouched.

**Tech Stack:** Python 3.14, Limnoria plugin framework, pytest + pytest-mock, `uv`, ruff + ty via pre-commit.

**Spec:** `docs/superpowers/specs/2026-08-17-incident-io-support-design.md`

## Global constraints

- **`parse_summary` is the only function whose behaviour changes.** `_parse_incident`, `classify`, `to_tool_payload`, `to_history_payload`, `parse_incidents`, `render_line`, `render_resolved_line` and the whole fetch stack are untouched.
- **Absent is not malformed.** A missing `incidents` / `scheduled_maintenances` / `components` key becomes `[]`. A key that is *present but not a list* still raises `InvalidPayload`. Every relaxation needs a paired test proving the malformed case still rejects.
- **Unknown component status keeps the component; unknown incident status still rejects.** The asymmetry is deliberate: `INCIDENT_STATUSES` drives `TERMINAL_STATUSES`, and misreading an incident as live or over corrupts the announce lifecycle. A wrong component label cannot.
- **Structural strictness is unchanged.** A component that is not an object, or whose `name` or `status` is not a string, still rejects.
- **Logging uses `%i`, never `%d`.** `supybot.utils.str.format` has no `%d`; the token is left in literally and positional args shift left.
- **Do not push.** Commit locally; the orchestrator handles pushes.
- The suite is 3083 passing, 0 failing. Report the exact count.

## File structure

| File | Responsibility | Change |
|---|---|---|
| `plugins/llm/src/llm/statuspage.py` | Pure parse/classify/render | `parse_summary` only, ~8 lines |
| `plugins/llm/src/llm/config.py` | Registry defaults | Add OpenAI to `statusPageUrls` |
| `plugins/llm/tests/test_statuspage_parse.py` | Parser strictness | Invert absent-key tests, add relaxation cases |
| `plugins/llm/tests/test_statuspage_payload.py` | Tool payload | Unknown status reaches `degraded` |
| `plugins/llm/tests/test_config.py` | Registry registration | Three-source default |
| `docs/guide/user/service-status.md` | User docs | incident.io works; OpenAI monitored |

---

### Task 1: Relax the parser

**Files:**
- Modify: `plugins/llm/src/llm/statuspage.py:214-235`
- Test: `plugins/llm/tests/test_statuspage_parse.py`, `plugins/llm/tests/test_statuspage_payload.py`

**Interfaces:**
- Produces: `parse_summary` accepts a payload with `incidents`, `scheduled_maintenances` or `components` absent, and a payload whose component statuses are outside `COMPONENT_STATUSES`. Its signature and return type are unchanged.

- [ ] **Step 1: Audit the existing strictness tests before touching anything**

Read `plugins/llm/tests/test_statuspage_parse.py` and list every test that asserts `InvalidPayload`. Classify each as:
- **absent-key** ("rejects when `incidents` is missing") — these now assert the wrong thing and must be inverted to assert the payload parses with zero incidents.
- **malformed** ("rejects when `incidents` is a string") — these must stay exactly as they are.

Write the classification into your report before editing. Do not delete any test in this file: an absent-key test becomes a parses-successfully test, it does not disappear. A deleted test is indistinguishable from a coverage gap six months from now.

- [ ] **Step 2: Write the failing tests**

Add to `plugins/llm/tests/test_statuspage_parse.py`:

```python
class TestOptionalCollectionsMayBeAbsent:
    """incident.io omits empty collections rather than sending []. Absence is
    not a structural violation; a present-but-wrong-type value still is."""

    def _base(self) -> dict:
        return {
            "page": {"name": "OpenAI", "url": "https://status.openai.com/"},
            "status": {"indicator": "none", "description": "All Systems Operational"},
            "components": [{"name": "API", "status": "operational"}],
            "incidents": [],
            "scheduled_maintenances": [],
        }

    @pytest.mark.parametrize("key", ["incidents", "scheduled_maintenances", "components"])
    def test_absent_collection_parses_as_empty(self, key):
        payload = self._base()
        del payload[key]
        snap = statuspage.parse_summary(payload, fetched_at=1000.0)
        assert snap.incidents == {}
        if key == "components":
            assert snap.components == {}

    @pytest.mark.parametrize("key", ["incidents", "scheduled_maintenances", "components"])
    @pytest.mark.parametrize("bad", ["not a list", 42, {"a": 1}])
    def test_present_but_not_a_list_still_rejects(self, key, bad):
        payload = self._base()
        payload[key] = bad
        with pytest.raises(statuspage.InvalidPayload):
            statuspage.parse_summary(payload, fetched_at=1000.0)


class TestUnknownComponentStatus:
    """Rejecting the whole page over one unrecognised status is worst-case
    timed: it fires during an outage, the only time anyone asks."""

    def _payload(self, comp_status: str) -> dict:
        return {
            "page": {"name": "OpenAI", "url": "https://status.openai.com/"},
            "status": {"indicator": "minor", "description": "Partial outage"},
            "components": [
                {"name": "API", "status": comp_status},
                {"name": "Dashboard", "status": "operational"},
            ],
            "incidents": [],
            "scheduled_maintenances": [],
        }

    def test_unknown_status_keeps_the_component(self):
        snap = statuspage.parse_summary(self._payload("degraded"), fetched_at=1000.0)
        assert snap.components["API"] == "degraded"
        assert snap.components["Dashboard"] == "operational"

    def test_unknown_status_still_reaches_the_model_as_degraded(self):
        snap = statuspage.parse_summary(self._payload("degraded"), fetched_at=1000.0)
        payload = statuspage.to_tool_payload(snap, now=1000.0)
        assert {"name": "API", "status": "degraded"} in payload["degraded"]
        assert all(d["name"] != "Dashboard" for d in payload["degraded"])

    def test_structural_violations_still_reject(self):
        for bad_components in (
            [{"name": "API"}],                       # no status
            [{"status": "operational"}],             # no name
            [{"name": 5, "status": "operational"}],  # non-string name
            [{"name": "API", "status": 5}],          # non-string status
            ["not an object"],
        ):
            payload = self._payload("operational")
            payload["components"] = bad_components
            with pytest.raises(statuspage.InvalidPayload):
                statuspage.parse_summary(payload, fetched_at=1000.0)


class TestIncidentStatusStaysStrict:
    """Deliberate asymmetry: INCIDENT_STATUSES drives TERMINAL_STATUSES, so
    misreading an incident as live or over corrupts the announce lifecycle in
    a way a wrong component label cannot."""

    def test_unknown_incident_status_still_rejects(self):
        payload = {
            "page": {"name": "OpenAI", "url": "https://status.openai.com/"},
            "status": {"indicator": "minor", "description": "Partial outage"},
            "components": [],
            "incidents": [{"id": "abc", "status": "brewing", "name": "X"}],
            "scheduled_maintenances": [],
        }
        with pytest.raises(statuspage.InvalidPayload):
            statuspage.parse_summary(payload, fetched_at=1000.0)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest plugins/llm/tests/test_statuspage_parse.py -q`
Expected: the two new relaxation classes FAIL with `InvalidPayload: incidents is not a list` and `bad component entry: 'API'/'degraded'`. `TestIncidentStatusStaysStrict` should PASS already — it pins existing behaviour, and a green result there is the point.

- [ ] **Step 4: Implement the relaxation**

In `plugins/llm/src/llm/statuspage.py`, replace lines 214-216:

```python
    # incident.io's Atlassian-compatible shim OMITS an empty collection rather
    # than sending []. Absence is not a structural violation; a present value
    # of the wrong type still is, so _require_list keeps guarding that.
    raw_components = _require_list(root.get("components", []), "components")
    _require_list(root.get("incidents", []), "incidents")
    _require_list(root.get("scheduled_maintenances", []), "scheduled_maintenances")
```

Replace the component loop's vocabulary check (lines 224-229) so an unknown status is kept rather than fatal:

```python
        if not isinstance(name, str) or not isinstance(comp_status, str):
            raise InvalidPayload(f"bad component entry: {name!r}/{comp_status!r}")
        # An unrecognised status keeps the component instead of rejecting the
        # page. That rejection was worst-case timed — it would fire during an
        # outage, the only time anyone asks — and the alternative of dropping
        # the component fails silently in the worse direction, reporting "all
        # operational" precisely because the broken one was discarded.
        # to_tool_payload lists anything != "operational" in `degraded`, so an
        # unfamiliar value still reaches the model, which reads prose anyway.
        components[name] = comp_status
```

and change line 233's direct index to tolerate absence:

```python
    for item in root.get("incidents", []):
```

`COMPONENT_STATUSES` is now unused by `parse_summary`. Do NOT delete the constant — check whether anything else imports it first, and report what you find.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest plugins/llm/tests/test_statuspage_parse.py plugins/llm/tests/test_statuspage_payload.py -q`
Expected: PASS, including every inverted absent-key test from Step 1.

- [ ] **Step 6: Verify against the real OpenAI payload**

Run this and paste the output into your report:

```bash
curl -s --max-time 20 https://status.openai.com/api/v2/summary.json -o /tmp/oai.json
uv run python3 -c "
import json, sys
sys.path.insert(0, 'plugins/llm/src')
from llm import statuspage
d = json.load(open('/tmp/oai.json'))
s = statuspage.parse_summary(d, fetched_at=1000.0)
print('page_name =', s.page_name, '| indicator =', s.indicator)
print('components =', len(s.components), '| incidents =', len(s.incidents))
print(json.dumps(statuspage.to_tool_payload(s, now=1000.0))[:300])
"
```

Expected: parses, `page_name = OpenAI`. If it does not, stop and report — the spec's central claim is wrong and the plan needs revisiting.

- [ ] **Step 7: Add a regression fixture from the real shape**

The live probe is not a test. Add to `test_statuspage_parse.py` a fixture reproducing the *shape* observed on 2026-08-17 — `page` / `status` / `components` present, `incidents` and `scheduled_maintenances` absent, 25 components of which two share a name — and assert it parses, that `page_name == "OpenAI"`, and that the duplicate name collapses to a single dict key. Do not embed the full 25-component payload; a representative handful with one deliberate duplicate is enough.

- [ ] **Step 8: Run the full suite**

Run: `uv run pytest plugins/llm/tests/ -q`
Expected: PASS, at least 3083.

- [ ] **Step 9: Commit**

```bash
git add plugins/llm/src/llm/statuspage.py plugins/llm/tests/test_statuspage_parse.py \
        plugins/llm/tests/test_statuspage_payload.py
git commit -m "feat(status): accept incident.io's Atlassian-compatible shim

incident.io omits empty collections rather than sending []; absence now
parses as empty while a present-but-wrong-type value still rejects. An
unrecognised component status keeps the component instead of rejecting the
whole page — that rejection fired during an outage, the only time anyone
asks. Incident status stays strict: it drives TERMINAL_STATUSES and the
announce lifecycle."
```

---

### Task 2: Default, docs, and verification

**Files:**
- Modify: `plugins/llm/src/llm/config.py` (the `statusPageUrls` default)
- Modify: `plugins/llm/tests/test_config.py`
- Modify: `docs/guide/user/service-status.md`

- [ ] **Step 1: Write the failing test**

Update the existing default assertion in `plugins/llm/tests/test_config.py`:

```python
    def test_status_page_urls_default_is_three_urls(self) -> None:
        import llm.config  # noqa: F401 — import side effect registers the values
        import supybot.conf as conf

        assert list(conf.supybot.plugins.LLM.statusPageUrls()) == [
            "https://status.claude.com",
            "https://www.githubstatus.com",
            "https://status.openai.com",
        ]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest plugins/llm/tests/test_config.py -q`
Expected: FAIL — the default is still two URLs.

- [ ] **Step 3: Add OpenAI to the default**

In `plugins/llm/src/llm/config.py`, extend the `statusPageUrls` default list to include `"https://status.openai.com"`. Update the help text if it names the monitored services.

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest plugins/llm/tests/test_config.py -q`
Expected: PASS

Check whether `conftest.py`'s `make_registry_side_effect` default or any test asserts a source count that this change breaks; the shared fixture deliberately uses a single source, so it should be unaffected. Report what you checked.

- [ ] **Step 5: Update the user documentation**

In `docs/guide/user/service-status.md`:
- State that both Atlassian Statuspage and incident.io pages work, and that incident.io is read through its Atlassian-compatible endpoints.
- Update the default source list to name all three services.
- Do not promise anything about incident.io's live-incident vocabulary — it has not been observed. If the page mentions what happens on an unfamiliar component status, say the component is still reported.

Repo doc style: `@` command prefix, en-CA spelling, sentence-case headings, "authenticated" never NickServ terminology.

- [ ] **Step 6: Verify**

Run: `uv run pytest plugins/llm/tests/ -q` — expected PASS, at least 3083.
Run: `uv run mkdocs build --strict` — expected no warnings.

- [ ] **Step 7: Commit**

```bash
git add plugins/llm/src/llm/config.py plugins/llm/tests/test_config.py \
        docs/guide/user/service-status.md
git commit -m "feat(status): monitor status.openai.com by default

Adds OpenAI to statusPageUrls, three of the five-source cap, and documents
that incident.io pages are read through their Atlassian-compatible endpoints."
```

---

## Deployment

Auto-deploy: CI green → Docker build → the 15-minute updater timer restarts the service.

No prod config change. `statusPageUrls` is at its registered default, so OpenAI is picked up on deploy. Any OpenAI incident already open at deploy is seeded as announced and stays silent until it resolves — expected cold-start behaviour, same as GitHub on the previous deploy.

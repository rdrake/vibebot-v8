# Verse Storybook Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a model-invoked verse tool that writes a short story in the bot's verse voice, illustrates its favourite moment(s) with up to 3 AI images, and bundles prose + images into one storybook-themed HTML page posted in channel.

**Architecture:** Tool-only trigger plugged into the existing verse tool/handler seams. The handler enforces account + `llm.draw` capability + per-account rate limit + per-completion cap + daily image ceiling, then **fire-and-returns**: it submits the job to the LLM executor and replies with an in-character ack; a worker generates the story (dedicated prompt, robust JSON parse), draws images **sequentially** (no prompt laundering), assembles the page, and posts the URL. Images render because `<img>` is allowlisted in the shared sanitizer **and** restricted to the bot's own host by a deterministic post-pass.

**Tech Stack:** Python 3.14, Limnoria plugin, litellm, nh3 (HTML sanitizer), markdown, pytest (+ pytest-mock). Run tests with `uv run python -m pytest`. `make lint typecheck` runs after every edit via hook.

**Spec:** `docs/superpowers/specs/2026-06-16-verse-storybook-tool-design.md` (v2, post red-team).

---

## File Structure

- `plugins/llm/src/llm/config.py` — 7 new registry keys + rate-limit block registration.
- `plugins/llm/src/llm/service.py` — `ImageResult.url`; `_sanitize_html` img allowlist; `_restrict_img_srcs` post-pass; storybook CSS; JSON-extract + marker-embed helpers; `STORYBOOK_SYSTEM_PROMPT`; `generate_storybook()`; DNS re-check in image download.
- `plugins/llm/src/llm/verse/avatar.py` — `verse_storybook` tool spec (config-gated).
- `plugins/llm/src/llm/assistant.py` — per-completion invocation counter for `verse_storybook`.
- `plugins/llm/src/llm/plugin.py` — handler in `_build_verse_handlers_for_route` (gates + fire-and-return + post URL).
- Tests: `plugins/llm/tests/test_storybook.py` (new, the core), plus additions to `test_html_output.py` (sanitizer) and `test_commands.py` (handler gates).

Implement in order — later tasks depend on earlier types/helpers.

---

## Task 1: Config keys + rate-limit block

**Files:**
- Modify: `plugins/llm/src/llm/config.py` (mirror existing `registerChannelValue` / draw keys ~line 222-254, and `_register_rate_limit_block` ~line 776)
- Test: `plugins/llm/tests/test_storybook.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# plugins/llm/tests/test_storybook.py
"""Tests for the verse storybook tool."""
from __future__ import annotations


def test_storybook_config_defaults(make_service):
    service, plugin = make_service()
    # registryValue is mocked in make_service; assert the keys are registered
    # by reading them off the real registry group instead:
    import llm.config as cfg
    group = cfg.LLM
    assert group.verseStorybookEnabled is not None
    assert int(group.verseStorybookMaxImages()) == 3
    assert int(group.verseStorybookMaxPerTurn()) == 1
    assert int(group.verseStorybookCooldownSeconds()) == 300
    assert int(group.verseStorybookDailyImageCap()) == 30
    assert int(group.verseStorybookMaxChars()) == 6000
    assert int(group.verseStorybookImageTimeout()) == 45
```

- [ ] **Step 2: Run test, verify it fails**

Run: `uv run python -m pytest plugins/llm/tests/test_storybook.py::test_storybook_config_defaults -v`
Expected: FAIL (`AttributeError: verseStorybookEnabled`).

- [ ] **Step 3: Add the registry keys**

In `config.py`, next to the existing verse keys, register (follow the exact `registerChannelValue` + `registry.Boolean/NonNegativeInteger` idiom already used in the file):

```python
conf.registerChannelValue(LLM, "verseStorybookEnabled",
    registry.Boolean(False, _("""Expose the verse_storybook tool in this channel.""")))
conf.registerChannelValue(LLM, "verseStorybookMaxImages",
    registry.NonNegativeInteger(3, _("""Max illustrations drawn per story.""")))
conf.registerChannelValue(LLM, "verseStorybookMaxPerTurn",
    registry.NonNegativeInteger(1, _("""Max verse_storybook calls honored per completion.""")))
conf.registerChannelValue(LLM, "verseStorybookCooldownSeconds",
    registry.NonNegativeInteger(300, _("""Per-account cooldown between stories.""")))
conf.registerChannelValue(LLM, "verseStorybookDailyImageCap",
    registry.NonNegativeInteger(30, _("""Per-account daily ceiling on storybook images.""")))
conf.registerChannelValue(LLM, "verseStorybookMaxChars",
    registry.NonNegativeInteger(6000, _("""Story length cap before image embedding.""")))
conf.registerChannelValue(LLM, "verseStorybookImageTimeout",
    registry.NonNegativeInteger(45, _("""Per-image timeout (seconds) for storybook draws.""")))
```

If the file uses a different registration helper (e.g. a local `_reg`), match that. Read 200-260 first.

- [ ] **Step 4: Run test, verify it passes**

Run: `uv run python -m pytest plugins/llm/tests/test_storybook.py::test_storybook_config_defaults -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/config.py plugins/llm/tests/test_storybook.py
git commit -m "feat(config): verse storybook registry keys"
```

---

## Task 2: `ImageResult.url`

**Files:**
- Modify: `plugins/llm/src/llm/service.py:340-349` (`ImageResult`), and the success return in `_attempt_image_generation` (~`service.py:3223`) / `image_generation` (~3886) where the saved URL is known.
- Test: `plugins/llm/tests/test_storybook.py`

- [ ] **Step 1: Write the failing test**

```python
def test_image_result_has_url_field():
    from llm.service import ImageResult
    r = ImageResult(content="msg", url="https://h/llm/img_a.jpg")
    assert r.url == "https://h/llm/img_a.jpg"
    assert ImageResult(content="x").url is None  # default
```

- [ ] **Step 2: Run, verify fail**

Run: `uv run python -m pytest plugins/llm/tests/test_storybook.py::test_image_result_has_url_field -v`
Expected: FAIL (`TypeError: unexpected keyword 'url'`).

- [ ] **Step 3: Add the field + populate it**

Add to `ImageResult` (after `rewritten_prompt`):
```python
    url: str | None = None
```
Then find where `image_generation` builds its **success** `ImageResult` (the path that has just called `save_image_to_http` / `_download_and_save_image`, ~`service.py:3284-3307`). That helper returns the saved URL string; capture it into a local (e.g. `saved_url`) and pass `url=saved_url` into the returned `ImageResult`. Do NOT parse the URL out of `content`.

- [ ] **Step 4: Run targeted + the existing image tests**

Run: `uv run python -m pytest plugins/llm/tests/test_storybook.py::test_image_result_has_url_field plugins/llm/tests/ -k "image or draw" -q`
Expected: PASS (new test passes; existing draw/image tests unaffected — `url` defaults to None).

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_storybook.py
git commit -m "feat(service): expose saved URL on ImageResult"
```

---

## Task 3: Allowlist `<img>` in the sanitizer + host-restrict post-pass

The shared `_sanitize_html` must allow `<img>` (or images never render), but it is used by **every** pastebin page, so we add a deterministic post-pass that drops any `<img>` whose `src` is not same-host/relative.

**Files:**
- Modify: `plugins/llm/src/llm/service.py:698-744` (`_sanitize_html`); add `_restrict_img_srcs`; call it from `_save_markdown_to_http` right after `rendered = self._sanitize_html(rendered)` (~`service.py:4190`).
- Test: `plugins/llm/tests/test_html_output.py`

- [ ] **Step 1: Write failing tests**

```python
# in test_html_output.py, TestXssPrevention (uses save_code_to_http) + a new storybook case
def test_relative_img_src_survives(self, service, tmp_path):
    md = "Hello\n\n![a cat](img_abc.jpg)\n"
    url = service.save_markdown_to_http(md, title="t")
    content = (tmp_path / url.split("/")[-1]).read_text()
    assert '<img' in content and 'src="img_abc.jpg"' in content
    assert 'alt="a cat"' in content

def test_external_img_src_dropped(self, service, tmp_path):
    md = "![x](https://evil.example/track.png)"
    url = service.save_code_to_http(md)
    content = (tmp_path / url.split("/")[-1]).read_text()
    assert "evil.example" not in content  # external img removed

def test_javascript_img_src_dropped(self, service, tmp_path):
    md = '<img src="javascript:alert(1)" alt="x">'
    url = service.save_code_to_http(md)
    content = (tmp_path / url.split("/")[-1]).read_text()
    assert "javascript:" not in content

def test_onerror_img_attr_dropped(self, service, tmp_path):
    md = '<img src="img_a.jpg" onerror="alert(1)">'
    url = service.save_code_to_http(md)
    content = (tmp_path / url.split("/")[-1]).read_text()
    assert "onerror" not in content
```

(The `service` fixture in `TestXssPrevention` already configures `httpRoot=tmp_path`, `httpUrlBase="http://localhost/llm"`.)

- [ ] **Step 2: Run, verify fail**

Run: `uv run python -m pytest plugins/llm/tests/test_html_output.py -k "img_src or img_attr or img" -v`
Expected: FAIL (img stripped → assertions fail).

- [ ] **Step 3: Implement**

In `_sanitize_html`, add `"img"` to the `tags` set and an entry to `attributes`:
```python
            "img": {"src", "alt", "title"},
```
Keep `url_schemes={"http", "https", "mailto"}` (nh3 already drops `javascript:`/`data:` and unknown-attribute `onerror`).

Add the host-restrict post-pass and call it:
```python
    def _restrict_img_srcs(self, html: str, url_base: str) -> str:
        """Drop <img> whose src is neither a bare relative path nor under url_base.

        Storybook embeds its illustrations by relative filename, so legitimate
        images survive; externally-hosted images (tracking pixels, SSRF-on-view)
        are removed from every pastebin page.
        """
        import re as _re

        def _ok(src: str) -> bool:
            s = src.strip()
            if "://" not in s and not s.startswith("//"):
                return "/" not in s.split("?", 1)[0].rstrip("/") or s.startswith(url_base)
            return s.startswith(url_base)

        def _sub(m: "_re.Match[str]") -> str:
            tag = m.group(0)
            src_m = _re.search(r'src\s*=\s*"([^"]*)"', tag)
            return tag if (src_m and _ok(src_m.group(1))) else ""

        return _re.sub(r"<img\b[^>]*>", _sub, html)
```
In `_save_markdown_to_http`, immediately after the existing `rendered = self._sanitize_html(rendered)` line, add:
```python
        rendered = self._restrict_img_srcs(rendered, url_base)
```
(`url_base` is already in scope from `get_http_paths()`.)

> Note: storybook illustrations are embedded as **relative filenames** (`img_xxx.jpg`), which live in the same `http_root` as the page — so `_ok` accepts them and they resolve same-origin regardless of `httpUrlBase`.

- [ ] **Step 4: Run, verify pass**

Run: `uv run python -m pytest plugins/llm/tests/test_html_output.py -q`
Expected: PASS (all, including the 8 pre-existing XSS tests).

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_html_output.py
git commit -m "feat(service): allow same-host <img> in pastebin pages, drop external/unsafe"
```

---

## Task 4: Storybook image CSS

**Files:**
- Modify: storybook `<style>` block in `_save_markdown_to_http` (the `img` rule is currently absent).
- Test: covered by Task 3's `test_relative_img_src_survives` (page renders); add a CSS-presence assertion.

- [ ] **Step 1: Write failing test**

```python
def test_storybook_has_img_css(self, service, tmp_path):
    url = service.save_markdown_to_http("![c](img_a.jpg)", title="t")
    content = (tmp_path / url.split("/")[-1]).read_text()
    assert "img {" in content or "img{" in content
```

- [ ] **Step 2: Run, verify fail.** `uv run python -m pytest plugins/llm/tests/test_html_output.py::TestXssPrevention::test_storybook_has_img_css -v` → FAIL.

- [ ] **Step 3: Add CSS** to the storybook `<style>` (note the f-string requires `{{`/`}}`):

```
img {{ display: block; max-width: 100%; height: auto; margin: 1.6em auto; border: 1px solid rgba(154,123,63,0.55); border-radius: 4px; box-shadow: 0 6px 22px rgba(0,0,0,0.30); }}
img + em {{ display: block; text-align: center; color: var(--ink-soft); font-size: 0.95rem; margin-top: -0.8em; }}
```

- [ ] **Step 4: Run, verify pass.** Same command → PASS.

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_html_output.py
git commit -m "feat(service): storybook illustration CSS"
```

---

## Task 5: Robust JSON extraction helper

**Files:**
- Modify: `plugins/llm/src/llm/service.py` (add module-level `_extract_json_object`)
- Test: `plugins/llm/tests/test_storybook.py`

- [ ] **Step 1: Write failing tests**

```python
def test_extract_json_object():
    from llm.service import _extract_json_object as ex
    assert ex('{"a": 1}') == {"a": 1}
    assert ex('Here ye go! ```json\n{"a": 2}\n``` enjoy') == {"a": 2}
    assert ex('prose {"a": {"b": 3}} more prose') == {"a": {"b": 3}}
    assert ex("not json at all") is None
    assert ex('') is None
```

- [ ] **Step 2: Run, verify fail.** `... ::test_extract_json_object -v` → FAIL.

- [ ] **Step 3: Implement**

```python
def _extract_json_object(text: str | None) -> dict | None:
    """Best-effort extract a single JSON object from possibly-prose model output.

    Finds the first '{' and the matching closing '}' (brace depth), tolerating
    leading/trailing in-character prose and ```json fences. Returns None if no
    balanced object parses.
    """
    import json as _json

    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = _json.loads(text[start : i + 1])
                except ValueError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None
```

- [ ] **Step 4: Run, verify pass.** Same command → PASS.

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_storybook.py
git commit -m "feat(service): brace-matching JSON object extractor"
```

---

## Task 6: Marker embedding helper

**Files:**
- Modify: `plugins/llm/src/llm/service.py` (add static `_embed_illustrations`)
- Test: `plugins/llm/tests/test_storybook.py`

- [ ] **Step 1: Write failing tests** (covers `1` vs `11`, duplicates, orphans, user-injected tokens, caption echo)

```python
def test_embed_illustrations_basic():
    from llm.service import LLMService as S
    md = "Intro [[illustration:1]] middle [[illustration:11]] end"
    illos = {1: ("cat", "u1.jpg"), 11: ("dog", "u11.jpg")}
    out, used = S._embed_illustrations(md, illos)
    assert "![cat](u1.jpg)" in out and "*cat*" in out
    assert "![dog](u11.jpg)" in out
    assert used == {1, 11}

def test_embed_duplicate_marker_first_wins():
    from llm.service import LLMService as S
    md = "[[illustration:2]] x [[illustration:2]]"
    out, used = S._embed_illustrations(md, {2: ("c", "u.jpg")})
    assert out.count("![c](u.jpg)") == 1
    assert "[[illustration:2]]" not in out  # later dup stripped
    assert used == {2}

def test_embed_orphan_marker_removed():
    from llm.service import LLMService as S
    out, used = S._embed_illustrations("a [[illustration:9]] b", {1: ("c", "u")})
    assert "[[illustration:9]]" not in out and used == set()

def test_embed_strips_user_injected_image_and_markers_first():
    from llm.service import LLMService as S
    # user-echoed raw image + a fake marker the model parroted into prose
    md = "![evil](http://evil/p.png) tale [[illustration:1]]"
    out, used = S._embed_illustrations(
        S._strip_untrusted_markup(md), {1: ("c", "u.jpg")}
    )
    assert "evil" not in out
    assert "![c](u.jpg)" in out
```

- [ ] **Step 2: Run, verify fail.** `... -k embed -v` → FAIL.

- [ ] **Step 3: Implement** two static helpers:

```python
    @staticmethod
    def _strip_untrusted_markup(story_markdown: str) -> str:
        """Remove model-echoed raw image syntax and illustration markers so only
        server-placed markers (re-inserted by the structured field) are honored."""
        s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", story_markdown)  # raw images
        return s  # markers are re-added by the caller from validated fields

    @staticmethod
    def _embed_illustrations(
        story_markdown: str, illos: dict[int, tuple[str, str]]
    ) -> tuple[str, set[int]]:
        """Replace [[illustration:N]] with ![caption](url) + emphasised caption.

        Single regex pass (no 1-vs-11 collision). First marker per id wins; later
        duplicates and orphan markers are removed. Returns (html_markdown, used_ids).
        """
        used: set[int] = set()

        def repl(m: "re.Match[str]") -> str:
            n = int(m.group(1))
            if n in illos and n not in used:
                used.add(n)
                caption, url = illos[n]
                return f"![{caption}]({url})\n\n*{caption}*"
            return ""  # orphan or already-used → strip

        out = re.sub(r"\[\[illustration:(\d+)\]\]", repl, story_markdown)
        return out, used
```

> Note: because `_strip_untrusted_markup` removes raw markers too in the real flow, the **caller** (Task 8) re-inserts validated `[[illustration:N]]` markers from the model's structured `story_markdown` field only after confirming the model authored them — see Task 8 for the exact ordering. The duplicate/orphan tests above call `_embed_illustrations` directly.

- [ ] **Step 4: Run, verify pass.** `... -k embed -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_storybook.py
git commit -m "feat(service): robust illustration marker embedding"
```

---

## Task 7: Storybook system prompt + story-generation call

A **dedicated** prompt (NOT the verse plain-text overlay, which forbids markdown). Inherits persona/voice via the channel's verse persona string but re-enables markdown + structured output. Includes retries and denial-detection on extracted prose only.

**Files:**
- Modify: `plugins/llm/src/llm/service.py` — add `STORYBOOK_SYSTEM_PROMPT`; add `_generate_story_struct(brief, *, channel, persona, conversation)` returning a validated dict or None.
- Test: `plugins/llm/tests/test_storybook.py` (mock the underlying completion).

- [ ] **Step 1: Write failing test**

```python
def test_generate_story_struct_parses_and_validates(make_service, mocker):
    service, plugin = make_service()
    payload = (
        'Here is your tale! ```json\n'
        '{"title":"The Tin Fox","story_markdown":"Once [[illustration:1]] fin.",'
        '"illustrations":[{"id":1,"caption":"a fox","image_prompt":"a tin fox"}]}\n```'
    )
    mocker.patch.object(service, "_ask_completion", return_value=payload)
    out = service._generate_story_struct("spin a tale", channel="#c", persona="voice", conversation=[])
    assert out["title"] == "The Tin Fox"
    assert out["illustrations"][0]["id"] == 1
    assert "[[illustration:1]]" in out["story_markdown"]

def test_generate_story_struct_retries_then_fails(make_service, mocker):
    service, plugin = make_service()
    m = mocker.patch.object(service, "_ask_completion", return_value="no json here")
    out = service._generate_story_struct("x", channel="#c", persona="v", conversation=[])
    assert out is None
    assert m.call_count >= 3  # initial + >=2 retries
```

- [ ] **Step 2: Run, verify fail.** `... -k generate_story_struct -v` → FAIL.

- [ ] **Step 3: Implement**

```python
STORYBOOK_SYSTEM_PROMPT = (
    "You are telling an illustrated short story IN CHARACTER, in the persona "
    "described below. Write vivid prose. Then choose the moment(s) most worth "
    "illustrating.\n\n"
    "Respond with ONLY a single JSON object, no prose outside it, no code fence:\n"
    '{"title": str, "story_markdown": str, '
    '"illustrations": [{"id": int, "caption": str, "image_prompt": str}]}\n'
    "Rules: story_markdown is Markdown and may contain [[illustration:N]] markers "
    "where an illustration belongs (0 to {max_images} of them, matching the ids in "
    "illustrations). image_prompt is a concrete visual scene description. Keep the "
    "story under {max_chars} characters.\n\nPERSONA:\n{persona}"
)
```

```python
    def _generate_story_struct(self, brief, *, channel, persona, conversation):
        max_images = int(self.plugin.registryValue("verseStorybookMaxImages", channel) or 3)
        max_chars = int(self.plugin.registryValue("verseStorybookMaxChars", channel) or 6000)
        system = STORYBOOK_SYSTEM_PROMPT.format(
            max_images=max_images, max_chars=max_chars, persona=persona or ""
        )
        user = brief or "Tell an illustrated story drawn from the recent scene."
        nudge = ""
        for _attempt in range(3):  # initial + 2 retries
            raw = self._ask_completion(system + nudge, user, channel)
            obj = _extract_json_object(raw)
            valid = self._validate_story_obj(obj)
            if valid is not None:
                return valid
            nudge = "\n\nIMPORTANT: emit ONLY the JSON object — no prose, no fence."
        return None

    @staticmethod
    def _validate_story_obj(obj):
        """Partial-tolerant validation. Requires title + story_markdown; drops
        malformed illustration entries rather than failing the whole story."""
        if not isinstance(obj, dict):
            return None
        title = obj.get("title")
        story = obj.get("story_markdown")
        if not isinstance(title, str) or not isinstance(story, str) or not story.strip():
            return None
        illos = []
        for it in obj.get("illustrations") or []:
            if (isinstance(it, dict) and isinstance(it.get("id"), int)
                    and isinstance(it.get("caption"), str)
                    and isinstance(it.get("image_prompt"), str)
                    and it["image_prompt"].strip()):
                illos.append({"id": it["id"], "caption": it["caption"],
                              "image_prompt": it["image_prompt"]})
        return {"title": title.strip(), "story_markdown": story, "illustrations": illos}
```

> Denial detection: if your `_ask_completion` path does not already run the verse denial guard, run `self._is_verse_denial(valid["story_markdown"])` on the **extracted prose** (not raw JSON) and treat a positive as a retry. Do NOT run denial-stripping on this call. Confirm `_ask_completion`'s signature near `service.py:3102` and adapt the call if it takes different args.

- [ ] **Step 4: Run, verify pass.** `... -k generate_story_struct -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_storybook.py
git commit -m "feat(service): storybook story generation with robust parse"
```

---

## Task 8: `generate_storybook()` orchestration

Ties Tasks 2–7 together: story → cap → sequential draws (no laundering) → embed → save. Returns a small result struct.

**Files:**
- Modify: `plugins/llm/src/llm/service.py` — add `StorybookResult` NamedTuple + `generate_storybook(...)`.
- Test: `plugins/llm/tests/test_storybook.py`

- [ ] **Step 1: Write failing tests**

```python
def test_generate_storybook_embeds_and_saves(make_service, mocker, tmp_path):
    service, plugin = make_service(httpRoot=str(tmp_path), httpUrlBase="http://h/llm")
    mocker.patch.object(service, "_generate_story_struct", return_value={
        "title": "The Tin Fox",
        "story_markdown": "Once [[illustration:1]] upon a time.",
        "illustrations": [{"id": 1, "caption": "a fox", "image_prompt": "a tin fox"}],
    })
    from llm.service import ImageResult
    mocker.patch.object(service, "image_generation",
        return_value=ImageResult(content="ok", url="img_fox.jpg"))
    res = service.generate_storybook("brief", channel="#c", persona="v", conversation=[])
    assert res.url and res.title == "The Tin Fox" and res.image_count == 1
    page = (tmp_path / res.url.split("/")[-1]).read_text()
    assert '<img' in page and 'src="img_fox.jpg"' in page

def test_generate_storybook_image_failure_drops_marker(make_service, mocker, tmp_path):
    service, plugin = make_service(httpRoot=str(tmp_path), httpUrlBase="http://h/llm")
    mocker.patch.object(service, "_generate_story_struct", return_value={
        "title": "T", "story_markdown": "a [[illustration:1]] b",
        "illustrations": [{"id": 1, "caption": "c", "image_prompt": "p"}]})
    from llm.service import ImageResult
    mocker.patch.object(service, "image_generation",
        return_value=ImageResult(content="blocked", url=None, error="safety"))
    res = service.generate_storybook("b", channel="#c", persona="v", conversation=[])
    assert res.url and res.image_count == 0  # marker dropped, prose-only page saved
    page = (tmp_path / res.url.split("/")[-1]).read_text()
    assert "[[illustration:1]]" not in page

def test_generate_storybook_caps_images(make_service, mocker, tmp_path):
    service, plugin = make_service(httpRoot=str(tmp_path), httpUrlBase="http://h/llm")
    illos = [{"id": i, "caption": f"c{i}", "image_prompt": f"p{i}"} for i in range(1, 6)]
    markers = " ".join(f"[[illustration:{i}]]" for i in range(1, 6))
    mocker.patch.object(service, "_generate_story_struct", return_value={
        "title": "T", "story_markdown": markers, "illustrations": illos})
    from llm.service import ImageResult
    gen = mocker.patch.object(service, "image_generation",
        return_value=ImageResult(content="ok", url="i.jpg"))
    service.generate_storybook("b", channel="#c", persona="v", conversation=[])
    assert gen.call_count == 3  # verseStorybookMaxImages default

def test_generate_storybook_none_when_story_fails(make_service, mocker):
    service, plugin = make_service()
    mocker.patch.object(service, "_generate_story_struct", return_value=None)
    assert service.generate_storybook("b", channel="#c", persona="v", conversation=[]) is None
```

- [ ] **Step 2: Run, verify fail.** `... -k generate_storybook -v` → FAIL.

- [ ] **Step 3: Implement**

```python
class StorybookResult(NamedTuple):
    url: str
    title: str
    image_count: int
    dropped: int


    def generate_storybook(self, brief, *, channel, persona, conversation):
        story = self._generate_story_struct(
            brief, channel=channel, persona=persona, conversation=conversation)
        if story is None:
            return None
        max_images = int(self.plugin.registryValue("verseStorybookMaxImages", channel) or 3)
        max_chars = int(self.plugin.registryValue("verseStorybookMaxChars", channel) or 6000)
        timeout = int(self.plugin.registryValue("verseStorybookImageTimeout", channel) or 45)

        wanted = story["illustrations"][:max_images]
        dropped_cap = len(story["illustrations"]) - len(wanted)
        if dropped_cap:
            self.log.info("storybook: dropped %d illustrations over cap", dropped_cap)

        # Draw sequentially; NO prompt laundering (drawAutoRewriteMax=0).
        drawn: dict[int, tuple[str, str]] = {}
        for it in wanted:
            res = self.image_generation(
                f"storybook illustration, painted fairytale style: {it['image_prompt']}",
                channel=channel, auto_rewrite_max=0, timeout=timeout)
            if res and res.url and not res.error:
                drawn[it["id"]] = (it["caption"], res.url)

        body = self._strip_untrusted_markup(story["story_markdown"])[:max_chars]
        embedded, used = self._embed_illustrations(body, drawn)
        markdown = f"# {story['title'] or 'An Untitled Tale'}\n\n{embedded}\n"
        url = self.save_markdown_to_http(markdown, title=story["title"] or "An Untitled Tale")
        if not url:
            return None
        return StorybookResult(url=url, title=story["title"] or "An Untitled Tale",
                               image_count=len(used), dropped=dropped_cap)
```

> Two integration details to confirm against the real `image_generation` signature (~`service.py:3886`): (a) it must accept an `auto_rewrite_max`/override and a `timeout` override — if it reads `drawAutoRewriteMax`/`drawTimeout` from registry instead, add explicit override parameters in this task and thread them through. (b) `_strip_untrusted_markup` removes raw `![](...)` but NOT `[[illustration:N]]` (the model authored those in `story_markdown`; they are server-trusted because we asked for them and they only map to validated ids). User chat cannot inject a marker that maps to a real `drawn` id without the model also producing that id in `illustrations`, which we control.

- [ ] **Step 4: Run, verify pass.** `... -k generate_storybook -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_storybook.py
git commit -m "feat(service): generate_storybook orchestration"
```

---

## Task 9: SSRF DNS re-check on image download

**Files:**
- Modify: `plugins/llm/src/llm/service.py` — in `_download_and_save_image` (~4409) / `validate_external_url` (~470), resolve the hostname and re-check the resolved IP against the private/reserved set before fetch.
- Test: `plugins/llm/tests/test_storybook.py`

- [ ] **Step 1: Write failing test**

```python
def test_download_rejects_private_resolved_ip(make_service, mocker):
    service, plugin = make_service()
    mocker.patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("127.0.0.1", 443))])
    assert service._resolves_to_public("http://rebind.example/x.png") is False
    mocker.patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.216.34", 443))])
    assert service._resolves_to_public("http://example.com/x.png") is True
```

- [ ] **Step 2: Run, verify fail.** → FAIL.

- [ ] **Step 3: Implement** a `_resolves_to_public(url) -> bool` (use `socket.getaddrinfo` + `ipaddress.ip_address(...).is_global`) and call it inside the download path before fetching; if it returns False, treat as a failed image (drop). Reuse the existing private/reserved logic in `validate_external_url` if present.

```python
    def _resolves_to_public(self, url: str) -> bool:
        import ipaddress, socket
        from urllib.parse import urlparse
        host = urlparse(url).hostname
        if not host:
            return False
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError:
            return False
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if not ip.is_global:
                return False
        return True
```

- [ ] **Step 4: Run, verify pass.** → PASS.

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_storybook.py
git commit -m "fix(service): re-check resolved IP before image download (DNS rebinding)"
```

---

## Task 10: `verse_storybook` tool spec (config-gated)

**Files:**
- Modify: `plugins/llm/src/llm/verse/avatar.py:34` (`make_verse_tool_specs`)
- Test: `plugins/llm/tests/test_storybook.py`

- [ ] **Step 1: Write failing test**

```python
def test_make_verse_tool_specs_includes_storybook_when_enabled():
    from llm.verse.avatar import make_verse_tool_specs
    specs = make_verse_tool_specs(max_actors=2, storybook=True)
    names = [s["function"]["name"] for s in specs]
    assert "verse_storybook" in names
    specs_off = make_verse_tool_specs(max_actors=2, storybook=False)
    assert "verse_storybook" not in [s["function"]["name"] for s in specs_off]
```

- [ ] **Step 2: Run, verify fail.** → FAIL (unexpected `storybook` kwarg).

- [ ] **Step 3: Implement.** Add a `storybook: bool = False` param to `make_verse_tool_specs`; when true, append a tool spec mirroring the existing spec dict shape in that function:

```python
    if storybook:
        specs.append({
            "type": "function",
            "function": {
                "name": "verse_storybook",
                "description": (
                    "Turn the current scene into a short illustrated story page. "
                    "Call ONLY when a scenario truly deserves a full tale. Returns a "
                    "link you should share in character."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "brief": {"type": "string",
                                  "description": "What the story should be about."},
                    },
                    "required": ["brief"],
                },
            },
        })
```

- [ ] **Step 4: Run, verify pass.** → PASS.

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/verse/avatar.py plugins/llm/tests/test_storybook.py
git commit -m "feat(verse): verse_storybook tool spec"
```

---

## Task 11: Per-completion invocation cap

**Files:**
- Modify: `plugins/llm/src/llm/assistant.py` (the `AssistantToolExecutor` that dispatches tools, ~line 988 where image tools dispatch) — count `verse_storybook` invocations per completion and refuse beyond `verseStorybookMaxPerTurn`.
- Test: `plugins/llm/tests/test_storybook.py`

- [ ] **Step 1: Write failing test** (adapt to the executor's actual constructor/dispatch — read ~960-1010 first):

```python
def test_storybook_capped_per_completion(make_storybook_executor):
    ex = make_storybook_executor(max_per_turn=1)
    first = ex.dispatch("verse_storybook", {"brief": "a"})
    second = ex.dispatch("verse_storybook", {"brief": "b"})
    assert "error" not in first.lower()
    assert "error" in second.lower() or "already" in second.lower()
```

- [ ] **Step 2: Run, verify fail.** → FAIL.

- [ ] **Step 3: Implement.** Add an integer counter on the executor instance (reset per completion construction), increment when `verse_storybook` is dispatched, and return a tool-error result before invoking the handler when the count exceeds the configured cap. Wire the cap from `verseStorybookMaxPerTurn`.

- [ ] **Step 4: Run, verify pass.** → PASS.

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/assistant.py plugins/llm/tests/test_storybook.py
git commit -m "feat(assistant): cap verse_storybook invocations per completion"
```

---

## Task 12: Plugin handler — gates + fire-and-return + post URL

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` — register the handler in `_build_verse_handlers_for_route` (~3572); pass `storybook=verseStorybookEnabled` into `make_verse_tool_specs` at the verse spec build site (~3426).
- Test: `plugins/llm/tests/test_commands.py` (new `TestVerseStorybook` class)

- [ ] **Step 1: Write failing tests** (mirror `TestSendLongReply`/draw test fixtures; mock `generate_storybook` and the executor `submit`):

```python
class TestVerseStorybook:
    def test_refuses_without_account(self, plugin_env, mocker):
        plugin, irc, msg = plugin_env
        mocker.patch.object(plugin, "_require_account", return_value=None)
        handler = plugin._storybook_handler(irc, msg, "#c")
        out = handler({"brief": "x"})
        assert "error" in out.lower() or "account" in out.lower()
        plugin.llm_service.generate_storybook.assert_not_called()

    def test_refuses_without_draw_capability(self, plugin_env, mocker):
        plugin, irc, msg = plugin_env
        mocker.patch.object(plugin, "_require_account", return_value="acct")
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=False)
        out = plugin._storybook_handler(irc, msg, "#c")({"brief": "x"})
        assert "error" in out.lower() or "cannot" in out.lower()
        plugin.llm_service.generate_storybook.assert_not_called()

    def test_fires_and_returns_ack(self, plugin_env, mocker):
        plugin, irc, msg = plugin_env
        mocker.patch.object(plugin, "_require_account", return_value="acct")
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        mocker.patch.object(plugin, "_storybook_rate_ok", return_value=True)
        submit = mocker.patch.object(plugin._llm_executor, "submit")
        out = plugin._storybook_handler(irc, msg, "#c")({"brief": "a heist"})
        assert out and "error" not in out.lower()       # in-character ack returned
        submit.assert_called_once()                     # job dispatched, not run inline
        plugin.llm_service.generate_storybook.assert_not_called()  # runs in worker
```

- [ ] **Step 2: Run, verify fail.** `uv run python -m pytest plugins/llm/tests/test_commands.py::TestVerseStorybook -v` → FAIL.

- [ ] **Step 3: Implement.**

Build a handler factory `_storybook_handler(self, irc, msg, channel)` returning a `handler(args: dict) -> str` that:
1. `account = self._require_account(irc, msg)`; if None → return a refusal string.
2. `if not ircdb.checkCapability(msg.prefix, "llm.draw")` → return refusal.
3. `if not self._storybook_rate_ok(account, channel)` (atomic check-and-reserve via the existing `_rate_buckets`/`_rate_buckets_lock`, keyed per-account; also enforce `verseStorybookDailyImageCap` against `db.log_usage` here) → return cooldown refusal.
4. Resolve the persona/voice string the verse path already builds for this channel/route, and snapshot the conversation.
5. Define a `_job()` closure that calls `self.llm_service.generate_storybook(brief, channel=channel, persona=persona, conversation=convo)` and, on a non-None result, posts the URL via `self._safe_reply`/the channel's reply path with an in-character lead-in; on None, posts a brief in-character apology.
6. **Worker-context aware dispatch:** `try: self._llm_executor.submit("verse_storybook", _job) except RecursiveSubmitError: _job()` (already on a worker — run inline).
7. Return the in-character ack string (this becomes the tool result the verse model weaves into its reply).

Register `combined_handlers["verse_storybook"] = self._storybook_handler(irc, msg, channel)` in `_build_verse_handlers_for_route`, gated on `self.registryValue("verseStorybookEnabled", channel)`. At the verse spec build site (~3426), pass `storybook=self.registryValue("verseStorybookEnabled", channel)` to `make_verse_tool_specs`.

- [ ] **Step 4: Run, verify pass.** Same command → PASS.

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_commands.py
git commit -m "feat(verse): verse_storybook handler — gated, fire-and-return"
```

---

## Task 13: Full-suite green + manual smoke notes

- [ ] **Step 1: Run the whole suite + lint/typecheck**

Run: `uv run python -m pytest plugins/llm/tests/ -q && make lint typecheck`
Expected: all pass, lint+typecheck clean.

- [ ] **Step 2: Manual smoke checklist** (record in the PR/commit body; do not automate):
  - In a channel with `verseStorybookEnabled=True`, a verse user prompts a story-worthy scene; confirm an immediate in-character ack, then a link a minute or two later.
  - Open the page: storybook theme, `<img>` plates render, captions present, title in the tab.
  - Confirm a non-`llm.draw` user gets a refusal; confirm cooldown blocks a rapid second call.

- [ ] **Step 3: Commit any fixes, then deploy** (auto-deploy on Docker green per repo norms).

```bash
git commit -am "test(verse): storybook full-suite green" || true
git push
```

---

## Self-Review Notes (author)

- **Spec coverage:** gating (T1,T12), execution model fire-and-return (T12), URL plumbing (T2), sanitizer img + host-scope (T3), CSS (T4), robust JSON (T5,T7), marker safety (T6), dedicated prompt/no-overlay (T7), no-laundering draws + cap (T8), SSRF DNS (T9), tool spec (T10), per-turn cap (T11) — all mapped.
- **Integration risks flagged for the implementer** (verify against real signatures, don't assume): `image_generation` override params for `auto_rewrite_max`/`timeout` (T8); `_ask_completion` signature (T7); `AssistantToolExecutor` constructor/dispatch shape (T11); the exact persona string the verse route builds (T12); `nh3` relative-URL pass-through (T3 — the test will reveal it; if relative src is stripped, switch `_ok` to compare against absolute `url_base` and embed absolute URLs).
- **Type consistency:** `StorybookResult(url,title,image_count,dropped)`, `ImageResult.url`, `generate_storybook(brief,*,channel,persona,conversation)`, `_embed_illustrations(md, dict[int,(caption,url)]) -> (str,set)` used consistently across tasks.

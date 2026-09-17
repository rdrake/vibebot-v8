"""Meme templates: request parsing, caption encoding, and template resolution.

The bot never chooses a template. The user names one — "drake", "distracted
boyfriend" — the catalog resolves that name deterministically, and
memegen.link renders the captions onto it. Rendering our own would mean
per-template text-box coordinates and a licensed Impact; memegen already has
both for two hundred templates, so this module only builds URLs.

Two callers share it: the ``@meme`` command (no LLM at all) and the
``make_meme`` chat tool, where the model's job is to transcribe the name the
user said, not to pick one. Both go through :func:`plan_meme`, so a wrong
caption count or an unknown name reads the same in either place.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

# memegen caps a whole URL well above this, but a caption past this length
# renders as unreadable four-point text anyway.
MAX_CAPTION_CHARS = 200
MAX_LINES = 8

# memegen's escape table (README, "Special Characters"). Order matters: the
# characters memegen uses AS escapes (_ - ~) are doubled/escaped first so a
# later substitution cannot produce one by accident.
_ESCAPES: tuple[tuple[str, str], ...] = (
    ("_", "__"),
    ("-", "--"),
    ("\n", "~n"),
    ("?", "~q"),
    ("&", "~a"),
    ("%", "~p"),
    ("#", "~h"),
    ("/", "~s"),
    ("\\", "~b"),
    ("<", "~l"),
    (">", "~g"),
    ('"', "''"),
    (" ", "_"),
)

_NOISE_WORDS = frozenset({"the", "a", "an", "meme"})
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class MemeTemplate:
    id: str
    name: str
    lines: int
    keywords: tuple[str, ...] = ()
    example: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemePlan:
    """What :func:`plan_meme` decided: a URL to fetch, or why there is none."""

    url: str | None
    template: MemeTemplate | None
    error: str | None


def parse_meme_request(text: str) -> tuple[str, list[str]] | None:
    """Split ``template | line | line`` into its parts.

    Returns None when there is no template name; a template with no lines is
    a valid request (plan_meme answers it with the template's line count and
    example) so ``@meme drake`` teaches the format.
    """
    parts = [p.strip() for p in text.split("|")]
    if not parts or not parts[0]:
        return None
    return parts[0], parts[1:]


def encode_caption(text: str) -> str:
    """Turn one caption into a memegen path segment. Blank is ``_``."""
    text = text.strip()
    if not text:
        return "_"
    for char, escape in _ESCAPES:
        text = text.replace(char, escape)
    return quote(text, safe="~'")


def build_meme_url(base: str, template_id: str, lines: list[str]) -> str:
    segments = "/".join(encode_caption(line) for line in lines)
    return f"{base.rstrip('/')}/images/{template_id}/{segments}.png"


def parse_templates(entries: Any) -> list[MemeTemplate]:
    """Read memegen's ``/templates/`` JSON. Malformed entries are dropped."""
    templates: list[MemeTemplate] = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        template_id = entry.get("id")
        name = entry.get("name")
        lines = entry.get("lines")
        if not (isinstance(template_id, str) and isinstance(name, str) and isinstance(lines, int)):
            continue
        if not template_id or lines < 1:
            continue
        keywords = tuple(k for k in entry.get("keywords") or [] if isinstance(k, str))
        example_obj = entry.get("example") or {}
        example_text = example_obj.get("text") if isinstance(example_obj, dict) else None
        example = tuple(t for t in example_text or [] if isinstance(t, str))
        templates.append(MemeTemplate(template_id, name, lines, keywords, example))
    return templates


def _normalise(text: str) -> str:
    tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t not in _NOISE_WORDS]
    return " ".join(tokens)


class MemeCatalog:
    """The template list plus deterministic name resolution."""

    def __init__(self, templates: list[MemeTemplate]) -> None:
        self.templates = list(templates)
        self._by_id = {t.id.lower(): t for t in self.templates}

    def __len__(self) -> int:
        return len(self.templates)

    def resolve(self, query: str) -> MemeTemplate | None:
        """Exact id, then exact name, then a UNIQUE name substring, then keyword.

        Ambiguity is a miss on purpose: "cat" matching Grumpy Cat and
        Business Cat must not silently pick one. The caller lists both.
        """
        q = query.strip().lower()
        if not q:
            return None
        if q in self._by_id:
            return self._by_id[q]
        nq = _normalise(q)
        if not nq:
            return None
        if nq in self._by_id:
            return self._by_id[nq]
        by_name = [t for t in self.templates if _normalise(t.name) == nq]
        if len(by_name) == 1:
            return by_name[0]
        by_substring = [t for t in self.templates if nq in _normalise(t.name)]
        if len(by_substring) == 1:
            return by_substring[0]
        if len(by_substring) > 1:
            return None
        by_keyword = [t for t in self.templates if any(nq == _normalise(k) for k in t.keywords)]
        if len(by_keyword) == 1:
            return by_keyword[0]
        return None

    def suggest(self, query: str, limit: int = 5) -> list[MemeTemplate]:
        """Closest names first; the head of the list when nothing is close."""
        nq = _normalise(query)
        scored: list[tuple[int, int, str, MemeTemplate]] = []
        for t in self.templates:
            haystack = _normalise(f"{t.id} {t.name} {' '.join(t.keywords)}")
            pos = haystack.find(nq) if nq else -1
            if pos < 0 and nq and any(tok in haystack for tok in nq.split()):
                pos = 1000
            if pos >= 0:
                scored.append((pos, len(t.name), t.id, t))
        scored.sort()
        picks = [s[3] for s in scored[:limit]]
        if len(picks) < limit:
            seen = {t.id for t in picks}
            picks.extend(t for t in self.templates if t.id not in seen)
        return picks[:limit]


def _describe(template: MemeTemplate) -> str:
    example = " | ".join(template.example) if template.example else ""
    tail = f" — e.g. {template.id} | {example}" if example else ""
    plural = "caption" if template.lines == 1 else "captions"
    return f"{template.id} ({template.name}) takes {template.lines} {plural}{tail}"


def plan_meme(catalog: MemeCatalog, base: str, template_query: str, lines: list[str]) -> MemePlan:
    """Resolve the name, check the captions, build the URL.

    Fewer captions than boxes is fine (the rest render blank — most people
    only want two on a three-box template); more is not, since memegen would
    drop them silently.
    """
    template = catalog.resolve(template_query)
    if template is None:
        names = ", ".join(f"{t.id} ({t.name})" for t in catalog.suggest(template_query))
        return MemePlan(None, None, f"No meme template called '{template_query}'. Try: {names}")
    lines = [line.strip() for line in lines]
    if not any(lines):
        return MemePlan(None, template, f"Give the captions: {_describe(template)}")
    if len(lines) > template.lines:
        return MemePlan(
            None,
            template,
            f"{template.id} takes {template.lines} captions, you gave {len(lines)}. "
            f"Separate them with |.",
        )
    for line in lines:
        if len(line) > MAX_CAPTION_CHARS:
            return MemePlan(
                None,
                template,
                f"That caption is too long ({len(line)} chars; max {MAX_CAPTION_CHARS}).",
            )
    padded = lines + [""] * (template.lines - len(lines))
    return MemePlan(build_meme_url(base, template.id, padded), template, None)


def fetch_templates(
    base: str,
    *,
    timeout: float,
    opener: urllib.request.OpenerDirector | None = None,
) -> list[MemeTemplate]:
    """GET ``/templates/`` from the memegen base. Raises on any failure."""
    from .service import validate_external_url

    base = base.rstrip("/")
    if not validate_external_url(base):
        raise ValueError(f"memeApiBase is not a safe http(s) URL: {base!r}")
    request = urllib.request.Request(
        f"{base}/templates/", headers={"User-Agent": "vibebot-meme/1.0"}
    )
    opener = opener or urllib.request.build_opener()
    with opener.open(request, timeout=timeout) as response:
        body = response.read()
    return parse_templates(json.loads(body))


class CachedCatalog:
    """A catalog that refreshes itself from memegen at most once per ``ttl``.

    A failed refresh keeps the last good list and waits ``retry_after``
    before trying again — a dead memegen must not cost a GET per @meme. Only
    a first-ever failure leaves the catalog empty, which callers report as
    "templates unavailable".
    """

    def __init__(
        self, base: str, *, timeout: float, ttl: float = 86400.0, retry_after: float = 300.0
    ) -> None:
        self.base = base
        self.timeout = timeout
        self.ttl = ttl
        self.retry_after = retry_after
        self._catalog: MemeCatalog | None = None
        self._next_attempt = 0.0
        self._last_error: str | None = None

    def get(self, now: float | None = None) -> MemeCatalog | None:
        now = time.time() if now is None else now
        if now >= self._next_attempt:
            try:
                self._catalog = MemeCatalog(fetch_templates(self.base, timeout=self.timeout))
                self._last_error = None
                self._next_attempt = now + self.ttl
            except Exception as exc:  # noqa: BLE001 — keep the last good list
                self._last_error = str(exc)
                self._next_attempt = now + self.retry_after
        return self._catalog

    @property
    def last_error(self) -> str | None:
        return self._last_error


def matches(template: MemeTemplate, word: str) -> bool:
    """Does ``word`` appear in the template's id, name, or keywords?"""
    nw = _normalise(word)
    if not nw:
        return True
    haystack = _normalise(f"{template.id} {template.name} {' '.join(template.keywords)}")
    return nw in haystack

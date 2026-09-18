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
from dataclasses import dataclass, replace
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

_NOISE_WORDS = frozenset({"the", "a", "an", "meme", "template", "of", "and"})
_TOKEN_RE = re.compile(r"[a-z0-9]+")


# memegen's bring-your-own-image template: two boxes, top and bottom, drawn
# over whatever ``?background=`` points at. Used for pasted image URLs and
# for URL-valued aliases.
CUSTOM_TEMPLATE_ID = "custom"
CUSTOM_LINES = 2


@dataclass(frozen=True)
class MemeTemplate:
    id: str
    name: str
    lines: int
    keywords: tuple[str, ...] = ()
    example: tuple[str, ...] = ()
    # Set only on custom templates: the image memegen draws the captions on.
    background: str | None = None


def custom_template(background: str, name: str = "custom image") -> MemeTemplate:
    return MemeTemplate(CUSTOM_TEMPLATE_ID, name, CUSTOM_LINES, background=background)


def _is_http_url(text: str) -> bool:
    return text.lower().startswith(("http://", "https://")) and " " not in text


def _safe_background(url: str) -> bool:
    from .service import validate_external_url

    return _is_http_url(url) and validate_external_url(url)


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


def build_meme_url(
    base: str, template_id: str, lines: list[str], *, background: str | None = None
) -> str:
    segments = "/".join(encode_caption(line) for line in lines)
    url = f"{base.rstrip('/')}/images/{template_id}/{segments}.png"
    if background:
        url += "?background=" + quote(background, safe="")
    return url


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


def _haystack(template: MemeTemplate) -> str:
    return _normalise(f"{template.id} {template.name} {' '.join(template.keywords)}")


def _overlap(query_tokens: list[str], template: MemeTemplate) -> int:
    words = set(_haystack(template).split())
    return sum(1 for tok in query_tokens if tok in words)


class MemeCatalog:
    """The template list plus deterministic name resolution."""

    def __init__(self, templates: list[MemeTemplate]) -> None:
        self.templates = list(templates)
        self._by_id = {t.id.lower(): t for t in self.templates}

    def __len__(self) -> int:
        return len(self.templates)

    def with_aliases(self, aliases: dict[str, str]) -> MemeCatalog:
        """A copy with operator aliases folded in.

        ``name=id`` makes ``name`` a keyword on that template, so both
        resolve and ``@meme list`` see it. ``name=https://...`` adds a two-box
        custom template drawn on that image. Aliases pointing at an unknown
        id or an unsafe URL are dropped, not raised: a typo in bot.conf must
        not take every meme down with it.
        """
        extra_keywords: dict[str, list[str]] = {}
        customs: list[MemeTemplate] = []
        for name, target in aliases.items():
            if _is_http_url(target):
                if _safe_background(target):
                    customs.append(
                        MemeTemplate(
                            _normalise(name).replace(" ", "-") or name,
                            name,
                            CUSTOM_LINES,
                            background=target,
                        )
                    )
            elif target.lower() in self._by_id:
                extra_keywords.setdefault(target.lower(), []).append(name)
        merged = [
            replace(t, keywords=(*t.keywords, *extra_keywords[t.id.lower()]))
            if t.id.lower() in extra_keywords
            else t
            for t in self.templates
        ]
        return MemeCatalog(merged + customs)

    def resolve(self, query: str) -> MemeTemplate | None:
        """Exact id, exact name, UNIQUE name substring, keyword, then word overlap.

        Ambiguity is a miss on purpose at every tier: "cat" matching Grumpy
        Cat and Business Cat must not silently pick one, and "expanding
        brain" tying Galaxy Brain with Scumbag Brain must not either. The
        caller lists the candidates instead.
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
        if by_keyword:
            return None
        # memegen names templates by the quote; people name them by the
        # character. "willy wonka" has to reach Condescending Wonka on the
        # one word they share, but one word of five is a coincidence.
        tokens = nq.split()
        scored = sorted(((_overlap(tokens, t), t) for t in self.templates), key=lambda s: -s[0])
        best, winner = scored[0]
        if best == 0 or best * 2 < len(tokens):
            return None
        if len(scored) > 1 and scored[1][0] == best:
            return None
        return winner

    def suggest(self, query: str, limit: int = 5) -> list[MemeTemplate]:
        """Closest names first. Empty when nothing shares a word with the
        query — an alphabetical head would only look like advice."""
        nq = _normalise(query)
        if not nq:
            return self.templates[:limit]
        tokens = nq.split()
        scored: list[tuple[int, int, int, str, MemeTemplate]] = []
        for t in self.templates:
            haystack = _haystack(t)
            pos = haystack.find(nq)
            overlap = _overlap(tokens, t)
            if pos < 0 and overlap == 0:
                continue
            scored.append((0 if pos >= 0 else 1, -overlap, len(t.name), t.id, t))
        scored.sort()
        return [s[4] for s in scored[:limit]]


def parse_aliases(entries: list[str]) -> dict[str, str]:
    """``name=id`` or ``name=https://image`` pairs from memeAliases."""
    aliases: dict[str, str] = {}
    for entry in entries or []:
        name, sep, target = str(entry).partition("=")
        name, target = name.strip(), target.strip()
        if sep and name and target:
            aliases[name] = target
    return aliases


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
    if _is_http_url(template_query.strip()):
        background = template_query.strip()
        if not _safe_background(background):
            return MemePlan(
                None, None, "That image URL is not one I can use (http(s), public host)."
            )
        template = custom_template(background)
    else:
        template = catalog.resolve(template_query)
    if template is None:
        names = ", ".join(f"{t.id} ({t.name})" for t in catalog.suggest(template_query))
        hint = (
            f"Did you mean: {names}?"
            if names
            else "Try @meme list <word>, or paste an image URL as the template."
        )
        return MemePlan(None, None, f"No meme template called '{template_query}'. {hint}")
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
    url = build_meme_url(base, template.id, padded, background=template.background)
    return MemePlan(url, template, None)


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

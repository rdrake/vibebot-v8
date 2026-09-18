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
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
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

# memegen normalises these to ASCII and answers with a 301 to the normalised
# URL — which the downloader refuses on purpose (a redirect could point at a
# private host). Phone keyboards produce all of them. Applied before the
# escape table so the ASCII forms get memegen's own escapes.
_PUNCTUATION_NORMALISE = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
    }
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
    # memegen's alternate images for this template ("bark", "yes", ...).
    # "animated" is the GIF variant and is exposed as ``animated`` instead.
    styles: tuple[str, ...] = ()

    # Stored apart from ``styles`` so that tuple lists only the choices a
    # user would pass to --style; the GIF variant has its own flag, --gif.
    _animated: bool = False

    @property
    def animated(self) -> bool:
        return self._animated


ANIMATED_STYLE = "animated"

# memegen's /fonts/ list. Validated here so a typo is a one-line answer
# instead of a memegen error page fetched and rejected as a non-image.
FONTS = (
    "titilliumweb",
    "titilliumweb-thin",
    "impact",
    "notosans",
    "notosanshebrew",
    "kalam",
    "segoe",
    "hgminchob",
)


@dataclass(frozen=True)
class MemeOptions:
    """The rest of memegen's query string, as the user asked for it."""

    animated: bool = False
    style: str | None = None
    font: str | None = None
    layout_top: bool = False
    # A picture to put into the captioned meme, by an image-edit model —
    # what goes in the Spirit Halloween costume's blank photo, say.
    draw: str | None = None


def custom_template(background: str, name: str = "custom image") -> MemeTemplate:
    return MemeTemplate(CUSTOM_TEMPLATE_ID, name, CUSTOM_LINES, background=background)


def is_http_url(text: str) -> bool:
    return text.lower().startswith(("http://", "https://")) and " " not in text


def _safe_background(url: str) -> bool:
    from .service import validate_external_url

    return is_http_url(url) and validate_external_url(url)


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
    text = text.strip().translate(_PUNCTUATION_NORMALISE)
    if not text:
        return "_"
    for char, escape in _ESCAPES:
        text = text.replace(char, escape)
    return quote(text, safe="~'")


def build_meme_url(
    base: str,
    template_id: str,
    lines: list[str],
    *,
    background: str | None = None,
    options: MemeOptions | None = None,
) -> str:
    options = options or MemeOptions()
    segments = "/".join(encode_caption(line) for line in lines)
    extension = "gif" if options.animated else "png"
    url = f"{base.rstrip('/')}/images/{template_id}/{segments}.{extension}"
    params: list[str] = []
    if options.animated:
        params.append(f"style={ANIMATED_STYLE}")
    elif options.style:
        params.append("style=" + quote(options.style, safe=""))
    if options.font:
        params.append("font=" + quote(options.font, safe=""))
    if options.layout_top:
        params.append("layout=top")
    if background:
        params.append("background=" + quote(background, safe=""))
    return url + ("?" + "&".join(params) if params else "")


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
        raw_styles = [st for st in entry.get("styles") or [] if isinstance(st, str)]
        styles = tuple(st for st in raw_styles if st not in ("default", ANIMATED_STYLE))
        templates.append(
            MemeTemplate(
                template_id,
                name,
                lines,
                keywords,
                example,
                styles=styles,
                _animated=ANIMATED_STYLE in raw_styles,
            )
        )
    return templates


def _normalise(text: str) -> str:
    tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t not in _NOISE_WORDS]
    return " ".join(tokens)


def _haystack(template: MemeTemplate) -> str:
    return _normalise(f"{template.id} {template.name} {' '.join(template.keywords)}")


def _search_text(template: MemeTemplate) -> str:
    """The haystack plus the example captions — what ``list`` and the
    suggestions search. Kept out of :meth:`MemeCatalog.resolve`: "workout"
    should find Butthurt Dweller in a listing, not caption it unasked."""
    return _normalise(f"{_haystack(template)} {' '.join(template.example)}")


def _overlap(query_tokens: list[str], template: MemeTemplate) -> tuple[int, int]:
    """``(words shared with the id or name, words shared with anything)``.

    The name counts first so the topic tags cannot outvote it: "elmo fire"
    must still reach Elmo when three other templates are tagged "fire".
    """
    named = set(_normalise(f"{template.id} {template.name}").split())
    words = set(_haystack(template).split())
    return (
        sum(1 for tok in query_tokens if tok in named),
        sum(1 for tok in query_tokens if tok in words),
    )


class MemeCatalog:
    """The template list plus deterministic name resolution."""

    def __init__(self, templates: list[MemeTemplate]) -> None:
        self.templates = list(templates)
        self._by_id = {t.id.lower(): t for t in self.templates}

    def __len__(self) -> int:
        return len(self.templates)

    def with_keywords(self, extra: Mapping[str, Iterable[str]]) -> MemeCatalog:
        """A copy with more keywords on the named templates.

        Ids the catalog does not have are ignored: the topic table in
        :mod:`meme_topics` outlives any one memegen template list.
        """
        merged = [
            replace(t, keywords=(*t.keywords, *extra[t.id.lower()])) if t.id.lower() in extra else t
            for t in self.templates
        ]
        return MemeCatalog(merged)

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
            if is_http_url(target):
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
        return MemeCatalog(self.with_keywords(extra_keywords).templates + customs)

    def exact(self, query: str) -> MemeTemplate | None:
        """The template whose id or name IS ``query`` — no fuzzy tiers.

        The command uses this to tell "@meme drake" (teach the format) from
        "@meme y'all got any more of them flamethrowers", which the overlap
        tier would also resolve to a template and which is a request, not a
        name.
        """
        q = query.strip().lower()
        if q in self._by_id:
            return self._by_id[q]
        nq = _normalise(q)
        if nq in self._by_id:
            return self._by_id[nq]
        by_name = [t for t in self.templates if _normalise(t.name) == nq]
        return by_name[0] if len(by_name) == 1 else None

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
        scored = sorted(((_overlap(tokens, t), t) for t in self.templates), key=lambda s: s[0])
        scored.reverse()
        best, winner = scored[0]
        if best[1] == 0 or best[1] * 2 < len(tokens):
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
        # "push-up" must not list every "shut up" and "wake up": a word this
        # short only counts when it is the whole query.
        tokens = [tok for tok in nq.split() if len(tok) >= 3] or nq.split()
        scored: list[tuple[int, int, int, str, MemeTemplate]] = []
        for t in self.templates:
            haystack = _search_text(t)
            words = set(haystack.split())
            pos = haystack.find(nq)
            overlap = sum(1 for tok in tokens if tok in words)
            if pos < 0 and overlap == 0:
                continue
            scored.append((0 if pos >= 0 else 1, -overlap, len(t.name), t.id, t))
        scored.sort()
        return [s[4] for s in scored[:limit]]


# One catalog line per template for the picker prompt. Eight tags is enough
# to say what a template is for (memegen's own keywords come first, then the
# topic tags, so the "blank picture area" tag survives the cut); the whole
# brief stays near 30 KB.
_BRIEF_TAGS = 8


def catalog_brief(catalog: MemeCatalog) -> str:
    """The catalog as the picker sees it: ``id | name | N | example | tags``.

    memegen's keywords and the topic tags are one column here; the picker
    does not care which is which. Custom (URL-alias) templates are listed
    too — an operator added them because people ask for them.
    """
    rows: list[str] = []
    for t in catalog.templates:
        example = " / ".join(x or "_" for x in t.example) if t.example else ""
        tags = ", ".join(t.keywords[:_BRIEF_TAGS])
        rows.append(f"{t.id} | {t.name} | {t.lines} | {example} | {tags}")
    return "\n".join(rows)


@dataclass(frozen=True)
class MemeChoice:
    template: MemeTemplate
    lines: list[str]
    draw: str | None = None


def parse_pick(content: str | None, catalog: MemeCatalog) -> MemeChoice | str:
    """Validate the picker's answer against the catalog; a string is why not.

    The model's JSON is ``{"template": id, "lines": [...]}`` or
    ``{"template": null, "reason": ...}``. Anything but an id from the
    catalog is a miss — the whole point of the picker running inside the
    tool is that an invented name never reaches memegen. Too many lines are
    cut, not refused: the model miscounting boxes is not the user's problem.
    """
    from .service import _extract_json_object

    parsed = _extract_json_object(content)
    if not isinstance(parsed, dict):
        return "The picker did not answer."
    template_id = parsed.get("template")
    if not isinstance(template_id, str) or not template_id.strip():
        reason = parsed.get("reason")
        return (
            str(reason).strip() if isinstance(reason, str) and reason.strip() else "Nothing fits."
        )
    template = catalog.resolve(template_id)
    if template is None:
        return f"The picker chose '{template_id}', which is not a template."
    raw_lines = parsed.get("lines")
    if not isinstance(raw_lines, list):
        return "The picker gave no captions."
    lines = [str(x).strip()[:MAX_CAPTION_CHARS] for x in raw_lines][: template.lines]
    if not any(lines):
        return f"The picker chose {template.id} but wrote no captions."
    draw = parsed.get("draw")
    draw = draw.strip()[:MAX_CAPTION_CHARS] if isinstance(draw, str) and draw.strip() else None
    return MemeChoice(template, lines, draw)


def parse_aliases(entries: list[str]) -> dict[str, str]:
    """``name=id`` or ``name=https://image`` pairs from memeAliases."""
    aliases: dict[str, str] = {}
    for entry in entries or []:
        name, sep, target = str(entry).partition("=")
        name, target = name.strip(), target.strip()
        if sep and name and target:
            aliases[name] = target
    return aliases


def describe_short(template: MemeTemplate) -> str:
    """``id (Name, lines[, gif][, styles: a/b])`` — the @meme list line."""
    bits = [template.name, str(template.lines)]
    if template.animated:
        bits.append("gif")
    if template.styles:
        bits.append("styles: " + "/".join(template.styles))
    return f"{template.id} ({', '.join(bits)})"


def _describe(template: MemeTemplate) -> str:
    example = " | ".join(template.example) if template.example else ""
    tail = f" — e.g. {template.id} | {example}" if example else ""
    plural = "caption" if template.lines == 1 else "captions"
    return f"{template.id} ({template.name}) takes {template.lines} {plural}{tail}"


def plan_meme(
    catalog: MemeCatalog,
    base: str,
    template_query: str,
    lines: list[str],
    options: MemeOptions | None = None,
) -> MemePlan:
    """Resolve the name, check the captions and options, build the URL.

    Fewer captions than boxes is fine (the rest render blank — most people
    only want two on a three-box template); more is not, since memegen would
    drop them silently. Every option is checked against what the template
    offers so the answer is a list of choices, not a memegen error page.
    """
    options = options or MemeOptions()
    if is_http_url(template_query.strip()):
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
    option_error = _check_options(catalog, template, options)
    if option_error:
        return MemePlan(None, template, option_error)
    padded = lines + [""] * (template.lines - len(lines))
    # A URL alias keeps its own id so it resolves by name, but memegen only
    # draws on a background under /images/custom/.
    template_id = CUSTOM_TEMPLATE_ID if template.background else template.id
    url = build_meme_url(base, template_id, padded, background=template.background, options=options)
    return MemePlan(url, template, None)


def _check_options(
    catalog: MemeCatalog, template: MemeTemplate, options: MemeOptions
) -> str | None:
    if options.animated and not template.animated:
        animated = ", ".join(t.id for t in catalog.templates if t.animated) or "none"
        return f"{template.id} has no animated version. Animated templates: {animated}."
    if options.style and options.style not in template.styles:
        if not template.styles:
            return f"{template.id} has no styles."
        return f"{template.id} styles: {'/'.join(template.styles)}."
    if options.font and options.font not in FONTS:
        return f"Fonts: {', '.join(FONTS)}."
    return None


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


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args: object, **_kwargs: object) -> None:
        return None


_MAX_CANONICAL_HOPS = 3


def canonical_url(
    url: str,
    *,
    timeout: float,
    opener: urllib.request.OpenerDirector | None = None,
) -> str:
    """Ask memegen for its canonical spelling of ``url``.

    memegen answers a caption it would spell differently (a curly quote, a
    lone dash, "y\u2019all" from a phone keyboard) with a 301 to the canonical
    URL — and the downloader refuses every redirect, since a Location can
    point anywhere. So probe with HEAD first, redirects off, and follow only
    a hop that stays on the same origin under /images/. Anything else —
    no redirect, a redirect elsewhere, a HEAD the server rejects (422 on
    gifs), a network error — leaves the URL as built.
    """
    from urllib.parse import urljoin, urlparse

    opener = opener or urllib.request.build_opener(_NoRedirect())
    origin = urlparse(url)
    current = url
    for _ in range(_MAX_CANONICAL_HOPS):
        request = urllib.request.Request(
            current, method="HEAD", headers={"User-Agent": "vibebot-meme/1.0"}
        )
        try:
            with opener.open(request, timeout=timeout):
                return current
        except urllib.error.HTTPError as err:
            if err.code not in (301, 302, 307, 308):
                return current
            location = err.headers.get("Location") if err.headers else None
        except Exception:  # noqa: BLE001 — a probe must not sink the meme
            return current
        if not location:
            return current
        target = urlparse(urljoin(current, location))
        if (target.scheme, target.netloc) != (origin.scheme, origin.netloc):
            return current
        if not target.path.startswith("/images/"):
            return current
        # memegen drops the query on its redirect; the options were ours.
        current = target._replace(query=origin.query).geturl()
    return current


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
    """Does ``word`` appear in the template's id, name, keywords, or example?"""
    nw = _normalise(word)
    if not nw:
        return True
    return nw in _search_text(template)

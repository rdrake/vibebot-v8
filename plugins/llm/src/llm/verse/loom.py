"""Forest-verse loom orchestrator: rotation, beats, digest, proposal apply."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol, cast

# Model control/special tokens that leak into the visible stream as literal
# text (e.g. a non-reasoning model emitting "<|eos|>" mid-line instead of
# terminating). Mirrors the assistant chokepoint's strip in service.py; loom
# is a standalone module with no service import, so it keeps its own copy.
_CONTROL_TOKEN_PATTERN = re.compile(r"<\|[^|>]*\|>")


class VerseCandidate(NamedTuple):
    channel: str
    weight: int
    """2 * active_avatars + recent_events."""
    last_cycle_at: float | None


class VerseSnapshot(NamedTuple):
    channel: str
    summary: str
    top_entities: list[tuple[str, str, int]]
    """``(kind, name, id)`` triples."""
    recent_events: list[str]
    """Newest-first."""


LOOM_STATIC_PREFIX = """\
You are the loom: a narrator that watches improv between several IRC bots
and proposes mutations to a shared fictional world. Your role is to
*propose*, not to declare canon. A reviewer either approves your proposals
or rejects them.

Each proposal MUST be valid JSON with these fields:
  op          — one of: add_event, set_attribute, add_relation, add_entity, crosspoll_seed
  payload     — object whose required keys depend on op:
                  add_event:     summary (str), entity_ids (list[int])
                  set_attribute: entity_id (int), key (str), value (str)
                  add_relation:  from_id (int), to_id (int), kind (str), note (str?)
                  add_entity:    kind (str: avatar|npc|place|faction|item),
                                 name (str), summary (str?)
                  crosspoll_seed: summary (str), entity_ids (list[int])
                                  — emit only if this verse has crosspoll
                                    send permission; the seed will appear
                                    as a *proposal* in another verse for
                                    that operator to approve or reject.
  confidence  — float between 0.0 and 1.0
  provenance  — short string identifying which transcript line(s) drove this
  rationale   — one sentence in your voice

Always emit the proposal list as a single JSON array, no prose around it.

Each entity in the focus verse appears as `- kind: name (id=N)`. When you
reference an existing entity in `entity_ids`, `from_id`, `to_id`, or
`entity_id`, reuse the id you saw — do not invent ids.
"""


def build_verse_stable_block(snap: VerseSnapshot) -> str:
    """Per-cycle prompt block reused across seed/beat/digest calls.

    Each entity line carries its numeric id so the digest model can
    reference real entities instead of inventing ids.
    """
    parts = [
        f"# Focus verse: {snap.channel}",
        f"# Summary: {snap.summary}",
        "# Active entities:",
    ]
    for kind, name, eid in snap.top_entities:
        parts.append(f"- {kind}: {name} (id={eid})")
    parts.append("# Recent events (newest first):")
    for ev in snap.recent_events:
        parts.append(f"- {ev}")
    return "\n".join(parts)


def build_chimein_tail(*, loom_transcript_so_far: list[tuple[str, str]]) -> str:
    lines = "\n".join(f"{nick}: {text}" for nick, text in loom_transcript_so_far)
    return (
        "You are idling in this channel, watching. The others just spoke, "
        "unprompted:\n"
        f"{lines}\n\n"
        "Chime in with a single line that picks up on what they're doing. "
        "Stay in fiction. One line, ≤ 350 chars. Do NOT emit JSON for this call."
    )


def build_digest_tail(*, loom_transcript_so_far: list[tuple[str, str]]) -> str:
    lines = "\n".join(f"{nick}: {text}" for nick, text in loom_transcript_so_far)
    return (
        "Full transcript:\n"
        f"{lines}\n\n"
        "Now emit a JSON array of proposals derived from this transcript. "
        "If nothing notable happened, emit []."
    )


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$", re.MULTILINE)

_VALID_OPS = ("add_event", "set_attribute", "add_relation", "add_entity", "crosspoll_seed")


def _is_strict_int(v: Any) -> bool:
    """Reject bool, accept int. (bool is a subclass of int in Python.)"""
    return isinstance(v, int) and not isinstance(v, bool)


def _is_int_list(v: Any) -> bool:
    return isinstance(v, list) and all(_is_strict_int(x) for x in v)


_PAYLOAD_SCHEMA: dict[str, tuple[tuple[str, Callable[[Any], bool], str], ...]] = {
    "add_event": (
        ("summary", lambda v: isinstance(v, str), "str"),
        ("entity_ids", _is_int_list, "list[int]"),
    ),
    "set_attribute": (
        ("entity_id", _is_strict_int, "int"),
        ("key", lambda v: isinstance(v, str), "str"),
        ("value", lambda v: isinstance(v, str), "str"),
    ),
    "add_relation": (
        ("from_id", _is_strict_int, "int"),
        ("to_id", _is_strict_int, "int"),
        ("kind", lambda v: isinstance(v, str), "str"),
    ),
    "add_entity": (
        ("kind", lambda v: isinstance(v, str), "str"),
        ("name", lambda v: isinstance(v, str), "str"),
    ),
    "crosspoll_seed": (
        ("summary", lambda v: isinstance(v, str), "str"),
        ("entity_ids", _is_int_list, "list[int]"),
    ),
    "update_entity": (("entity_id", _is_strict_int, "int"),),
}


def validate_payload(op: str, payload: dict[str, Any]) -> str | None:
    """Return None if *payload* is valid for *op*, else a human reason string.

    Shared by parse_digest (loom) and the verse_edit tool so one schema
    governs both. Only constructive ops have entries; an op without a
    schema entry is rejected.
    """
    schema = _PAYLOAD_SCHEMA.get(op)
    if schema is None:
        return f"unknown or non-constructive op: {op!r}"
    for key, predicate, label in schema:
        if key not in payload:
            return f"missing {key}"
        if not predicate(payload[key]):
            return f"{key} not {label}"
    return None


class ParsedProposal(NamedTuple):
    op: str
    payload: dict[str, Any]
    confidence: float
    provenance: str
    rationale: str


def parse_digest(text: str) -> list[ParsedProposal]:
    """Parse a digest-call response into ParsedProposal instances.

    Strips an optional ``json`` code fence, parses JSON, validates each
    proposal's shape, and drops bad proposals with a warning. Returns
    ``[]`` on hard parse error.
    """
    cleaned = _FENCE_RE.sub("", text).strip()
    log = logging.getLogger("llm.verse.loom")
    try:
        raw = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        log.warning("loom digest hard parse error: %s", exc)
        return []
    if not isinstance(raw, list):
        log.warning("loom digest top-level was %s, expected list", type(raw).__name__)
        return []

    out: list[ParsedProposal] = []
    for i, raw_item in enumerate(raw):
        if not isinstance(raw_item, dict):
            log.warning("loom proposal %d not a dict; dropped", i)
            continue
        item = cast("dict[str, Any]", raw_item)
        op = item.get("op")
        if op not in _VALID_OPS:
            log.warning("loom proposal %d bad op %r; dropped", i, op)
            continue
        payload = item.get("payload")
        if not isinstance(payload, dict):
            log.warning("loom proposal %d payload not dict; dropped", i)
            continue
        bad_field = validate_payload(op, payload)
        if bad_field is not None:
            log.warning("loom proposal %d %s; dropped", i, bad_field)
            continue
        try:
            conf = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        out.append(
            ParsedProposal(
                op=op,
                payload=payload,
                confidence=conf,
                provenance=str(item.get("provenance", "")),
                rationale=str(item.get("rationale", "")),
            )
        )
    return out


def _referenced_entity_ids(prop: ParsedProposal) -> list[int]:
    """All entity ids the proposal references, dispatched by op.

    Used by apply_or_queue to bump last_seen_ts on every entity an
    applied/crosspoll-emitted proposal touches. Mirrors the op
    dispatch of ``_proposal_entity_refs_resolve``.

    Returns an empty list for ``add_entity`` (creates a new row;
    nothing to bump) and for any unrecognised op.
    """
    op = prop.op
    payload = prop.payload
    if op == "add_event":
        ids = payload.get("entity_ids") or []
        return [int(x) for x in ids if isinstance(x, int)]
    if op == "set_attribute":
        eid = payload.get("entity_id")
        return [int(eid)] if isinstance(eid, int) else []
    if op == "add_relation":
        out: list[int] = []
        for key in ("from_id", "to_id"):
            v = payload.get(key)
            if isinstance(v, int):
                out.append(int(v))
        return out
    if op == "crosspoll_seed":
        ids = payload.get("entity_ids") or []
        return [int(x) for x in ids if isinstance(x, int)]
    # add_entity and unknown ops bump nothing.
    return []


def _proposal_entity_refs_resolve(store: Any, prop: ParsedProposal) -> bool:
    """True iff every entity id the proposal references resolves to a row.

    The model occasionally emits relations and events with ids it never
    saw in the verse snapshot (e.g. ``from_id=0`` when no entity 0
    exists). We drop those before they reach the operator queue.
    ``add_entity`` creates new rows so it's exempt.
    """
    op = prop.op
    payload = prop.payload
    if op == "add_entity":
        return True
    if op == "set_attribute":
        return store.entity_exists(payload.get("entity_id"))
    if op == "add_relation":
        return store.entity_exists(payload.get("from_id")) and store.entity_exists(
            payload.get("to_id")
        )
    if op == "add_event":
        ids = payload.get("entity_ids") or []
        return all(store.entity_exists(eid) for eid in ids)
    if op == "crosspoll_seed":
        ids = payload.get("entity_ids") or []
        return all(store.entity_exists(eid) for eid in ids)
    return True


class ApplyOutcome(NamedTuple):
    """Outcome of :func:`apply_or_queue`.

    ``outcome`` is one of: ``applied``, ``queued``, ``rejected_invalid_refs``,
    ``crosspoll_emitted``, ``crosspoll_skipped_disabled``,
    ``crosspoll_skipped_limit``. ``seed_id`` is set only on
    ``crosspoll_emitted``.
    """

    outcome: str
    seed_id: int | None = None


def apply_or_queue(
    store: Any,
    prop: ParsedProposal,
    *,
    cycle_id: str,
    threshold: float,
    crosspoll_store: Any | None = None,
    source_channel: str | None = None,
    allow_send: bool = False,
    per_cycle_limit: int = 0,
    already_emitted: int = 0,
) -> ApplyOutcome:
    """Always insert a proposal row OR enqueue a crosspoll seed.

    Crosspoll seeds bypass the per-channel proposals table and instead
    go to the shared crosspoll queue. An audit event row with
    ``source='loom'`` is written locally so the emit shows up in
    ``@verse`` recents.

    Note: the existing ``proposals.op`` CHECK constraint
    (``schema.sql``: ``op IN ('add_event','set_attribute','add_relation','add_entity')``)
    does **not** include ``'crosspoll_seed'``. We do not write a
    ``proposals`` row for any ``crosspoll_seed`` outcome — auto-rejects,
    skips, and successful emits all stay out of the proposals table.
    The local ``events`` audit row covers the success case.

    Non-crosspoll proposals referencing nonexistent entity ids get
    auto-rejected with ``reviewer='auto-validator'`` so the operator's
    pending queue stays clean. Otherwise auto-apply uses
    ``apply_and_record_proposal`` so mutation + audit row land in one
    ``write_transaction``.
    """
    if not _proposal_entity_refs_resolve(store, prop):
        if prop.op == "crosspoll_seed":
            # Cannot insert into proposals (CHECK constraint excludes
            # crosspoll_seed). Drop silently; no real seed was emitted.
            return ApplyOutcome(outcome="rejected_invalid_refs")
        store.add_proposal(
            cycle_id=cycle_id,
            op=prop.op,
            payload=prop.payload,
            confidence=prop.confidence,
            provenance=prop.provenance,
            status="rejected",
            reviewer="auto-validator",
        )
        return ApplyOutcome(outcome="rejected_invalid_refs")

    if prop.op == "crosspoll_seed":
        if not allow_send:
            return ApplyOutcome(outcome="crosspoll_skipped_disabled")
        if already_emitted >= per_cycle_limit:
            return ApplyOutcome(outcome="crosspoll_skipped_limit")
        assert crosspoll_store is not None and source_channel is not None
        seed_id = crosspoll_store.enqueue_seed(
            source_channel=source_channel,
            summary=prop.payload["summary"],
            payload=prop.payload,
        )
        store.add_event(
            summary=f"crosspoll seed emitted: {prop.payload['summary']}",
            entity_ids=prop.payload.get("entity_ids") or [],
            source="loom",
        )
        store.bump_last_seen_ts(_referenced_entity_ids(prop), ts=time.time())
        return ApplyOutcome(outcome="crosspoll_emitted", seed_id=seed_id)

    auto = prop.op != "add_entity" and prop.confidence >= threshold
    if auto:
        store.apply_and_record_proposal(
            cycle_id=cycle_id,
            op=prop.op,
            payload=prop.payload,
            confidence=prop.confidence,
            provenance=prop.provenance,
            reviewer="loom",
        )
        store.bump_last_seen_ts(_referenced_entity_ids(prop), ts=time.time())
        return ApplyOutcome(outcome="applied")
    store.add_proposal(
        cycle_id=cycle_id,
        op=prop.op,
        payload=prop.payload,
        confidence=prop.confidence,
        provenance=prop.provenance,
    )
    return ApplyOutcome(outcome="queued")


def truncate_transcript(
    lines: list[tuple[str, str]],
    *,
    max_lines: int,
    max_chars: int,
) -> list[tuple[str, str]]:
    """Drop consecutive duplicates of the (nick, text) tuple, then cap.

    Caps to ``max_lines`` (most recent kept) and ``max_chars`` (most
    recent kept). Input is oldest-first.
    """
    deduped: list[tuple[str, str]] = []
    for nick, text in lines:
        if deduped and deduped[-1] == (nick, text):
            continue
        deduped.append((nick, text))
    deduped = deduped[-max_lines:]
    out: list[tuple[str, str]] = []
    total = 0
    for nick, text in reversed(deduped):
        if total + len(text) > max_chars:
            break
        out.append((nick, text))
        total += len(text)
    out.reverse()
    return out


def pick_focus_verse(
    candidates: list[VerseCandidate],
    *,
    now: float,
    cooldown_s: int,
    pointer: int,
) -> VerseCandidate | None:
    """Highest-weighted candidate outside cooldown; round-robin ties."""
    eligible = [
        c for c in candidates if c.last_cycle_at is None or (now - c.last_cycle_at) >= cooldown_s
    ]
    if not eligible:
        return None
    top_weight = max(c.weight for c in eligible)
    top = [c for c in eligible if c.weight == top_weight]
    return top[pointer % len(top)]


class LoomCallUsage(NamedTuple):
    prompt_tokens: int
    completion_tokens: int
    cost: float


class LoomModelClient(Protocol):
    def call(
        self, *, op: str, model: str, messages: list[dict[str, str]]
    ) -> tuple[str, LoomCallUsage]: ...


class LiteLLMLoomClient:
    """Default loom client.

    Calls ``litellm.completion`` synchronously (already on a worker thread
    by the time this runs) and returns the content string plus a
    ``LoomCallUsage``. Errors propagate to the caller.
    """

    def __init__(
        self,
        log: logging.Logger | None = None,
        *,
        api_key: str | None = None,
    ) -> None:
        self._log = log or logging.getLogger("llm.verse.loom")
        self._api_key = api_key or None

    def call(
        self, *, op: str, model: str, messages: list[dict[str, str]]
    ) -> tuple[str, LoomCallUsage]:
        import litellm

        t0 = time.monotonic()
        kwargs: dict[str, Any] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        response = litellm.completion(model=model, messages=messages, **kwargs)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        try:
            content = response.choices[0].message.content or ""
        except (AttributeError, IndexError):
            content = ""
        try:
            usage = response.usage
            pt = int(getattr(usage, "prompt_tokens", 0) or 0)
            ct = int(getattr(usage, "completion_tokens", 0) or 0)
        except AttributeError:
            pt = ct = 0
        try:
            cost = float(litellm.completion_cost(completion_response=response, model=model) or 0.0)
        except Exception:
            cost = 0.0
        # Sanity clamp: litellm.completion_cost falls back to a token count
        # for models without pricing data (observed in prod for
        # gemini-flash-lite-latest, returning ~365). Anything over $1 for a
        # single short cheap-model call is implausible — assume the
        # accounting is wrong and zero it out so @usage isn't polluted with
        # nonsense. This is a soft clamp; remove once litellm pricing
        # catches up or once we add explicit pricing tables.
        if cost > 1.0:
            self._log.warning(
                f"loom completion_cost returned implausible value {cost!r} "
                f"for model={model}; clamping to 0.0 (likely missing "
                "pricing data in litellm)"
            )
            cost = 0.0
        # Match service.py:_log_completion_timing's f-string convention.
        # %-args formatting was partially failing under the bot's runtime
        # logger setup (some args substituted, %d ones not) — see #66.
        self._log.warning(
            f"completion_timing op=loom:{op} model={model} "
            f"elapsed_ms={elapsed_ms:.0f} "
            f"prompt_tokens={pt} completion_tokens={ct} cost={cost:.6f}"
        )
        return content, LoomCallUsage(pt, ct, cost)


@dataclass(frozen=True, slots=True)
class LoomConfig:
    """All registry-derived knobs the loom needs for one cycle."""

    network: str
    loom_channel: str
    bot_nicks: tuple[str, ...]
    """Empty tuple = capture all non-self lines (bot-heavy channel default)."""
    model: str
    cycle_interval_s: int
    verse_cooldown_s: int
    beat_window_s: int
    transcript_max_lines: int
    transcript_max_chars: int
    auto_apply_threshold: float
    crosspoll_per_cycle_limit: int = 1


@dataclass
class LoomCycle:
    cycle_id: str
    channel: str
    started_at: float
    verse_stable_block: str
    transcript: list[tuple[str, str]] = field(default_factory=list)
    beats_posted: int = 0
    emitted_seeds: int = 0

    def append_transcript(self, nick: str, text: str) -> None:
        self.transcript.append((nick, text))

    def snapshot_transcript(self) -> list[tuple[str, str]]:
        return list(self.transcript)


class LoomBridge(Protocol):
    """Adapter the plugin implements. The driver only talks to this."""

    def list_candidate_channels(self) -> list[str]: ...
    def candidate_weight(self, channel: str) -> int: ...
    def snapshot(self, channel: str) -> VerseSnapshot: ...
    def post_to_loom_channel(self, text: str) -> bool:
        """Return True if posted; False if the loom Irc is not available."""
        ...

    def schedule_after(self, delay_s: float, fn: Callable[[], None], name: str) -> None: ...
    def submit(self, label: str, fn: Callable[[], None]) -> None:
        """Run *fn* on the LLM worker thread pool. Returns immediately.

        *label* is forwarded to ``LLMExecutor.submit`` for telemetry; loom
        phases pass ``loom:seed`` / ``loom:beat`` / ``loom:digest``.
        """
        ...

    def now(self) -> float: ...
    def store_for(self, channel: str) -> Any: ...
    def log_usage(
        self,
        *,
        channel: str,
        op: str,
        model: str,
        usage: LoomCallUsage,
    ) -> None: ...

    def crosspoll_store(self) -> Any | None: ...
    def verse_allow_send(self, channel: str) -> bool: ...
    def verse_allow_receive(self, channel: str) -> bool: ...


class Loom:
    """Forest-verse loom orchestrator.

    Owns ``_active`` cycle state, ``_last_cycle_by_channel`` cooldowns,
    the round-robin pointer, and a ``threading.Lock`` guarding all of
    them. ``tick`` runs on the scheduler thread; the per-phase workers
    run on the LLM executor and reacquire the lock only to mutate cycle
    state.
    """

    def __init__(
        self,
        *,
        cfg: LoomConfig,
        bridge: LoomBridge,
        client: LoomModelClient,
    ) -> None:
        self._cfg = cfg
        self._bridge = bridge
        self._client = client
        self._active: LoomCycle | None = None
        self._last_chime_at: float | None = None
        self._last_cycle_by_channel: dict[str, float] = {}
        self._pointer = 0
        self._lock = threading.Lock()
        self._log = logging.getLogger("llm.verse.loom")

    def observe_transcript(self, nick: str, text: str) -> None:
        """Reactive trigger. The plugin's doPrivmsg hook calls this for every
        loom-channel line that survived the source filter.

        Runs on the IRC driver thread, so the path stays cheap: lock,
        timestamp compare, list append. If a cycle is active, append. Otherwise,
        if at least ``cycle_interval_s`` has elapsed since the last chime-in,
        form a cycle recording this line as ``transcript[0]`` and offload the
        heavy verse-pick + snapshot + chime-in to the LLM worker.
        """
        with self._lock:
            if self._active is not None:
                self._active.append_transcript(nick, text)
                return
            now = self._bridge.now()
            if (
                self._last_chime_at is not None
                and (now - self._last_chime_at) < self._cfg.cycle_interval_s
            ):
                return
            prev_last_chime = self._last_chime_at
            cycle = LoomCycle(
                cycle_id=uuid.uuid4().hex[:12],
                channel="",
                started_at=now,
                verse_stable_block="",
                transcript=[(nick, text)],
            )
            self._active = cycle
            self._last_chime_at = now
        # Outside the lock: heavy DB work must not block the driver thread.
        self._bridge.submit("loom:open", lambda: self._open_and_chime(cycle, prev_last_chime))

    def _maybe_consume_one_seed_for(self, channel: str) -> None:
        """If this verse opts into receiving, atomically claim one pending
        seed and insert it as a pending ``add_event`` proposal in the
        receiver's table.

        The consume flow is:
          1. Pre-generate ``proposal_id`` (uuid).
          2. ``claim_seed_for`` writes the consumption row in one TX. If
             this caller wins the claim, it returns the seed; otherwise
             ``None`` (no pending seed, or another receiver claimed it).
          3. If we won, insert the local proposal with the pre-generated
             id. If the proposal insert fails, release the consumption
             row so a future cycle (or another receiver) can re-claim
             — without this the seed would be lost forever.

        Bridge-construction failures (``crosspoll_store()`` or
        ``verse_allow_receive`` raising) are logged and swallowed so the
        loom tick continues; the seed/beat/digest path doesn't depend on
        the receive hook.
        """
        try:
            if not self._bridge.verse_allow_receive(channel):
                return
            cx = self._bridge.crosspoll_store()
            if cx is None:
                return
        except Exception:
            self._log.exception("crosspoll: bridge query failed")
            return
        proposal_id = uuid.uuid4().hex
        try:
            seed = cx.claim_seed_for(channel, proposal_id=proposal_id)
        except Exception:
            self._log.exception("crosspoll: claim_seed_for failed")
            return
        if seed is None:
            return
        store = self._bridge.store_for(channel)
        try:
            store.add_proposal(
                cycle_id="crosspoll-recv",
                op="add_event",
                payload={"summary": seed.summary, "entity_ids": []},
                confidence=0.0,
                provenance=(f"crosspoll from {seed.source_channel} (seed-id={seed.id})"),
                proposal_id=proposal_id,
            )
        except Exception:
            self._log.exception(
                "crosspoll: claimed seed %s but proposal insert failed; "
                "releasing claim so a future cycle can re-claim",
                seed.id,
            )
            self._release_claim_with_retry(cx, seed.id, channel)

    def _release_claim_with_retry(
        self, cx: Any, seed_id: int, channel: str, *, attempts: int = 3
    ) -> bool:
        """Release a crosspoll claim, retrying transient failures.

        ``release_claim`` is a single DELETE under a write lock, so a failure
        is almost always a transient SQLite lock that clears on retry. Without
        a successful release the consumption row is permanent and the seed is
        orphaned (lost for this dest forever). Retry a few times; if it still
        fails, log at ERROR with the seed/channel so an operator can clean up.

        Returns True if the claim was released, False if every attempt failed.
        """
        for attempt in range(1, attempts + 1):
            try:
                cx.release_claim(seed_id, channel)
                return True
            except Exception:
                if attempt == attempts:
                    self._log.error(
                        "crosspoll: release_claim failed %d times for seed %s "
                        "dest=%s; consumption row orphaned, seed lost for this "
                        "dest until manual cleanup",
                        attempts,
                        seed_id,
                        channel,
                        exc_info=True,
                    )
                    return False
                self._log.warning(
                    "crosspoll: release_claim attempt %d/%d failed for seed %s; retrying",
                    attempt,
                    attempts,
                    seed_id,
                )
        return False

    def _open_and_chime(self, cycle: LoomCycle, prev_last_chime: float | None) -> None:
        """Worker phase: pick the focus verse, snapshot it, run the crosspoll
        receive hook, then post a single chime-in reacting to the transcript.

        Runs on the LLM executor thread; all the heavy DB reads live here, off
        the IRC driver thread. Four abort paths (no eligible verse, empty
        response, call exception, post failure) roll the cycle back so a
        no-op attempt does not consume the ``cycle_interval_s`` gate.
        """
        now = self._bridge.now()
        channels = self._bridge.list_candidate_channels()
        candidates = [
            VerseCandidate(
                channel=c,
                weight=self._bridge.candidate_weight(c),
                last_cycle_at=self._last_cycle_by_channel.get(c),
            )
            for c in channels
        ]
        choice = pick_focus_verse(
            candidates,
            now=now,
            cooldown_s=self._cfg.verse_cooldown_s,
            pointer=self._pointer,
        )
        if choice is None:
            self._log.debug("loom: no eligible verse at chime-in; rolling back")
            with self._lock:
                self._active = None
                self._last_chime_at = prev_last_chime
            return
        snap = self._bridge.snapshot(choice.channel)
        with self._lock:
            self._pointer += 1
            cycle.channel = choice.channel
            cycle.verse_stable_block = build_verse_stable_block(snap)
            self._last_cycle_by_channel[choice.channel] = now
            transcript = truncate_transcript(
                cycle.snapshot_transcript(),
                max_lines=self._cfg.transcript_max_lines,
                max_chars=self._cfg.transcript_max_chars,
            )
        # Crosspoll receive (unchanged), outside the cycle lock.
        self._maybe_consume_one_seed_for(choice.channel)
        messages = [
            {"role": "system", "content": LOOM_STATIC_PREFIX},
            {"role": "system", "content": cycle.verse_stable_block},
            {"role": "user", "content": build_chimein_tail(loom_transcript_so_far=transcript)},
        ]
        try:
            content, usage = self._client.call(
                op="chimein", model=self._cfg.model, messages=messages
            )
        except Exception:
            self._log.exception("loom chime-in call failed; aborting cycle")
            with self._lock:
                self._active = None
                self._last_cycle_by_channel.pop(choice.channel, None)
                self._last_chime_at = prev_last_chime
            return
        self._bridge.log_usage(
            channel=choice.channel,
            op="chimein",
            model=self._cfg.model,
            usage=usage,
        )
        line = (content.strip().splitlines() or [""])[0]
        # Strip leaked control tokens BEFORE the empty-check, so a line that is
        # nothing but a sentinel (e.g. "<|eos|>") collapses to empty and rolls
        # back instead of posting a blank line to the channel.
        line = _CONTROL_TOKEN_PATTERN.sub("", line).strip()
        if not line:
            # Empty/whitespace response is a no-op attempt: nothing posted, no
            # digest. Roll back exactly like the post-failure path so it does
            # not consume the cycle_interval_s gate or cool down the verse.
            self._log.warning("loom chime-in: empty model response; rolling back cycle")
            with self._lock:
                self._active = None
                self._last_cycle_by_channel.pop(choice.channel, None)
                self._last_chime_at = prev_last_chime
            return
        if not self._bridge.post_to_loom_channel(line):
            self._log.warning(
                "loom chime-in: post_to_loom_channel failed (network down?); "
                "rolling back cycle for %s",
                choice.channel,
            )
            with self._lock:
                self._active = None
                self._last_cycle_by_channel.pop(choice.channel, None)
                self._last_chime_at = prev_last_chime
            return
        with self._lock:
            cycle.beats_posted = 1
        self._bridge.schedule_after(
            self._cfg.beat_window_s,
            self.after_chime,
            "llm_loom_after_chime",
        )

    def after_chime(self) -> None:
        with self._lock:
            cycle = self._active
            if cycle is None:
                return
        self._bridge.submit("loom:digest", lambda: self._digest_phase(cycle))

    def _digest_phase(self, cycle: LoomCycle) -> None:
        try:
            with self._lock:
                transcript = truncate_transcript(
                    cycle.snapshot_transcript(),
                    max_lines=self._cfg.transcript_max_lines,
                    max_chars=self._cfg.transcript_max_chars,
                )
            if not transcript:
                self._log.info(
                    "loom: empty transcript at digest; finalizing cycle %s",
                    cycle.cycle_id,
                )
                return
            messages = [
                {"role": "system", "content": LOOM_STATIC_PREFIX},
                {"role": "system", "content": cycle.verse_stable_block},
                {
                    "role": "user",
                    "content": build_digest_tail(loom_transcript_so_far=transcript),
                },
            ]
            try:
                content, usage = self._client.call(
                    op="digest", model=self._cfg.model, messages=messages
                )
            except Exception:
                self._log.exception("loom digest call failed")
                return
            self._bridge.log_usage(
                channel=cycle.channel,
                op="digest",
                model=self._cfg.model,
                usage=usage,
            )
            proposals = parse_digest(content)
            store = self._bridge.store_for(cycle.channel)
            # Crosspoll-store + allow_send are *only* needed for
            # ``crosspoll_seed`` proposals. Defer their lookup until a
            # seed actually shows up, and isolate failures so a broken
            # crosspoll path doesn't drop non-crosspoll proposals.
            cx: Any | None = None
            cx_failed = False
            try:
                allow_send = self._bridge.verse_allow_send(cycle.channel)
            except Exception:
                self._log.exception("crosspoll: verse_allow_send query failed")
                allow_send = False
            for p in proposals:
                if p.op == "crosspoll_seed" and cx is None and not cx_failed:
                    try:
                        cx = self._bridge.crosspoll_store()
                    except Exception:
                        self._log.exception("crosspoll: store acquisition failed in digest")
                        cx_failed = True
                try:
                    # Snapshot the running emit counter under the lock for
                    # this iteration only — the cap is enforced against
                    # the count *before* this proposal is evaluated.
                    with self._lock:
                        already = cycle.emitted_seeds
                    outcome = apply_or_queue(
                        store,
                        p,
                        cycle_id=cycle.cycle_id,
                        threshold=self._cfg.auto_apply_threshold,
                        crosspoll_store=cx,
                        source_channel=cycle.channel,
                        # If cx is unavailable, gate emit so apply_or_queue
                        # returns ``crosspoll_skipped_disabled`` instead of
                        # asserting on a None store.
                        allow_send=allow_send and cx is not None,
                        per_cycle_limit=self._cfg.crosspoll_per_cycle_limit,
                        already_emitted=already,
                    )
                    if outcome.outcome == "crosspoll_emitted":
                        with self._lock:
                            cycle.emitted_seeds += 1
                except Exception:
                    self._log.exception(
                        "loom proposal apply failed: op=%s payload=%s",
                        p.op,
                        p.payload,
                    )
        finally:
            with self._lock:
                self._active = None

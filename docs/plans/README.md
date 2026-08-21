# Design and implementation plans

Historical records of feature design and implementation planning. These
documents capture intent at design time, not current behaviour: treat
the [guide](../guide/index.md) and the source as the truth.

Active plans live in this directory. When the work ships, the plan moves
to `archive/`.

Active:

- [@animate progress and delivery UX](2026-08-21-animate-ux.md) — a 135s render
  is silent, so a working clip and a failed one look the same. Hold the typing
  indicator for the render, and give the delivered link its context back.
  Designed, not started.
- [Draw refusal follow-up](2026-08-23-draw-refusal-followup.md) — 56% of draws
  were refused; the fixes shipped 2026-08-15/16 but the rewrite loop is still
  unexercised. Waiting on traffic, earliest useful date 2026-08-23.
- [Status-announce restart gap](2026-08-14-status-announce-restart-gap.md) —
  `_status_state` is memory-only, so an incident opening during a restart is
  seeded as already-announced and never fires. Noted, not started.

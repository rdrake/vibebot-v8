# Design and implementation plans

Historical records of feature design and implementation planning. These
documents capture intent at design time, not current behaviour: treat
the [guide](../guide/index.md) and the source as the truth.

Active plans live in this directory. When the work ships, the plan moves
to `archive/`.

Active:

- [Subject dossier pre-stage](2026-09-05-subject-dossier-prestage.md) — image and
  video generators know nothing by name, so a grounded research call in front of
  the `@draw` and `@animate` planners looks up how the real people, places and
  events a request names actually look. Planned, not started.
- [Draw refusal follow-up](2026-08-23-draw-refusal-followup.md) — 56% of draws
  were refused; the fixes shipped 2026-08-15/16 but the rewrite loop is still
  unexercised. Waiting on traffic, earliest useful date 2026-08-23.
- [Status-announce restart gap](2026-08-14-status-announce-restart-gap.md) —
  `_status_state` is memory-only, so an incident opening during a restart is
  seeded as already-announced and never fires. Noted, not started.

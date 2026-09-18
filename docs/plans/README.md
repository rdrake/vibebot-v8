# Design and implementation plans

Historical records of feature design and implementation planning. These
documents capture intent at design time, not current behaviour: treat
the [guide](../guide/index.md) and the source as the truth.

Active plans live in this directory. When the work ships, the plan moves
to `archive/`.

Active:

- [Draw refusal follow-up](2026-08-23-draw-refusal-followup.md) — 56% of draws
  were refused; the fixes shipped 2026-08-15/16 but the rewrite loop is still
  unexercised. Waiting on traffic, earliest useful date 2026-08-23.
- [Status-announce restart gap](2026-08-14-status-announce-restart-gap.md) —
  `_status_state` is memory-only, so an incident opening during a restart is
  seeded as already-announced and never fires. Noted, not started.
- [Plugin INFO log lines missing](2026-09-11-plugin-info-logs-missing.md) —
  `LLM.log.info` and `LLM.bridge` INFO lines never reach `docker logs` on
  prod while WARNING lines from the same code do; the `irc_lookup:` dispatch
  line is therefore invisible. Noted, not started.

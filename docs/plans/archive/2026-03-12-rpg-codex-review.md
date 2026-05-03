# Codex Review — RPG Plugin Design (2026-03-12)

## Verdict
The core concept is strong and differentiated. Keep the Linux-filesystem metaphor and Python-as-source-of-truth architecture. Cut roughly 60% of planned mechanics for v1 or this will stall.

## Findings (ordered by severity)

1. **Critical:** Command surface is too broad for IRC v1 and includes shell syntax that will be fragile in chat parsing.
Reference: design command table lines 25-50.
Impact: High parser complexity, high user error rate, slower gameplay, and painful moderation.
Recommendation: Ship a constrained command set first and treat advanced shell-like syntax as future unlocks.

2. **High:** Shared-room, turn-based party combat with all present users auto-joining will deadlock on AFK/offline players.
Reference: combat model lines 142-149 and party assumption line 148.
Impact: Encounters stall, channel spam increases, players churn.
Recommendation: Use round timers plus auto-actions; remove mandatory full-party participation.

3. **High:** Economy/security exploits are currently implicit in command mappings.
Reference: `cp` cloning line 43, `touch` crafting line 42, `sudo` power line 41, signal attacks line 44/146.
Impact: Infinite item loops, boss trivialization, hard-to-balance progression.
Recommendation: Cut `cp` from v1, gate `sudo` behind scarce resources, limit signals to a tiny fixed set.

4. **High:** “LLM-generated world per campaign” risks inconsistency and state contradictions.
Reference: world generation note lines 83-84.
Impact: Broken exits/items/quests, poor reproducibility, hard bug triage.
Recommendation: Deterministic engine-generated structure; LLM only adds flavor text.

5. **Medium:** Full D&D-style stats/classes/combat depth is too much for IRC onboarding.
Reference: RPG depth line 15, classes/stats lines 133-140.
Impact: New players bounce; balancing and testing cost explodes.
Recommendation: Start class-light and stat-light; add depth after retention proves out.

6. **Medium:** Cross-plugin dependency is underspecified and likely to become tight coupling if done via direct import.
Reference: dependency note line 175 and open question 10.
Impact: Reload/test fragility, config leakage, harder refactors.
Recommendation: Use a shared minimal narrator client module, not direct plugin import.

## Keep vs Cut (opinionated)

### Keep for v1
- Linux filesystem world metaphor.
- Commands: `cd`, `ls`, `ls -a`, `cat`, `rm`, `mv`, `pwd`, `whoami`, `history`, `man`.
- One shared starter map with 10-15 rooms.
- Basic combat loop (HP, attack roll, simple loot, XP).
- SQLite persistence for characters, location, room state, active combat.
- Optional LLM narration with strict output budget and deterministic fallback text.

### Cut from v1
- `cp` item duplication as gameplay.
- Full spell matrix through `chmod` flags.
- Broad `kill -<signal>` matrix.
- Crafting (`touch`) and trading (`chown`) until economy controls exist.
- Procedural world generation by LLM.
- Full quest board system.
- Full six-stat D&D modeling at launch.

## Answers to Open Questions

1. **Command mapping completeness**
Add: `sleep` (rest), `mkdir` (create camp/checkpoint), `ln -s` (waypoint/shortcut), `readlink` (reveal true destination), `du` (inventory weight), `uname` (zone metadata).
Drop or defer: `cp`, free-form `pipe`, complex redirection (`echo ... > merchant`), broad `chmod`/`kill` matrices.
Opinion: Preserve command fantasy, but do not emulate full shell semantics in v1.

2. **Class design**
Use traditional class names (`Warrior`, `Mage`, `Rogue`, `Cleric`) for clarity. Keep Linux flavor as subclass names, skill names, and lore.
Opinion: Linux-themed primary class names are fun for veterans but confusing for casuals and harder to balance communicate.

3. **Combat pacing for IRC**
Use hybrid combat.
- Round length: 20 seconds.
- If player acts: execute action.
- If player is AFK: auto-`defend` on first missed turn, auto-basic attack on second, then marked idle.
- After 3 missed rounds or disconnect: character auto-withdraws to last safe room.
- Trash mobs can auto-resolve in one summary message.
Opinion: Fully manual turns will feel slow; fully auto loses agency.

4. **World generation**
Start hand-crafted plus deterministic expansion.
- v1: fixed starter world.
- v1.1: seeded template expansion (`mount`) owned by engine.
- LLM role: descriptions only, never topology or mechanics.
Opinion: Fully procedural-by-LLM at launch is a reliability trap.

5. **Party mechanics**
Define combat participation at encounter start as a snapshot of active players in room.
- Join: moving into room outside active combat.
- Leave: explicit leave command outside combat, or auto-withdraw on timeout/disconnect.
- Offline mid-dungeon: mark idle, then evac to town/safe room with small penalty.
Opinion: Avoid hard real-time party orchestration in channel chat.

6. **Scope for v1**
Your suggested slice is directionally right; cut further.
Minimum viable v1:
- Movement and room inspection.
- One combat verb (`rm`) and one special (`sudo rm` with charge/cooldown).
- One biome + one boss encounter.
- HP/attack/XP/level only.
- Persistence for character + room + combat.
No quests, crafting, trading, class trees, or procedural map expansion yet.

7. **Signal-based abilities**
Use at most 4 signals total in v1, fixed and documented.
Suggested: `TERM` (finisher), `HUP` (disrupt), `INT` (interrupt cast), `USR1` (class signature).
Make only one class-specific per class in later versions.
Opinion: Mapping many Unix signals will be clever but unreadable and unbalanced.

8. **Permission model (`sudo`)**
`sudo` should be earned, not class-locked.
Model: each player has limited `sudo` charges gained from milestones; each use has personal cooldown plus optional channel cooldown.
Opinion: `sudo` as a class identity narrows composition and increases balance risk.

9. **IRC output length**
Set hard caps.
- Standard action: max 2 lines.
- Combat round summary: max 3 lines.
- Boss/quest milestone: max 4 lines.
- Hard truncate lists (`ls`, `history`) with “+N more”.
Also set narrator token cap low and fallback to deterministic text on timeout.

10. **Cross-plugin dependency**
Do not directly import and call the LLM plugin class.
Preferred: create a shared lightweight narrator client module (LiteLLM wrapper) used by both plugins.
If shared extraction is delayed, RPG should call LiteLLM directly behind its own interface and keep the dependency one-way.
Opinion: direct plugin imports will couple config, lifecycle, and tests too tightly.

## Linux Commands Missing (worth considering)

- `sleep`: rest/recover outside combat.
- `mkdir`: create temporary camp or rally point.
- `ln -s`: create fast-travel shortcut after discovery.
- `readlink`: inspect illusions/portal targets.
- `du`: display inventory burden.
- `uname`: quick area difficulty/biome info.

## Mechanics Likely to Fail in IRC (without redesign)

- Full shell syntax emulation (`|`, redirection, glob-heavy `find`) in raw chat.
- Long turn-by-turn multi-player combat without strict timers.
- Rich crafting/economy before anti-exploit and anti-grief rules.
- Large narration blocks that flood channel context.
- Ambiguous global commands (`cd`, `ls`, `top`) without plugin namespace strategy.

## Scope Concerns and Proposed Phasing

### v1 (ship this)
- Deterministic engine, minimal command set, one dungeon arc, capped narration.

### v1.1
- Add one class layer, simple quest chain, one additional biome.

### v2
- Signal abilities expansion, crafting/trading, seeded world expansion, richer party systems.

## Additional Implementation Notes

- Namespace strategy: prefer `rpg <cmd>` mode or explicit channel opt-in to avoid command collisions.
- Add anti-grief controls early: per-user action cooldown, per-room combat lock, and ignore non-identified users if needed.
- Add replayable deterministic tests for combat and persistence before introducing LLM narration.

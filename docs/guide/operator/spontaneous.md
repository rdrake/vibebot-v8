# Spontaneous participation

Spontaneous participation lets the bot jump into a channel conversation without anyone calling `@ask`. The bot watches recent messages, occasionally rolls a dice on whether to consider a reply, and sends one short message when it has something useful to say. Otherwise it silently passes.

This makes the bot feel like a regular rather than a vending machine. It also burns more tokens than command-driven traffic, so operators turn it on per channel and tune the cadence.

## When to use it

Spontaneous mode fits channels where:

- The bot already participates often through `@ask` or natural-language requests.
- Regulars enjoy a chatty, opinion-having bot rather than a pure tool.
- Operators accept the extra token cost per active hour.

It does not fit support channels, low-traffic announcement channels, or channels where unprompted bot speech would feel out of place.

## Configuration

| Setting | Type | Default | Scope |
|---------|------|---------|-------|
| `spontaneousEnabled` | boolean | `False` | per-channel |
| `spontaneousChance` | integer 1–100 | `15` | per-channel |
| `spontaneousCooldown` | minutes | `2` | per-channel |
| `spontaneousSystemPrompt` | string | (built-in) | per-channel |
| `contextTrackAllMessages` | boolean | `False` | per-channel (prerequisite) |

`contextTrackAllMessages` must be `True` for the channel. Without it, the channel history that feeds the spontaneous evaluation stays empty and the bot has nothing to react to.

### Turn it on for a channel

```
@config channel #afternet plugins.LLM.contextTrackAllMessages True
@config channel #afternet plugins.LLM.spontaneousEnabled True
@flush
```

### Tune the cadence

```
@config channel #afternet plugins.LLM.spontaneousChance 10
@config channel #afternet plugins.LLM.spontaneousCooldown 5
```

`spontaneousChance` is the percent chance the bot evaluates a candidate reply on each message it sees. `spontaneousCooldown` blocks evaluation for that many minutes after the last spontaneous reply, regardless of dice rolls.

The two settings combine to set the volume floor and ceiling. With `spontaneousChance=15` and `spontaneousCooldown=2`, an active channel sees at most one spontaneous reply every two minutes.

### Customize the persona

```
@config channel #afternet plugins.LLM.spontaneousSystemPrompt You are a regular in this channel. Match the tone, jump in only when you have something concrete or funny to add. If the conversation is dead, respond with exactly PASS.
```

The default prompt already instructs the bot to return `PASS` when it has nothing to add. Keep that contract in any custom prompt: the dispatcher reads `PASS` as "stay quiet."

## What happens on each message

For every channel message the bot sees:

1. The bot checks `spontaneousEnabled` for the channel. If `False`, stop.
2. The bot checks the per-channel cooldown. If the last spontaneous reply landed within `spontaneousCooldown` minutes, stop.
3. The bot rolls a 1–100 dice. If the roll exceeds `spontaneousChance`, stop.
4. The bot updates the cooldown timestamp and schedules an evaluation 0.5 seconds out.
5. At evaluation time, the bot reads recent channel history, runs an LLM completion with `spontaneousSystemPrompt`, and either sends the reply or honours a `PASS` response.

The 0.5-second delay lets follow-up messages land in the same evaluation, so the bot reacts to a short burst rather than the first line of one.

## What the bot can and cannot do

- **Tools at evaluation time:** spontaneous evaluations run a basic completion without the bridge or chat tool surface. The bot can speak, but it cannot search, fetch, draw, or set reminders during a spontaneous turn.

- **Length:** spontaneous replies pass through `_collapse_for_irc` to fold any newlines into one line. This avoids the AfterNET Excess Flood disconnect that raw multi-line PRIVMSGs trigger.
- **Memory extraction:** the trigger user's message still flows through memory extraction after the reply, so spontaneous turns can promote or reinforce facts.
- **Usage logging:** each spontaneous reply records as `spontaneous` in the usage table against the bot's own nick, so `@usage` reports include the cost.

## Interaction with other settings

| Setting | Interaction |
|---------|-------------|
| `assistantModel` | Spontaneous evaluations use the same model as `@ask`. A cheaper channel-level model keeps spontaneous cost low. |
| `assistantApiKey` | Required. Without an API key the evaluation aborts after the cooldown updates, which silently consumes the cooldown without a reply. |
| `forestNicks` | No interaction. Spontaneous mode talks to the channel; forest mode reshapes per-user `@ask` replies. |
| `memoryEnabled` | Spontaneous turns still run memory extraction on the trigger message. Disable `memoryEnabled` if you want zero memory writes from passive listening. |
| `enforceRateLimits` | Spontaneous traffic does not consume per-user `@ask` buckets, since no user invoked it. |

## Operational notes

- Spontaneous evaluations show up in the bot's logs as `Spontaneous evaluation failed for #channel` only on exception. Successful `PASS` responses leave no log line, so silent channels and channels with the feature turned off look identical from outside.
- A `PASS` round still costs an LLM completion. Channels with `spontaneousChance=50` and steady traffic generate real cost even when the bot rarely speaks.
- Plugin unload cancels any pending evaluations and clears all cooldowns. Reloading the plugin resets the cooldown floor for every channel.
- `@usage` rolls spontaneous cost under the bot's nick rather than any user. Operators tracking per-user cost should remember that the bot itself becomes a usage account once spontaneous mode is on.

## Turning it off

```
@config channel #afternet plugins.LLM.spontaneousEnabled False
@flush
```

Disabling does not clear `contextTrackAllMessages`. If the channel does not need full message tracking for any other reason, turn that off too:

```
@config channel #afternet plugins.LLM.contextTrackAllMessages False
@flush
```

# AI commands

You don't need commands to ask the bot questions. Mention it by name in a channel or send a PM:

```
<you>     VibeBot, what causes the northern lights?
<VibeBot> The northern lights happen when charged particles from the sun
          collide with gases in Earth's atmosphere...
```

Natural language also manages memories, reminders, instructions, and usage. Use the following commands when you want direct, predictable behaviour, or features such as code generation and image creation.

---

## `ask`

Ask the AI a question. The bot remembers your recent conversation, so follow-up questions work naturally.

**Usage:** `@ask <question>`

```
@ask What is the capital of France?
@ask And what's its population?
@ask Explain quantum entanglement in plain terms
```

### Follow-up questions

The bot keeps conversation context per user, per channel. Your next `@ask` in the same channel carries the history forward:

```
@ask What's the fastest land animal?
  → The cheetah, reaching speeds of 113 km/h.
@ask How does it compare to the fastest bird?
  → The peregrine falcon dives at over 380 km/h...
```

Context expires after a few minutes of inactivity. Use `@forget` to clear it sooner.

### Vision

Include an image URL in your question and the bot analyzes it:

```
@ask What's in this image? https://example.com/photo.jpg
@ask https://i.imgur.com/abc123.png Is this a bird or a plane?
```

More than one image URL in a question works too. The bot detects HTTP and HTTPS image links automatically.

### Custom instructions

Use `@instruct` to set persistent instructions that shape how `@ask` responds to you. See [Custom instructions](memory.md#custom-instructions).

---

## `code`

Generate code, delivered as a syntax-highlighted web page. The bot can search the web for current documentation before generating.

**Usage:** `@code <request>`

```
@code Python function to calculate Fibonacci numbers
@code JavaScript async fetch with error handling
```

### Iterating on code

Conversation context carries over, so you can refine across requests:

```
@code Python function to sort a list of dictionaries by key
  → https://bot.example.com/llm/abc123.html
@code Now add type hints and handle empty lists
  → https://bot.example.com/llm/def456.html
```

If the web server is unavailable, the bot pastes the code directly into IRC instead.

---

## `draw`

Generate an image from a text description.

**Usage:** `@draw <prompt>`

```
@draw A sunset over mountains in watercolour style
@draw A golden retriever in a field of sunflowers, oil painting
```

`@draw` needs an authenticated account. The bot asks you to log in first if you aren't.

Image generation applies content safety filters. If a filter blocks your prompt, the bot rewrites the prompt and retries, so you still get an image, though the result might interpret your request a little differently.

---

## `story`

Generate an illustrated page from a short brief and post a link when it's ready. The bot picks one of two modes from your brief:

- **Story:** an illustrated tale.
- **Explainer:** a concept explained with labelled diagrams, as a learning aid.

**Usage:** `@story <brief>`

```
@story an illustrated tale of the lads winning the pub quiz
@story explain how photosynthesis works, with diagrams
```

The page renders in the background; the bot posts the link when it finishes. `@story` needs an authenticated account, uses the same permission as `@draw`, and has a short per-account cooldown between pages.

---

## `forget`

Clear your conversation context for the current channel, or a named one:

```
@forget
@forget #otherchannel
```

---

## `usage`

View API usage statistics for yourself, another user, or a channel.

**Usage:** `@usage [nick | #channel]`

```
@usage              → your stats plus the channel's, this month
@usage someone      → another user's stats
@usage #somechannel → a channel's stats
```

Natural language works too: "VibeBot, how much have I used this month?"

---

## Rate limits

Each command family carries its own rate limit, and your tier depends on account status: unregistered, registered, or trusted. Image commands (`@draw`, `@story`) have the tightest limits. If you reach a limit, the bot tells you; wait and retry. Admins and owners are exempt.

---

## The verse

In channels with the verse enabled, you can join a persistent shared fiction as an avatar:

| Command | Description |
|---------|-------------|
| `@verseopt in` / `@verseopt out` | Opt your avatar in or out of the channel's verse |
| `@avatar <persona>` | Set the persona that shapes your avatar's voice |
| `@verse` | Show where your avatar currently is |
| `@look [<target>]` | Describe your scene, or a named person or place |
| `@who` | List active avatars and their locations |

Once opted in, your regular channel messages join the story when they reference someone or something in the world, or match the channel's trigger word. Start a message with `//` or wrap it in `((...))` to speak out of character.

# AI commands

You don't need commands to ask the bot questions. Mention it by name in a channel or send a PM:

```
<you>     VibeBot, what causes the northern lights?
<VibeBot> The northern lights happen when charged particles from the sun
          collide with gases in Earth's atmosphere...
```

Plain language covers questions, web lookups, images, code, service status, and saving a memory ("remember that I use Debian"). Reminders and scheduled tasks join that list where an operator has enabled `pendingTasksEnabled`. Managing memories, instructions, usage and context is command-only — use the commands below.

## Icons on a reply

Two icons can appear at the front of a reply. They are the bot telling you
something about how the answer was produced, and both can show at once.

| Icon | Meaning |
|------|---------|
| 🌐 | The answer used a live web lookup rather than the model's own knowledge |
| 🔁 | A content filter refused the image, so the bot reworded the prompt and tried again. You got a picture, but the wording behind it moved, so it may read a little differently from what you asked for |

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

Include an image URL in your question and the bot analyses it:

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

If the page cannot be saved, the bot answers without a link — ask your operator to check `httpRoot` and `httpUrlBase`. A `@code` request that times out is retried in the background and delivered later, pasted inline if the page still cannot be saved.

---

## `draw`

Generate an image from a text description.

**Usage:** `@draw <prompt>`

```
@draw A sunset over mountains in watercolour style
@draw A golden retriever in a field of sunflowers, oil painting
```

`@draw` needs an authenticated account. The bot asks you to log in first if you aren't.

Image generation applies content safety filters. If a filter blocks your prompt, the bot rewords it and tries again — once by default (`drawAutoRewriteMax`). The reworded version keeps your subject, what it is doing, the setting and the style you asked for, and changes only what the filter is likely to have objected to, so you should still recognise the picture you asked for. A reply carrying a reworded image is marked 🔁. If the reworded version is blocked too, the bot says so and tells you it already tried rewording, because rewording it again yourself is not the move — pick a different subject instead.

---

## `animate`

Generate a short video from a text description. Also available as `@video`.

**Usage:** `@animate <prompt>`

```
@animate A slow aerial shot over a pine forest at sunrise, mist in the valleys
@animate A neon sign flickering on a rainy street at night
```

`@animate` needs an authenticated account. The bot asks you to log in first if you aren't.

Rendering takes a minute or two — about 135 seconds for the default seven-second clip. The bot confirms it has started, then posts the link as a reply to your original request when the clip is ready, so your client threads it back to what you asked for and there's no need to wait around. Clips have sound by default.

While the clip renders the bot shows as typing continuously, not just for the first few seconds, so a working render still looks different from a failed one without another line appearing in the channel. Typing stops after six minutes even if the render is still going; the clip still arrives once it's ready. The delivered link also carries your nick and the prompt you gave it — `rdrake: your video is ready! "a corgi riding a unicorn" → https://…mp4` — so it's clear which request it answers even in clients that don't show reply threads.

Because each clip occupies the video hardware exclusively while it renders, `@animate` carries the tightest rate limit of any command, and requests queue behind each other. If the bot is restarted mid-render it picks the job back up afterwards; a clip is only lost if the video server itself restarts.

`@animate` writes the video model's prompt from your request, the same way `@draw` does for pictures. In a channel with the verse enabled, that means naming someone or something from the world gets you a clip of the actual character: the bot looks the canon up and hands the model a description, since the model has never heard of your cast and a bare name would render as words on screen. The shot you asked for, the action, the camera move and the mood, is left alone.

You can also just ask for a video in conversation — the bot has a tool for it and will queue one the same way.

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

The page renders in the background; the bot posts the link when it finishes. `@story` needs an authenticated account, uses the same permission as `@draw`, and allows one page per account every five minutes by default (`verseStorybookCooldownSeconds`).

---

## `forget`

Clear your conversation context for the current channel, or a named one:

```
@forget
@forget #otherchannel
```

Run in a channel, `@forget` also clears that channel's shared recent history, so stale bot answers stop feeding everyone's follow-ups. Naming a different channel clears only your own thread there — you cannot wipe another channel's shared history from outside it.

---

## `usage`

View API usage statistics for yourself, another user, or a channel.

**Usage:** `@usage [nick | #channel]`

```
@usage              → your stats plus the channel's, this month
@usage someone      → another user's stats, scoped to this channel (account-wide by PM)
@usage #somechannel → a channel's stats
```

The bot carries no usage tool and no usage figures in its prompt, so asking in conversation can get you a made-up number.

---

## Rate limits

Each command family carries its own rate limit. Your tier is unregistered or registered depending on whether you are authenticated, or trusted if your operator has granted you the `trusted` capability. Media commands have the tightest limits: `@draw` and `@story` for images, and `@animate` tighter still, since a clip holds the video hardware for over a minute. If you reach a limit, the bot tells you; wait and retry. Admins and owners are exempt.

---

## The verse

In channels with the verse enabled, you can join a persistent shared fiction as an avatar:

| Command | Description |
|---------|-------------|
| `@verseopt in` / `@verseopt out` | Opt your avatar in or out of the channel's verse |
| `@rp <text>` | Speak or act as your avatar for one turn |
| `@rp on` / `@rp off` | Stay in character without prefixing every line |
| `@avatar <persona>` | Set the persona that shapes your avatar's voice |
| `@verse` | Show where your avatar currently is |
| `@look [<target>]` | Show your scene, same as `@verse`, or describe any named person, place, faction, or item in the world |
| `@who` | List active avatars and their locations |

Once opted in, address the bot and mention someone or something in the world, or the channel's trigger word, and you get a tale: six or more paragraphs in your avatar's voice. The channel sees a one-line teaser and a link, because anything longer than a single IRC line is saved to the bot's web server rather than flooded into the channel. Where the operator has enabled `verseStorybookEnabled`, ask it to "illustrate" and you get an illustrated page instead; ask it to "draw" and you get a single image.

The bot stays in character for that turn only. `@rp` gives you a deliberate in-character turn without needing a mention, and `@rp on` holds character across a run of them. Without an avatar, a mention just keeps an ordinary answer true to the world.

Start a message with `//` or wrap it in `((...))` to speak out of character. Questions with nothing to do with the world get a straight answer, not a tall tale.

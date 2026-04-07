# AI Commands

## ask

Ask the AI a question. The bot remembers your recent conversation, so you can ask follow-up questions naturally.

**Usage:** `@ask <question>`

### Examples

```
@ask What is the capital of France?
@ask And what's its population?
@ask Explain quantum entanglement in simple terms
```

### Follow-up questions

The bot automatically maintains conversation context per user per channel. After asking a question, your next `@ask` in the same channel carries the history forward:

```
@ask What's the fastest land animal?
  → The cheetah, reaching speeds of 70 mph.
@ask How does it compare to the fastest bird?
  → The peregrine falcon dives at over 240 mph...
```

Context expires after a few minutes of inactivity. Use `@forget` to clear it manually.

### Vision

Include an image URL in your question and the bot will analyze it:

```
@ask What's in this image? https://example.com/photo.jpg
@ask https://i.imgur.com/abc123.png Is this a bird or a plane?
```

Multiple image URLs work too. The bot detects HTTP/HTTPS image links automatically.

### Custom instructions

Use `@instruct` to set persistent instructions that shape how `@ask` responds to you. See [Custom Instructions](memory.md#custom-instructions) for details.

---

## code

Generate code with syntax highlighting, delivered as an HTTP link.

**Usage:** `@code <request>`

### Examples

```
@code Python function to calculate fibonacci numbers
@code JavaScript async fetch with error handling
@code Rust struct for a binary tree with insert and search
```

### Iterating on code

Like `@ask`, conversation context carries over. You can refine your code across multiple requests:

```
@code Python function to sort a list of dictionaries by key
  → https://bot.example.com/llm/abc123.html
@code Now add type hints and handle empty lists
  → https://bot.example.com/llm/def456.html
```

### Output format

Code is saved to an HTML page with syntax highlighting and returned as a clickable link. If the HTTP server is unavailable, the bot falls back to pasting the code directly into IRC.

---

## draw

Generate an image from a text description.

**Usage:** `@draw <prompt>`

### Examples

```
@draw A sunset over mountains in watercolor style
@draw A cyberpunk cityscape at night
@draw A golden retriever sitting in a field of sunflowers, oil painting
```

### Account required

`@draw` requires you to be authenticated. If you're not authenticated, the bot will ask you to log in first.

### Safety filter

Image generation has content safety filters. If your prompt is blocked by the filter, the bot automatically rewrites your prompt to be compliant and retries. You'll still get an image -- it may just be a slightly adjusted interpretation of your request.

# Routing/Intent Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement sub-project E of the vibebot Go rewrite: the routing/intent layer (pure `router.Route` + stateful `exec.Executor`), with interface contracts and test fakes for sub-projects A/B/C/D/F so this slice is independently testable.

**Architecture:** Pure decision function emits a `RouteDecision` struct → stateful executor consumes it, drives an LLM tool-call loop, delivers via IRC. Cache discipline is enforced via canonically serialized `CachedPrefix` bytes keyed by `CacheScope`. Test fakes for every external interface keep the slice runnable in isolation.

**Tech Stack:** Go 1.23 (`log/slog`, `crypto/sha256`, `encoding/json`), no external runtime dependencies in this slice (real `openai-go`, `ergochat/irc-go`, Gemini SDK enter in sub-projects A/B). Module path: `github.com/rdrake/vibebot-v8/v9`. Code lives in `go/` subdirectory of the existing repo.

**Spec:** `docs/superpowers/specs/2026-05-11-routing-intent-layer-design.md` (commit `2b07346`).

---

## File structure

All paths relative to repo root `/Users/rdrake/workspace/afternet/vibebot-v8/`.

```
go/
├── go.mod                          # module github.com/rdrake/vibebot-v8/v9
├── Makefile                        # build/test/lint targets
├── llmcore/
│   ├── client.go                   # Client interface, CachedPrefix, Message, Completion types
│   └── fake/fake.go                # in-memory fake LLM client
├── tooling/
│   ├── dispatcher.go               # Dispatcher interface, ToolCall, ToolResult
│   └── fake/fake.go                # in-memory fake tool dispatcher
├── overlay/
│   ├── resolver.go                 # Resolver interface
│   └── fake/fake.go                # in-memory fake overlay resolver
├── persist/
│   ├── store.go                    # Store interface
│   └── fake/fake.go                # in-memory fake store
├── ircout/
│   ├── sender.go                   # Sender interface
│   └── fake/fake.go                # in-memory fake IRC sender
├── router/
│   ├── doc.go                      # package docstring
│   ├── types.go                    # IRCEvent, ChannelState, BotState, RouteDecision, ...
│   ├── addressed.go                # addressed() helper
│   ├── addressed_test.go
│   ├── scope.go                    # OverlayHash, CacheScope construction
│   ├── scope_test.go
│   ├── route.go                    # Route() pure function
│   ├── route_test.go
│   └── profile/
│       ├── profile.go              # Profile struct, Registry
│       ├── builtin.go              # quiet, chat, scene, loom, admin
│       ├── profile_test.go
│       └── builtin_test.go
├── exec/
│   ├── doc.go
│   ├── errors.go                   # typed error categories
│   ├── prefix.go                   # CachedPrefix builder, canonical serializer
│   ├── prefix_test.go
│   ├── delivery.go                 # chunking, typing, reply-to fallback
│   ├── delivery_test.go
│   ├── loop.go                     # tool-call loop
│   ├── loop_test.go
│   ├── executor.go                 # Executor struct, Run()
│   └── executor_test.go
└── cmd/
    └── routerdemo/
        └── main.go                 # wires fakes, sends a sample event, prints result
```

Total: 27 tasks. Each task is one logical unit (a struct + its tests, an interface + a smoke check, etc.). Steps within a task are 2–5 minutes each.

---

## Phase 0 — Project skeleton

### Task 1: Initialize Go module and Makefile

**Files:**
- Create: `go/go.mod`
- Create: `go/Makefile`
- Create: `go/.gitignore`

- [ ] **Step 1: Verify Go is installed**

Run: `go version`
Expected: `go version go1.23.x ...` or newer. If missing, install Go 1.23+ before proceeding.

- [ ] **Step 2: Create the module**

```bash
mkdir -p /Users/rdrake/workspace/afternet/vibebot-v8/go
cd /Users/rdrake/workspace/afternet/vibebot-v8/go
go mod init github.com/rdrake/vibebot-v8/v9
```

Expected: `go/go.mod` created with `module github.com/rdrake/vibebot-v8/v9` and `go 1.23` (or newer).

- [ ] **Step 3: Write `go/Makefile`**

```make
.PHONY: build test lint vet fmt all

GO_PKGS := ./...

all: fmt vet lint test build

build:
	go build $(GO_PKGS)

test:
	go test -race -count=1 $(GO_PKGS)

vet:
	go vet $(GO_PKGS)

fmt:
	gofmt -l -w .

lint:
	@command -v golangci-lint >/dev/null 2>&1 || { echo "golangci-lint not installed; skipping"; exit 0; }
	golangci-lint run $(GO_PKGS)
```

- [ ] **Step 4: Write `go/.gitignore`**

```
# Go build artifacts
*.test
*.out
/bin/
/dist/
coverage.txt
```

- [ ] **Step 5: Smoke-build**

Run from `go/`: `make build`
Expected: succeeds with no output (no packages yet, nothing to build, but module is valid).

- [ ] **Step 6: Commit**

```bash
cd /Users/rdrake/workspace/afternet/vibebot-v8
git add go/go.mod go/Makefile go/.gitignore
git commit -m "feat(go): initialize v9 Go module skeleton"
```

---

## Phase 1 — External interface contracts

These five tasks define the interfaces E expects from sub-projects A/B/C/D/F. Each interface lives in its own package. Real implementations land in later sub-project specs; this plan provides only the contracts and test fakes.

### Task 2: Define `llmcore` interface (sub-project B contract)

**Files:**
- Create: `go/llmcore/client.go`

- [ ] **Step 1: Write the file**

```go
// Package llmcore defines the LLM client contract consumed by the executor.
// The real implementation lives in sub-project B.
package llmcore

import "context"

// CachedPrefix is the deterministic byte block that goes into the LLM cache.
// All four fields are serialized into a single cacheable prompt prefix.
// E builds this; B canonicalizes and submits it.
type CachedPrefix struct {
	// FrameworkPrompt is the per-profile system prompt. Stable per Profile.
	FrameworkPrompt string

	// Overlay is the channel-specific overlay text from sub-project D.
	// MUST be a pure function of CacheScope (no user data).
	Overlay string

	// Tools is the canonical JSON schemas for every tool in the profile's
	// allowlist. Sorted by tool name; within each schema, JSON keys sorted
	// alphabetically.
	ToolSchemasJSON []byte

	// ChannelContext is a fixed-key-order block:
	//   "Network: {n}\nChannel: {c}\nProfile: {p}\n"
	ChannelContext string
}

// Message is one entry in the uncached tail (history, memories, user msg,
// or tool result). Role is "system", "user", "assistant", or "tool".
type Message struct {
	Role       string
	Content    string
	ToolCallID string // populated for role=="tool"
}

// ToolCall is what the model emitted.
type ToolCall struct {
	ID        string
	Name      string
	Arguments string // raw JSON from the model
}

// Completion is one LLM response.
type Completion struct {
	Text         string
	ToolCalls    []ToolCall
	CachedTokens int // for cache-hit verification
}

// Scope identifies a cache lane. Mirror of router.CacheScope; lives here
// to avoid an import cycle (llmcore must not import router).
type Scope struct {
	Network     string
	Channel     string
	Profile     string
	OverlayHash string
}

// Client is the contract.
type Client interface {
	// EnsureCache creates or refreshes a CachedContent for scope+prefix and
	// returns an opaque cache name. For providers without explicit cache
	// (e.g. xAI), returns "" and a nil error.
	EnsureCache(ctx context.Context, scope Scope, prefix CachedPrefix) (cacheName string, err error)

	// Complete sends the uncached tail referencing cacheName (when non-empty)
	// or sends the full prefix inline (when empty). Returns one Completion.
	Complete(ctx context.Context, messages []Message, tools []string, model, cacheName string) (Completion, error)
}
```

- [ ] **Step 2: Build to check**

Run from `go/`: `go build ./llmcore/...`
Expected: success, no output.

- [ ] **Step 3: Commit**

```bash
cd /Users/rdrake/workspace/afternet/vibebot-v8
git add go/llmcore/client.go
git commit -m "feat(go): define llmcore.Client interface"
```

---

### Task 3: Define `tooling` interface (sub-project C contract)

**Files:**
- Create: `go/tooling/dispatcher.go`

- [ ] **Step 1: Write the file**

```go
// Package tooling defines the tool-dispatch contract consumed by the executor.
// The real implementation lives in sub-project C.
package tooling

import "context"

// ToolCall is what the LLM asked the bot to run.
type ToolCall struct {
	ID        string
	Name      string
	Arguments string // raw JSON
}

// ToolResult is what to feed back to the LLM.
type ToolResult struct {
	Content string // JSON, or plain text
	Err     error  // non-nil if the tool failed
	Denied  bool   // true if profile/rate disallowed; Err must also be non-nil
}

// Dispatcher resolves a tool name to a handler and runs it within ctx.
type Dispatcher interface {
	// SchemaJSON returns the canonical JSON schema for a tool name.
	// Used by exec.prefix to build CachedPrefix.ToolSchemasJSON.
	// MUST be byte-identical across calls.
	SchemaJSON(name string) ([]byte, error)

	// Dispatch runs a tool call. Honors ctx cancellation. The per-tool
	// timeout is enforced inside Dispatch.
	Dispatch(ctx context.Context, call ToolCall) ToolResult
}
```

- [ ] **Step 2: Build**

Run from `go/`: `go build ./tooling/...`
Expected: success.

- [ ] **Step 3: Commit**

```bash
git add go/tooling/dispatcher.go
git commit -m "feat(go): define tooling.Dispatcher interface"
```

---

### Task 4: Define `overlay` interface (sub-project D contract)

**Files:**
- Create: `go/overlay/resolver.go`

- [ ] **Step 1: Write the file**

```go
// Package overlay defines the persona/overlay resolution contract consumed
// by the executor. The real implementation lives in sub-project D.
package overlay

import "context"

// Scope mirrors llmcore.Scope and router.CacheScope. Lives here to avoid
// import cycles.
type Scope struct {
	Network     string
	Channel     string
	Profile     string
	OverlayHash string
}

// Resolver returns the overlay text for a scope. MUST be a pure function:
// same Scope → byte-identical text, across goroutines and calls.
//
// MAY NOT inject user-specific text (nick, account, timestamps). User-
// specific layering happens in the uncached tail.
type Resolver interface {
	Get(ctx context.Context, scope Scope) (text string, err error)
}
```

- [ ] **Step 2: Build**

Run from `go/`: `go build ./overlay/...`
Expected: success.

- [ ] **Step 3: Commit**

```bash
git add go/overlay/resolver.go
git commit -m "feat(go): define overlay.Resolver interface"
```

---

### Task 5: Define `persist` interface (sub-project F contract)

**Files:**
- Create: `go/persist/store.go`

- [ ] **Step 1: Write the file**

```go
// Package persist defines the persistence contract consumed by the executor.
// The real implementation lives in sub-project F.
package persist

import (
	"context"
	"time"
)

// HistoryEntry is one past turn pulled from storage.
type HistoryEntry struct {
	Role      string // "user" or "assistant"
	Nick      string
	Content   string
	Timestamp time.Time
}

// Memory is one user-facing memory line surfaced into the uncached tail.
type Memory struct {
	ID      int64
	Nick    string // owner of the memory
	Fact    string
	Channel string
}

// UsageRow is what gets recorded after a turn.
type UsageRow struct {
	Timestamp        time.Time
	Network          string
	Channel          string
	Nick             string
	Profile          string
	Model            string
	PromptTokens     int
	CompletionTokens int
	CachedTokens     int
	Cost             float64
	Status           string // "success", "transient_fail", "fatal_fail", "budget_exceeded"
	ErrorDetail      string
}

// Store is the contract.
type Store interface {
	History(ctx context.Context, network, channel string, limit int) ([]HistoryEntry, error)
	Memories(ctx context.Context, network, channel, nick string) ([]Memory, error)
	AppendTurn(ctx context.Context, network, channel, nick, assistantText string) error
	RecordUsage(ctx context.Context, row UsageRow) error
}
```

- [ ] **Step 2: Build**

Run from `go/`: `go build ./persist/...`
Expected: success.

- [ ] **Step 3: Commit**

```bash
git add go/persist/store.go
git commit -m "feat(go): define persist.Store interface"
```

---

### Task 6: Define `ircout` interface (sub-project A output contract)

**Files:**
- Create: `go/ircout/sender.go`

- [ ] **Step 1: Write the file**

```go
// Package ircout defines the IRC output contract consumed by the executor.
// The real implementation lives in sub-project A.
package ircout

import "context"

// SendOpts shapes one outbound message.
type SendOpts struct {
	Network  string
	Target   string // channel or nick
	Text     string // already chunked by the caller; one chunk == one PRIVMSG
	ReplyTo  string // IRCv3 +draft/reply target message-id; "" to omit
	NickPrefix string // fallback "nick: " prefix when ReplyTo CAP is unavailable; "" to omit
}

// TypingState is the value of the typing tag.
type TypingState string

const (
	TypingActive TypingState = "active"
	TypingDone   TypingState = "done"
)

// Sender is the contract.
type Sender interface {
	Send(ctx context.Context, opts SendOpts) error

	// SendTyping is best-effort: it should not return an error if the network
	// doesn't support draft/typing. Best-effort failures are logged but not
	// surfaced.
	SendTyping(ctx context.Context, network, target string, state TypingState)

	// HasReplyCAP returns whether IRCv3 draft/reply was negotiated on this
	// network. Used by the executor to decide between ReplyTo and NickPrefix.
	HasReplyCAP(network string) bool
}
```

- [ ] **Step 2: Build**

Run from `go/`: `go build ./ircout/...`
Expected: success.

- [ ] **Step 3: Commit**

```bash
git add go/ircout/sender.go
git commit -m "feat(go): define ircout.Sender interface"
```

---

## Phase 2 — Test fakes

Each external interface gets an in-memory fake the router/exec tests will use.

### Task 7: `llmcore/fake` — scripted LLM

**Files:**
- Create: `go/llmcore/fake/fake.go`
- Create: `go/llmcore/fake/fake_test.go`

- [ ] **Step 1: Write the failing test**

```go
package fake_test

import (
	"context"
	"testing"

	"github.com/rdrake/vibebot-v8/v9/llmcore"
	"github.com/rdrake/vibebot-v8/v9/llmcore/fake"
)

func TestScriptedCompletions(t *testing.T) {
	c := fake.New()
	c.Script = []llmcore.Completion{
		{Text: "hello"},
		{ToolCalls: []llmcore.ToolCall{{ID: "1", Name: "save_memory", Arguments: `{"fact":"x"}`}}},
		{Text: "done"},
	}

	for i, want := range c.Script {
		got, err := c.Complete(context.Background(), nil, nil, "m", "")
		if err != nil {
			t.Fatalf("iter %d: %v", i, err)
		}
		if got.Text != want.Text {
			t.Fatalf("iter %d text: got %q want %q", i, got.Text, want.Text)
		}
		if len(got.ToolCalls) != len(want.ToolCalls) {
			t.Fatalf("iter %d toolcalls: got %d want %d", i, len(got.ToolCalls), len(want.ToolCalls))
		}
	}
}

func TestEnsureCacheReturnsName(t *testing.T) {
	c := fake.New()
	c.CacheName = "fake-cache-1"
	name, err := c.EnsureCache(context.Background(), llmcore.Scope{Channel: "#x"}, llmcore.CachedPrefix{})
	if err != nil {
		t.Fatal(err)
	}
	if name != "fake-cache-1" {
		t.Fatalf("name: got %q want %q", name, "fake-cache-1")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run from `go/`: `go test ./llmcore/fake/...`
Expected: FAIL — package `fake` doesn't exist yet.

- [ ] **Step 3: Write the fake**

```go
// Package fake provides an in-memory llmcore.Client for tests.
package fake

import (
	"context"
	"errors"
	"sync"

	"github.com/rdrake/vibebot-v8/v9/llmcore"
)

// Client is a scripted llmcore.Client.
type Client struct {
	// Script is the queue of completions returned by Complete, in order.
	Script []llmcore.Completion
	// CacheName is what EnsureCache returns.
	CacheName string
	// EnsureCacheErr is returned by EnsureCache when non-nil.
	EnsureCacheErr error
	// CompleteErr is returned by Complete when non-nil.
	CompleteErr error

	mu sync.Mutex
	i  int

	// Recorded calls (read-only after the test runs).
	EnsureCacheCalls []EnsureCacheCall
	CompleteCalls    []CompleteCall
}

type EnsureCacheCall struct {
	Scope  llmcore.Scope
	Prefix llmcore.CachedPrefix
}

type CompleteCall struct {
	Messages  []llmcore.Message
	Tools     []string
	Model     string
	CacheName string
}

// New returns a zero-value Client ready to use.
func New() *Client { return &Client{} }

func (c *Client) EnsureCache(ctx context.Context, scope llmcore.Scope, prefix llmcore.CachedPrefix) (string, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.EnsureCacheCalls = append(c.EnsureCacheCalls, EnsureCacheCall{scope, prefix})
	if c.EnsureCacheErr != nil {
		return "", c.EnsureCacheErr
	}
	return c.CacheName, nil
}

func (c *Client) Complete(ctx context.Context, messages []llmcore.Message, tools []string, model, cacheName string) (llmcore.Completion, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.CompleteCalls = append(c.CompleteCalls, CompleteCall{messages, tools, model, cacheName})
	if c.CompleteErr != nil {
		return llmcore.Completion{}, c.CompleteErr
	}
	if c.i >= len(c.Script) {
		return llmcore.Completion{}, errors.New("fake: script exhausted")
	}
	out := c.Script[c.i]
	c.i++
	return out, nil
}
```

- [ ] **Step 4: Run test to verify it passes**

Run from `go/`: `go test ./llmcore/fake/...`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add go/llmcore/fake/
git commit -m "feat(go): add llmcore fake client for tests"
```

---

### Task 8: `tooling/fake` — scripted tool dispatcher

**Files:**
- Create: `go/tooling/fake/fake.go`
- Create: `go/tooling/fake/fake_test.go`

- [ ] **Step 1: Write the failing test**

```go
package fake_test

import (
	"context"
	"errors"
	"testing"

	"github.com/rdrake/vibebot-v8/v9/tooling"
	"github.com/rdrake/vibebot-v8/v9/tooling/fake"
)

func TestDispatchReturnsScripted(t *testing.T) {
	d := fake.New()
	d.Results = map[string]tooling.ToolResult{
		"save_memory":  {Content: `{"ok":true}`},
		"search_web":   {Err: errors.New("rate limited"), Denied: true},
	}
	r := d.Dispatch(context.Background(), tooling.ToolCall{Name: "save_memory"})
	if r.Content != `{"ok":true}` {
		t.Fatalf("save_memory: got %q", r.Content)
	}
	r = d.Dispatch(context.Background(), tooling.ToolCall{Name: "search_web"})
	if !r.Denied || r.Err == nil {
		t.Fatalf("search_web: got %+v", r)
	}
}

func TestSchemaJSONStable(t *testing.T) {
	d := fake.New()
	d.Schemas = map[string][]byte{"save_memory": []byte(`{"name":"save_memory"}`)}
	got1, _ := d.SchemaJSON("save_memory")
	got2, _ := d.SchemaJSON("save_memory")
	if string(got1) != string(got2) || string(got1) != `{"name":"save_memory"}` {
		t.Fatalf("schema not stable: %q vs %q", got1, got2)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run from `go/`: `go test ./tooling/fake/...`
Expected: FAIL — package `fake` doesn't exist.

- [ ] **Step 3: Write the fake**

```go
// Package fake provides an in-memory tooling.Dispatcher for tests.
package fake

import (
	"context"
	"errors"
	"sync"

	"github.com/rdrake/vibebot-v8/v9/tooling"
)

type Dispatcher struct {
	Results map[string]tooling.ToolResult
	Schemas map[string][]byte

	mu    sync.Mutex
	Calls []tooling.ToolCall
}

func New() *Dispatcher {
	return &Dispatcher{
		Results: map[string]tooling.ToolResult{},
		Schemas: map[string][]byte{},
	}
}

func (d *Dispatcher) SchemaJSON(name string) ([]byte, error) {
	s, ok := d.Schemas[name]
	if !ok {
		return nil, errors.New("fake: no schema for " + name)
	}
	return s, nil
}

func (d *Dispatcher) Dispatch(ctx context.Context, call tooling.ToolCall) tooling.ToolResult {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.Calls = append(d.Calls, call)
	r, ok := d.Results[call.Name]
	if !ok {
		return tooling.ToolResult{Err: errors.New("fake: no result for " + call.Name)}
	}
	return r
}
```

- [ ] **Step 4: Run test to verify it passes**

Run from `go/`: `go test ./tooling/fake/...`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add go/tooling/fake/
git commit -m "feat(go): add tooling fake dispatcher for tests"
```

---

### Task 9: `overlay/fake` — static-map resolver

**Files:**
- Create: `go/overlay/fake/fake.go`
- Create: `go/overlay/fake/fake_test.go`

- [ ] **Step 1: Write the failing test**

```go
package fake_test

import (
	"context"
	"testing"

	"github.com/rdrake/vibebot-v8/v9/overlay"
	"github.com/rdrake/vibebot-v8/v9/overlay/fake"
)

func TestGetReturnsCanned(t *testing.T) {
	r := fake.New()
	r.Texts = map[overlay.Scope]string{
		{Network: "afternet", Channel: "#x", Profile: "chat", OverlayHash: "abc"}: "hello overlay",
	}
	got, err := r.Get(context.Background(), overlay.Scope{Network: "afternet", Channel: "#x", Profile: "chat", OverlayHash: "abc"})
	if err != nil {
		t.Fatal(err)
	}
	if got != "hello overlay" {
		t.Fatalf("got %q", got)
	}
}

func TestGetIsPureForSameScope(t *testing.T) {
	r := fake.New()
	r.Texts = map[overlay.Scope]string{{Channel: "#x"}: "stable"}
	a, _ := r.Get(context.Background(), overlay.Scope{Channel: "#x"})
	b, _ := r.Get(context.Background(), overlay.Scope{Channel: "#x"})
	if a != b {
		t.Fatalf("not pure: %q vs %q", a, b)
	}
}
```

- [ ] **Step 2: Run to fail**

Run: `go test ./overlay/fake/...`
Expected: FAIL.

- [ ] **Step 3: Write the fake**

```go
package fake

import (
	"context"

	"github.com/rdrake/vibebot-v8/v9/overlay"
)

type Resolver struct {
	Texts map[overlay.Scope]string
	Err   error
}

func New() *Resolver { return &Resolver{Texts: map[overlay.Scope]string{}} }

func (r *Resolver) Get(ctx context.Context, scope overlay.Scope) (string, error) {
	if r.Err != nil {
		return "", r.Err
	}
	return r.Texts[scope], nil
}
```

- [ ] **Step 4: Run to pass**

Run: `go test ./overlay/fake/...`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add go/overlay/fake/
git commit -m "feat(go): add overlay fake resolver for tests"
```

---

### Task 10: `persist/fake` — in-memory store

**Files:**
- Create: `go/persist/fake/fake.go`
- Create: `go/persist/fake/fake_test.go`

- [ ] **Step 1: Write the failing test**

```go
package fake_test

import (
	"context"
	"testing"
	"time"

	"github.com/rdrake/vibebot-v8/v9/persist"
	"github.com/rdrake/vibebot-v8/v9/persist/fake"
)

func TestAppendAndHistory(t *testing.T) {
	s := fake.New()
	ctx := context.Background()
	if err := s.AppendTurn(ctx, "afternet", "#x", "alice", "hi alice"); err != nil {
		t.Fatal(err)
	}
	if err := s.AppendTurn(ctx, "afternet", "#x", "bob", "hi bob"); err != nil {
		t.Fatal(err)
	}
	got, err := s.History(ctx, "afternet", "#x", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 4 {
		t.Fatalf("history len: got %d want 4", len(got))
	}
}

func TestRecordUsage(t *testing.T) {
	s := fake.New()
	row := persist.UsageRow{Timestamp: time.Now(), Profile: "chat", Status: "success"}
	if err := s.RecordUsage(context.Background(), row); err != nil {
		t.Fatal(err)
	}
	if len(s.Usage) != 1 {
		t.Fatalf("usage len: got %d", len(s.Usage))
	}
}
```

- [ ] **Step 2: Run to fail**

Run: `go test ./persist/fake/...`
Expected: FAIL.

- [ ] **Step 3: Write the fake**

```go
package fake

import (
	"context"
	"sync"
	"time"

	"github.com/rdrake/vibebot-v8/v9/persist"
)

type Store struct {
	mu             sync.Mutex
	HistoryEntries []persist.HistoryEntry
	MemoryRows     map[memoryKey][]persist.Memory
	Usage          []persist.UsageRow
}

type memoryKey struct{ network, channel, nick string }

func New() *Store {
	return &Store{MemoryRows: map[memoryKey][]persist.Memory{}}
}

func (s *Store) History(ctx context.Context, network, channel string, limit int) ([]persist.HistoryEntry, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []persist.HistoryEntry
	for _, e := range s.HistoryEntries {
		out = append(out, e)
	}
	if limit > 0 && len(out) > limit {
		out = out[len(out)-limit:]
	}
	return out, nil
}

func (s *Store) Memories(ctx context.Context, network, channel, nick string) ([]persist.Memory, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.MemoryRows[memoryKey{network, channel, nick}], nil
}

func (s *Store) AppendTurn(ctx context.Context, network, channel, nick, assistantText string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.HistoryEntries = append(s.HistoryEntries,
		persist.HistoryEntry{Role: "user", Nick: nick, Content: "(redacted user msg)", Timestamp: time.Now()},
		persist.HistoryEntry{Role: "assistant", Content: assistantText, Timestamp: time.Now()},
	)
	return nil
}

func (s *Store) RecordUsage(ctx context.Context, row persist.UsageRow) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Usage = append(s.Usage, row)
	return nil
}
```

- [ ] **Step 4: Run to pass**

Run: `go test ./persist/fake/...`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add go/persist/fake/
git commit -m "feat(go): add persist fake store for tests"
```

---

### Task 11: `ircout/fake` — recording sender

**Files:**
- Create: `go/ircout/fake/fake.go`
- Create: `go/ircout/fake/fake_test.go`

- [ ] **Step 1: Write the failing test**

```go
package fake_test

import (
	"context"
	"testing"

	"github.com/rdrake/vibebot-v8/v9/ircout"
	"github.com/rdrake/vibebot-v8/v9/ircout/fake"
)

func TestSendRecords(t *testing.T) {
	s := fake.New()
	if err := s.Send(context.Background(), ircout.SendOpts{Target: "#x", Text: "hi"}); err != nil {
		t.Fatal(err)
	}
	if len(s.Sent) != 1 || s.Sent[0].Text != "hi" {
		t.Fatalf("sent: %+v", s.Sent)
	}
}

func TestSendTypingRecords(t *testing.T) {
	s := fake.New()
	s.SendTyping(context.Background(), "afternet", "#x", ircout.TypingActive)
	s.SendTyping(context.Background(), "afternet", "#x", ircout.TypingDone)
	if got := len(s.TypingStates); got != 2 {
		t.Fatalf("typing states: %d", got)
	}
}

func TestHasReplyCAP(t *testing.T) {
	s := fake.New()
	s.ReplyCAP = map[string]bool{"afternet": true}
	if !s.HasReplyCAP("afternet") {
		t.Fatal("should have reply cap")
	}
	if s.HasReplyCAP("other") {
		t.Fatal("should not have reply cap")
	}
}
```

- [ ] **Step 2: Run to fail**

Run: `go test ./ircout/fake/...`
Expected: FAIL.

- [ ] **Step 3: Write the fake**

```go
package fake

import (
	"context"
	"sync"

	"github.com/rdrake/vibebot-v8/v9/ircout"
)

type Sender struct {
	ReplyCAP map[string]bool

	mu           sync.Mutex
	Sent         []ircout.SendOpts
	TypingStates []TypingRecord
	SendErr      error
}

type TypingRecord struct {
	Network string
	Target  string
	State   ircout.TypingState
}

func New() *Sender { return &Sender{ReplyCAP: map[string]bool{}} }

func (s *Sender) Send(ctx context.Context, opts ircout.SendOpts) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.SendErr != nil {
		return s.SendErr
	}
	s.Sent = append(s.Sent, opts)
	return nil
}

func (s *Sender) SendTyping(ctx context.Context, network, target string, state ircout.TypingState) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.TypingStates = append(s.TypingStates, TypingRecord{network, target, state})
}

func (s *Sender) HasReplyCAP(network string) bool { return s.ReplyCAP[network] }
```

- [ ] **Step 4: Run to pass**

Run: `go test ./ircout/fake/...`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add go/ircout/fake/
git commit -m "feat(go): add ircout fake sender for tests"
```

---

## Phase 3 — Profile registry

### Task 12: `router/profile` — Profile struct + Registry

**Files:**
- Create: `go/router/profile/profile.go`
- Create: `go/router/profile/profile_test.go`

- [ ] **Step 1: Write the failing test**

```go
package profile_test

import (
	"testing"

	"github.com/rdrake/vibebot-v8/v9/router/profile"
)

func TestRegistryLookup(t *testing.T) {
	r := profile.NewRegistry()
	p := profile.Profile{
		Name:             "test",
		Tools:            []string{"a", "b"},
		Model:            "m",
		MaxIters:         2,
		FrameworkPrompt:  "fp",
		AllowAmbient:     true,
	}
	r.Register(p)
	got, ok := r.Get("test")
	if !ok {
		t.Fatal("not found")
	}
	if got.Name != "test" || got.MaxIters != 2 {
		t.Fatalf("got %+v", got)
	}
}

func TestRegistryUnknownFallback(t *testing.T) {
	r := profile.NewRegistry()
	r.Register(profile.Profile{Name: "quiet", MaxIters: 2})
	got, ok := r.Get("does-not-exist")
	if ok {
		t.Fatal("should not find missing profile")
	}
	_ = got
}

func TestToolsAreSortedOnRegister(t *testing.T) {
	r := profile.NewRegistry()
	r.Register(profile.Profile{Name: "x", Tools: []string{"c", "a", "b"}})
	p, _ := r.Get("x")
	if got := []string(p.Tools); got[0] != "a" || got[1] != "b" || got[2] != "c" {
		t.Fatalf("not sorted: %v", got)
	}
}
```

- [ ] **Step 2: Run to fail**

Run: `go test ./router/profile/...`
Expected: FAIL — package doesn't exist.

- [ ] **Step 3: Write the package**

```go
// Package profile defines the v9 engagement profiles and the lookup registry.
package profile

import "sort"

// Profile is one engagement mode (e.g. "chat", "scene").
type Profile struct {
	Name            string
	Tools           []string // alphabetical
	Model           string
	MaxIters        int
	FrameworkPrompt string
	AllowAmbient    bool // true if channels MAY enable ambient on this profile
}

// Registry is a lookup table populated at startup.
type Registry struct {
	m map[string]Profile
}

func NewRegistry() *Registry { return &Registry{m: map[string]Profile{}} }

// Register adds (or replaces) a profile. Tool list is canonicalized to
// alphabetical order on insert.
func (r *Registry) Register(p Profile) {
	sort.Strings(p.Tools)
	r.m[p.Name] = p
}

// Get returns the profile and true, or zero and false if not present.
func (r *Registry) Get(name string) (Profile, bool) {
	p, ok := r.m[name]
	return p, ok
}
```

- [ ] **Step 4: Run to pass**

Run: `go test ./router/profile/...`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add go/router/profile/profile.go go/router/profile/profile_test.go
git commit -m "feat(go): add router/profile registry"
```

---

### Task 13: `router/profile/builtin.go` — v9 profile definitions

**Files:**
- Create: `go/router/profile/builtin.go`
- Create: `go/router/profile/builtin_test.go`

- [ ] **Step 1: Write the failing test**

```go
package profile_test

import (
	"testing"

	"github.com/rdrake/vibebot-v8/v9/router/profile"
)

func TestBuiltinsRegistered(t *testing.T) {
	r := profile.NewRegistry()
	profile.RegisterBuiltins(r)
	for _, name := range []string{"quiet", "chat", "scene", "loom", "admin"} {
		if _, ok := r.Get(name); !ok {
			t.Errorf("missing builtin: %s", name)
		}
	}
}

func TestQuietHasNoDestructive(t *testing.T) {
	r := profile.NewRegistry()
	profile.RegisterBuiltins(r)
	p, _ := r.Get("quiet")
	for _, banned := range []string{"delete_memory", "update_memory", "generate_image", "search_web", "fetch_url"} {
		for _, t1 := range p.Tools {
			if t1 == banned {
				t.Errorf("quiet contains banned tool: %s", banned)
			}
		}
	}
}

func TestSceneExcludesDestructiveMemoryAndInstruction(t *testing.T) {
	r := profile.NewRegistry()
	profile.RegisterBuiltins(r)
	p, _ := r.Get("scene")
	for _, banned := range []string{"delete_memory", "update_memory", "set_instruction", "clear_instruction"} {
		for _, t1 := range p.Tools {
			if t1 == banned {
				t.Errorf("scene contains banned tool: %s", banned)
			}
		}
	}
}

func TestChatHasDrawAndSearch(t *testing.T) {
	r := profile.NewRegistry()
	profile.RegisterBuiltins(r)
	p, _ := r.Get("chat")
	gotDraw, gotSearch := false, false
	for _, t1 := range p.Tools {
		if t1 == "generate_image" {
			gotDraw = true
		}
		if t1 == "search_web" {
			gotSearch = true
		}
	}
	if !gotDraw || !gotSearch {
		t.Fatalf("chat missing draw/search: %v", p.Tools)
	}
}

func TestNoRemindersInAnyProfile(t *testing.T) {
	r := profile.NewRegistry()
	profile.RegisterBuiltins(r)
	for _, name := range []string{"quiet", "chat", "scene", "loom", "admin"} {
		p, _ := r.Get(name)
		for _, banned := range []string{"set_reminder", "cancel_pending_task", "cancel_all_pending_tasks", "schedule_llm_task", "list_pending_tasks"} {
			for _, t1 := range p.Tools {
				if t1 == banned {
					t.Errorf("%s contains reminder tool %s", name, banned)
				}
			}
		}
	}
}
```

- [ ] **Step 2: Run to fail**

Run: `go test ./router/profile/...`
Expected: FAIL — `RegisterBuiltins` not defined.

- [ ] **Step 3: Write the builtins**

```go
package profile

// RegisterBuiltins inserts the v9 profile set into r.
func RegisterBuiltins(r *Registry) {
	r.Register(Profile{
		Name:            "quiet",
		Tools:           []string{"clear_instruction", "get_instruction", "list_memories", "save_memory", "set_instruction"},
		Model:           "gemini-2.5-flash",
		MaxIters:        2,
		FrameworkPrompt: "You are a quiet, helpful IRC assistant. Speak only when addressed.",
		AllowAmbient:    false,
	})
	r.Register(Profile{
		Name: "chat",
		Tools: []string{
			"clear_instruction", "delete_memory", "fetch_url", "generate_image",
			"get_instruction", "list_memories", "save_memory", "search_web",
			"set_instruction", "update_memory",
		},
		Model:           "gemini-2.5-flash",
		MaxIters:        2,
		FrameworkPrompt: "You are a conversational IRC assistant.",
		AllowAmbient:    true,
	})
	r.Register(Profile{
		Name: "scene",
		Tools: []string{
			"fetch_url", "generate_image", "get_instruction",
			"list_memories", "save_memory", "search_web",
		},
		Model:           "gemini-2.5-pro",
		MaxIters:        3,
		FrameworkPrompt: "You are participating in an ongoing narrative scene. Stay in character.",
		AllowAmbient:    false,
	})
	r.Register(Profile{
		Name: "loom",
		Tools: []string{
			"fetch_url", "generate_image", "get_instruction",
			"list_memories", "loom_propose", "loom_seed", "loom_yield",
			"save_memory", "search_web",
		},
		Model:           "gemini-2.5-pro",
		MaxIters:        3,
		FrameworkPrompt: "You are one of several voices weaving a collaborative narrative.",
		AllowAmbient:    false,
	})
	r.Register(Profile{
		Name: "admin",
		Tools: []string{
			"clear_instruction", "delete_memory", "fetch_url",
			"generate_code", "generate_image", "get_instruction",
			"list_memories", "save_memory", "search_web",
			"set_instruction", "update_memory",
		},
		Model:           "gemini-2.5-pro",
		MaxIters:        5,
		FrameworkPrompt: "You are the assistant. The user has admin privileges.",
		AllowAmbient:    false,
	})
}
```

- [ ] **Step 4: Run to pass**

Run: `go test ./router/profile/...`
Expected: PASS, all builtin tests green.

- [ ] **Step 5: Commit**

```bash
git add go/router/profile/builtin.go go/router/profile/builtin_test.go
git commit -m "feat(go): add v9 builtin profile definitions"
```

---

## Phase 4 — Router types and helpers

### Task 14: `router/types.go` — data types

**Files:**
- Create: `go/router/doc.go`
- Create: `go/router/types.go`

- [ ] **Step 1: Write doc.go**

```go
// Package router converts inbound IRC events into RouteDecision values.
// Route is a pure function; the stateful executor lives in package exec.
package router
```

- [ ] **Step 2: Write types.go**

```go
package router

import "time"

// IRCEvent is one inbound message from sub-project A.
type IRCEvent struct {
	Network    string
	Channel    string // "" for DM
	Nick       string
	Account    string // SASL-authenticated account, "" if unauth
	Text       string
	IsAction   bool
	Tags       map[string]string
	MessageID  string
	ReceivedAt time.Time
}

// SceneRef and LoomRef are opaque pointers indicating an active mode.
type SceneRef struct{ ID string }
type LoomRef struct{ ID string }

// ChannelKey uniquely identifies a channel per network.
type ChannelKey struct {
	Network string
	Channel string
}

// ChannelState is the persisted runtime state for one channel.
type ChannelState struct {
	Profile         string
	Overlay         string // raw overlay text from D's resolver
	AmbientEnabled  bool
	AmbientCooldown time.Duration
	SceneActive     *SceneRef
	LoomActive      *LoomRef
}

// BotState is the snapshot the dispatcher hands to Route.
type BotState struct {
	SelfNick      string
	LastAmbientAt map[ChannelKey]time.Time
	Now           time.Time
	RecentSentIDs []string
}

// Action is what Route decides.
type Action int

const (
	Ignore Action = iota
	RespondChat
	RespondScene
	RespondLoom
)

func (a Action) String() string {
	switch a {
	case Ignore:
		return "Ignore"
	case RespondChat:
		return "RespondChat"
	case RespondScene:
		return "RespondScene"
	case RespondLoom:
		return "RespondLoom"
	default:
		return "Unknown"
	}
}

// CacheScope is the cache lane identifier.
type CacheScope struct {
	Network     string
	Channel     string
	Profile     string
	OverlayHash string
}

// PromptSpec is the shape of the prompt body (history tail policy + user msg).
type PromptSpec struct {
	HistoryLimit int    // last-N turns to include from the store
	UserText     string // verbatim user message
	UserNick     string // for memory scoping in the tail; never goes into the prefix
}

// DeliverySpec controls how the response goes back to IRC.
type DeliverySpec struct {
	Typing        bool
	ChunkSize     int    // bytes per PRIVMSG chunk
	ReplyToID     string // IRCv3 +draft/reply; "" to skip
	NickPrefixOnFallback bool // if ReplyToID set but CAP missing, use "nick: " prefix
	MaxIters      int
}

// RouteDecision is the output of Route.
type RouteDecision struct {
	Action     Action
	Profile    string
	Tools      []string
	CacheScope CacheScope
	Model      string
	Prompt     PromptSpec
	Delivery   DeliverySpec
	Event      IRCEvent // carried through so the executor has the source event
}
```

- [ ] **Step 3: Build**

Run from `go/`: `go build ./router/...`
Expected: success.

- [ ] **Step 4: Commit**

```bash
git add go/router/doc.go go/router/types.go
git commit -m "feat(go): add router data types"
```

---

### Task 15: `router/addressed.go` — addressed() helper

**Files:**
- Create: `go/router/addressed.go`
- Create: `go/router/addressed_test.go`

- [ ] **Step 1: Write the failing tests**

```go
package router

import "testing"

func TestAddressedDM(t *testing.T) {
	got := addressed(IRCEvent{Channel: "", Text: "anything"}, "vibebot", nil)
	if !got {
		t.Fatal("DM should be addressed")
	}
}

func TestAddressedFirstTokenMatch(t *testing.T) {
	cases := []struct {
		text string
		want bool
	}{
		{"vibebot: hello", true},
		{"vibebot, hi", true},
		{"vibebot; what?", true},
		{"VIBEBOT hi", true},
		{"  vibebot   hi", true},
		{"hi vibebot", false}, // mid-message — explicitly out of scope for v1
		{"vibebotsomething hi", false}, // not equal after punctuation strip
		{"", false},
	}
	for _, c := range cases {
		got := addressed(IRCEvent{Channel: "#x", Text: c.text}, "vibebot", nil)
		if got != c.want {
			t.Errorf("addressed(%q) = %v want %v", c.text, got, c.want)
		}
	}
}

func TestAddressedReplyTag(t *testing.T) {
	recent := []string{"msg-1", "msg-2"}
	evt := IRCEvent{Channel: "#x", Text: "yes", Tags: map[string]string{"+draft/reply": "msg-1"}}
	if !addressed(evt, "vibebot", recent) {
		t.Fatal("reply to recent bot msg should be addressed")
	}
	evt.Tags["+draft/reply"] = "unknown-id"
	if addressed(evt, "vibebot", recent) {
		t.Fatal("reply to non-bot msg should not be addressed")
	}
}
```

- [ ] **Step 2: Run to fail**

Run: `go test ./router/ -run TestAddressed`
Expected: FAIL — `addressed` not defined.

- [ ] **Step 3: Write the function**

```go
package router

import "strings"

// addressed reports whether evt is addressed to selfNick.
//
// True iff ANY of:
//   - evt.Channel == "" (DM)
//   - evt.Tags["+draft/reply"] is in recentSentIDs (bot is being replied to)
//   - first whitespace-delimited token of evt.Text, lowercased and stripped
//     of trailing punctuation in [:,;.!?], equals strings.ToLower(selfNick)
//
// Mid-message mention is intentionally NOT addressed for v1.
func addressed(evt IRCEvent, selfNick string, recentSentIDs []string) bool {
	if evt.Channel == "" {
		return true
	}
	if replyID := evt.Tags["+draft/reply"]; replyID != "" {
		for _, id := range recentSentIDs {
			if id == replyID {
				return true
			}
		}
	}
	first := firstToken(evt.Text)
	if first == "" {
		return false
	}
	return strings.EqualFold(stripTrailingPunct(first), selfNick)
}

func firstToken(s string) string {
	s = strings.TrimLeft(s, " \t")
	end := strings.IndexAny(s, " \t")
	if end == -1 {
		return s
	}
	return s[:end]
}

func stripTrailingPunct(s string) string {
	return strings.TrimRight(s, ":,;.!?")
}
```

- [ ] **Step 4: Run to pass**

Run: `go test ./router/ -run TestAddressed -v`
Expected: PASS for all cases.

- [ ] **Step 5: Commit**

```bash
git add go/router/addressed.go go/router/addressed_test.go
git commit -m "feat(go): add router addressed() with strict v1 semantics"
```

---

### Task 16: `router/scope.go` — OverlayHash + CacheScope builder

**Files:**
- Create: `go/router/scope.go`
- Create: `go/router/scope_test.go`

- [ ] **Step 1: Write the failing tests**

```go
package router

import "testing"

func TestOverlayHashStable(t *testing.T) {
	a := overlayHash("hello world")
	b := overlayHash("hello world")
	if a != b {
		t.Fatalf("not stable: %q vs %q", a, b)
	}
	if len(a) != 32 { // 16 bytes hex
		t.Fatalf("len: %d", len(a))
	}
}

func TestOverlayHashDifferentInputs(t *testing.T) {
	if overlayHash("a") == overlayHash("b") {
		t.Fatal("collision on trivial inputs")
	}
}

func TestBuildCacheScope(t *testing.T) {
	got := buildCacheScope("afternet", "#x", "chat", "overlay text")
	if got.Network != "afternet" || got.Channel != "#x" || got.Profile != "chat" {
		t.Fatalf("scope: %+v", got)
	}
	if got.OverlayHash == "" || len(got.OverlayHash) != 32 {
		t.Fatalf("overlay hash: %q", got.OverlayHash)
	}
}
```

- [ ] **Step 2: Run to fail**

Run: `go test ./router/ -run "TestOverlayHash|TestBuildCacheScope"`
Expected: FAIL.

- [ ] **Step 3: Write the helpers**

```go
package router

import (
	"crypto/sha256"
	"encoding/hex"
)

// overlayHash returns the truncated hex sha256 of overlay text.
// 16 bytes = 32 hex chars. Same input → same hash, byte-identical.
func overlayHash(text string) string {
	sum := sha256.Sum256([]byte(text))
	return hex.EncodeToString(sum[:16])
}

// buildCacheScope assembles the cache key for a turn.
func buildCacheScope(network, channel, profile, overlay string) CacheScope {
	return CacheScope{
		Network:     network,
		Channel:     channel,
		Profile:     profile,
		OverlayHash: overlayHash(overlay),
	}
}
```

- [ ] **Step 4: Run to pass**

Run: `go test ./router/ -run "TestOverlayHash|TestBuildCacheScope" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add go/router/scope.go go/router/scope_test.go
git commit -m "feat(go): add router overlay hash and CacheScope builder"
```

---

## Phase 5 — Route() decision function

### Task 17: `router/route.go` — Route() core decision tree

**Files:**
- Create: `go/router/route.go`
- Create: `go/router/route_test.go`

- [ ] **Step 1: Write the failing tests**

```go
package router

import (
	"testing"
	"time"

	"github.com/rdrake/vibebot-v8/v9/router/profile"
)

func newTestRegistry(t *testing.T) *profile.Registry {
	t.Helper()
	r := profile.NewRegistry()
	profile.RegisterBuiltins(r)
	return r
}

func TestRouteIgnoresUnaddressedNoAmbient(t *testing.T) {
	r := newTestRegistry(t)
	evt := IRCEvent{Channel: "#x", Text: "random chatter", Network: "afternet", ReceivedAt: time.Now()}
	state := ChannelState{Profile: "chat", AmbientEnabled: false}
	bot := BotState{SelfNick: "vibebot", Now: time.Now()}
	d := Route(evt, state, bot, r)
	if d.Action != Ignore {
		t.Fatalf("got %v want Ignore", d.Action)
	}
}

func TestRouteAddressedChat(t *testing.T) {
	r := newTestRegistry(t)
	evt := IRCEvent{Channel: "#x", Text: "vibebot: hi", Network: "afternet", ReceivedAt: time.Now()}
	state := ChannelState{Profile: "chat", Overlay: "be friendly"}
	bot := BotState{SelfNick: "vibebot", Now: time.Now()}
	d := Route(evt, state, bot, r)
	if d.Action != RespondChat {
		t.Fatalf("got %v want RespondChat", d.Action)
	}
	if d.Profile != "chat" {
		t.Fatalf("profile %q", d.Profile)
	}
	if d.CacheScope.OverlayHash == "" {
		t.Fatal("overlay hash empty")
	}
}

func TestRouteAddressedSceneTakesPriority(t *testing.T) {
	r := newTestRegistry(t)
	state := ChannelState{Profile: "chat", SceneActive: &SceneRef{ID: "scene-1"}}
	evt := IRCEvent{Channel: "#x", Text: "vibebot: hi", Network: "afternet", ReceivedAt: time.Now()}
	bot := BotState{SelfNick: "vibebot", Now: time.Now()}
	d := Route(evt, state, bot, r)
	if d.Action != RespondScene {
		t.Fatalf("got %v want RespondScene", d.Action)
	}
}

func TestRouteAddressedLoomTakesPriorityOverScene(t *testing.T) {
	r := newTestRegistry(t)
	state := ChannelState{Profile: "chat", SceneActive: &SceneRef{ID: "s"}, LoomActive: &LoomRef{ID: "l"}}
	evt := IRCEvent{Channel: "#x", Text: "vibebot: hi", Network: "afternet", ReceivedAt: time.Now()}
	bot := BotState{SelfNick: "vibebot", Now: time.Now()}
	d := Route(evt, state, bot, r)
	if d.Action != RespondLoom {
		t.Fatalf("got %v want RespondLoom", d.Action)
	}
}

func TestRouteAmbientPassesCooldown(t *testing.T) {
	r := newTestRegistry(t)
	now := time.Now()
	state := ChannelState{Profile: "chat", AmbientEnabled: true, AmbientCooldown: 60 * time.Second}
	evt := IRCEvent{Channel: "#x", Text: "general chatter", Network: "afternet", ReceivedAt: now}
	bot := BotState{
		SelfNick:      "vibebot",
		Now:           now,
		LastAmbientAt: map[ChannelKey]time.Time{{Network: "afternet", Channel: "#x"}: now.Add(-90 * time.Second)},
	}
	d := Route(evt, state, bot, r)
	if d.Action != RespondChat {
		t.Fatalf("got %v want RespondChat (cooldown elapsed)", d.Action)
	}
}

func TestRouteAmbientBlockedByCooldown(t *testing.T) {
	r := newTestRegistry(t)
	now := time.Now()
	state := ChannelState{Profile: "chat", AmbientEnabled: true, AmbientCooldown: 60 * time.Second}
	evt := IRCEvent{Channel: "#x", Text: "general chatter", Network: "afternet", ReceivedAt: now}
	bot := BotState{
		SelfNick:      "vibebot",
		Now:           now,
		LastAmbientAt: map[ChannelKey]time.Time{{Network: "afternet", Channel: "#x"}: now.Add(-30 * time.Second)},
	}
	d := Route(evt, state, bot, r)
	if d.Action != Ignore {
		t.Fatalf("got %v want Ignore (cooldown not elapsed)", d.Action)
	}
}

func TestRouteAmbientNeverTriggersSceneOrLoom(t *testing.T) {
	r := newTestRegistry(t)
	now := time.Now()
	state := ChannelState{
		Profile:         "chat",
		SceneActive:     &SceneRef{ID: "s"},
		AmbientEnabled:  true,
		AmbientCooldown: time.Second,
	}
	evt := IRCEvent{Channel: "#x", Text: "general chatter", Network: "afternet", ReceivedAt: now}
	bot := BotState{
		SelfNick:      "vibebot",
		Now:           now,
		LastAmbientAt: map[ChannelKey]time.Time{{Network: "afternet", Channel: "#x"}: now.Add(-1 * time.Hour)},
	}
	d := Route(evt, state, bot, r)
	if d.Action != RespondChat {
		t.Fatalf("got %v want RespondChat (ambient never triggers scene)", d.Action)
	}
}

func TestRouteUnknownProfileFallsBackToQuiet(t *testing.T) {
	r := newTestRegistry(t)
	evt := IRCEvent{Channel: "#x", Text: "vibebot: hi", Network: "afternet", ReceivedAt: time.Now()}
	state := ChannelState{Profile: "doesnotexist"}
	bot := BotState{SelfNick: "vibebot", Now: time.Now()}
	d := Route(evt, state, bot, r)
	if d.Profile != "quiet" {
		t.Fatalf("fallback profile: got %q want quiet", d.Profile)
	}
	if d.Action != RespondChat {
		t.Fatalf("action: %v", d.Action)
	}
}

func TestRouteCarriesEvent(t *testing.T) {
	r := newTestRegistry(t)
	evt := IRCEvent{Channel: "#x", Text: "vibebot: hi", Network: "afternet", MessageID: "m-1", ReceivedAt: time.Now()}
	state := ChannelState{Profile: "chat"}
	bot := BotState{SelfNick: "vibebot", Now: time.Now()}
	d := Route(evt, state, bot, r)
	if d.Event.MessageID != "m-1" {
		t.Fatalf("event not carried: %+v", d.Event)
	}
}
```

- [ ] **Step 2: Run to fail**

Run: `go test ./router/ -run TestRoute`
Expected: FAIL — `Route` not defined.

- [ ] **Step 3: Implement Route**

```go
package router

import "github.com/rdrake/vibebot-v8/v9/router/profile"

// Route is the pure decision function. Same inputs → same RouteDecision.
//
// The dispatcher is responsible for ambient claim atomicity (turn lock +
// compare-and-set on LastAmbientAt) before invoking Executor.Run. Route
// reads the snapshot it was given and does not mutate anything.
func Route(evt IRCEvent, state ChannelState, bot BotState, reg *profile.Registry) RouteDecision {
	profileName := state.Profile
	p, ok := reg.Get(profileName)
	if !ok {
		profileName = "quiet"
		p, ok = reg.Get(profileName)
		if !ok {
			// Registry is missing even quiet; return an Ignore decision rather
			// than panicking. The dispatcher logs at WARN.
			return RouteDecision{Action: Ignore, Event: evt}
		}
	}

	action := decideAction(evt, state, bot)
	if action == Ignore {
		return RouteDecision{Action: Ignore, Event: evt, Profile: profileName}
	}

	scope := buildCacheScope(evt.Network, evt.Channel, profileName, state.Overlay)
	return RouteDecision{
		Action:     action,
		Profile:    profileName,
		Tools:      append([]string(nil), p.Tools...), // defensive copy
		CacheScope: scope,
		Model:      p.Model,
		Prompt: PromptSpec{
			HistoryLimit: 20,
			UserText:     evt.Text,
			UserNick:     evt.Nick,
		},
		Delivery: DeliverySpec{
			Typing:               true,
			ChunkSize:            380,
			ReplyToID:            evt.MessageID,
			NickPrefixOnFallback: evt.Channel != "", // skip nick prefix in DMs
			MaxIters:             p.MaxIters,
		},
		Event: evt,
	}
}

func decideAction(evt IRCEvent, state ChannelState, bot BotState) Action {
	if addressed(evt, bot.SelfNick, bot.RecentSentIDs) {
		switch {
		case state.LoomActive != nil:
			return RespondLoom
		case state.SceneActive != nil:
			return RespondScene
		default:
			return RespondChat
		}
	}
	// Not addressed → ambient check
	if !state.AmbientEnabled {
		return Ignore
	}
	key := ChannelKey{Network: evt.Network, Channel: evt.Channel}
	last := bot.LastAmbientAt[key]
	if evt.ReceivedAt.Sub(last) < state.AmbientCooldown {
		return Ignore
	}
	// Ambient ALWAYS routes to RespondChat regardless of scene/loom
	return RespondChat
}
```

- [ ] **Step 4: Run to pass**

Run: `go test ./router/...`
Expected: PASS for all router tests.

- [ ] **Step 5: Commit**

```bash
git add go/router/route.go go/router/route_test.go
git commit -m "feat(go): add router.Route() pure decision function"
```

---

## Phase 6 — Executor

### Task 18: `exec/errors.go` — typed errors

**Files:**
- Create: `go/exec/doc.go`
- Create: `go/exec/errors.go`

- [ ] **Step 1: Write doc.go**

```go
// Package exec runs the LLM tool-call loop and delivers responses to IRC.
// Constructed once at startup with concrete A/B/C/D/F implementations.
package exec
```

- [ ] **Step 2: Write errors.go**

```go
package exec

import "errors"

// Typed errors for category-based recovery.
var (
	ErrLLMTransient    = errors.New("exec: llm transient failure")
	ErrLLMFatal        = errors.New("exec: llm fatal failure")
	ErrToolDenied      = errors.New("exec: tool denied")
	ErrToolFailed      = errors.New("exec: tool execution failed")
	ErrIRCSend         = errors.New("exec: irc send failed")
	ErrCacheStale      = errors.New("exec: cache stale")
	ErrBudgetExceeded  = errors.New("exec: budget exceeded")
)
```

- [ ] **Step 3: Build**

Run from `go/`: `go build ./exec/...`
Expected: success.

- [ ] **Step 4: Commit**

```bash
git add go/exec/doc.go go/exec/errors.go
git commit -m "feat(go): add exec typed errors"
```

---

### Task 19: `exec/prefix.go` — CachedPrefix builder

**Files:**
- Create: `go/exec/prefix.go`
- Create: `go/exec/prefix_test.go`

- [ ] **Step 1: Write the failing tests**

```go
package exec

import (
	"bytes"
	"testing"

	"github.com/rdrake/vibebot-v8/v9/llmcore"
	toolfake "github.com/rdrake/vibebot-v8/v9/tooling/fake"
)

func TestBuildCachedPrefixFieldsSet(t *testing.T) {
	td := toolfake.New()
	td.Schemas = map[string][]byte{
		"a": []byte(`{"name":"a"}`),
		"b": []byte(`{"name":"b"}`),
	}
	cp, err := BuildCachedPrefix(BuildPrefixArgs{
		Framework: "fp",
		Overlay:   "ov",
		Tools:     []string{"a", "b"},
		Network:   "afternet",
		Channel:   "#x",
		Profile:   "chat",
		Dispatch:  td,
	})
	if err != nil {
		t.Fatal(err)
	}
	if cp.FrameworkPrompt != "fp" {
		t.Errorf("framework: %q", cp.FrameworkPrompt)
	}
	if cp.Overlay != "ov" {
		t.Errorf("overlay: %q", cp.Overlay)
	}
	if cp.ChannelContext != "Network: afternet\nChannel: #x\nProfile: chat\n" {
		t.Errorf("channel ctx: %q", cp.ChannelContext)
	}
	// Tools JSON should contain both schemas, in a deterministic envelope.
	if !bytes.Contains(cp.ToolSchemasJSON, []byte(`"name":"a"`)) ||
		!bytes.Contains(cp.ToolSchemasJSON, []byte(`"name":"b"`)) {
		t.Errorf("tool schemas: %s", cp.ToolSchemasJSON)
	}
}

func TestBuildCachedPrefixToolsByteIdentical(t *testing.T) {
	td := toolfake.New()
	td.Schemas = map[string][]byte{
		"a": []byte(`{"name":"a"}`),
		"b": []byte(`{"name":"b"}`),
	}
	args := BuildPrefixArgs{
		Framework: "fp",
		Overlay:   "ov",
		Tools:     []string{"b", "a"}, // unsorted input
		Network:   "afternet",
		Channel:   "#x",
		Profile:   "chat",
		Dispatch:  td,
	}
	first, _ := BuildCachedPrefix(args)
	args.Tools = []string{"a", "b"} // pre-sorted
	second, _ := BuildCachedPrefix(args)
	if !bytes.Equal(first.ToolSchemasJSON, second.ToolSchemasJSON) {
		t.Fatalf("not byte-identical:\nA=%s\nB=%s", first.ToolSchemasJSON, second.ToolSchemasJSON)
	}
}

func TestBuildCachedPrefixMissingToolErrors(t *testing.T) {
	td := toolfake.New() // empty schemas
	_, err := BuildCachedPrefix(BuildPrefixArgs{
		Framework: "fp",
		Overlay:   "ov",
		Tools:     []string{"missing"},
		Network:   "afternet",
		Channel:   "#x",
		Profile:   "chat",
		Dispatch:  td,
	})
	if err == nil {
		t.Fatal("expected error for missing schema")
	}
}

var _ = llmcore.CachedPrefix{} // ensure llmcore is referenced
```

- [ ] **Step 2: Run to fail**

Run: `go test ./exec/ -run TestBuildCachedPrefix`
Expected: FAIL.

- [ ] **Step 3: Implement BuildCachedPrefix**

```go
package exec

import (
	"bytes"
	"fmt"
	"sort"

	"github.com/rdrake/vibebot-v8/v9/llmcore"
	"github.com/rdrake/vibebot-v8/v9/tooling"
)

// BuildPrefixArgs is the input to BuildCachedPrefix.
type BuildPrefixArgs struct {
	Framework string
	Overlay   string
	Tools     []string
	Network   string
	Channel   string
	Profile   string
	Dispatch  tooling.Dispatcher
}

// BuildCachedPrefix assembles the four-block cacheable prefix per the spec
// section "Cache prefix composition". Tools are sorted alphabetically and
// concatenated with a fixed separator into ToolSchemasJSON.
func BuildCachedPrefix(args BuildPrefixArgs) (llmcore.CachedPrefix, error) {
	tools := append([]string(nil), args.Tools...)
	sort.Strings(tools)

	var buf bytes.Buffer
	buf.WriteByte('[')
	for i, name := range tools {
		schema, err := args.Dispatch.SchemaJSON(name)
		if err != nil {
			return llmcore.CachedPrefix{}, fmt.Errorf("schema for %q: %w", name, err)
		}
		if i > 0 {
			buf.WriteByte(',')
		}
		buf.Write(schema)
	}
	buf.WriteByte(']')

	return llmcore.CachedPrefix{
		FrameworkPrompt: args.Framework,
		Overlay:         args.Overlay,
		ToolSchemasJSON: buf.Bytes(),
		ChannelContext:  fmt.Sprintf("Network: %s\nChannel: %s\nProfile: %s\n", args.Network, args.Channel, args.Profile),
	}, nil
}
```

- [ ] **Step 4: Run to pass**

Run: `go test ./exec/ -run TestBuildCachedPrefix -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add go/exec/prefix.go go/exec/prefix_test.go
git commit -m "feat(go): add exec.BuildCachedPrefix with canonical serialization"
```

---

### Task 20: `exec/delivery.go` — chunking + reply fallback

**Files:**
- Create: `go/exec/delivery.go`
- Create: `go/exec/delivery_test.go`

- [ ] **Step 1: Write the failing tests**

```go
package exec

import (
	"testing"
)

func TestChunkAt(t *testing.T) {
	got := chunkAt("aaaaaaaaaaaaaaa", 5)
	if len(got) != 3 {
		t.Fatalf("len: %d", len(got))
	}
	for _, c := range got {
		if len(c) > 5 {
			t.Errorf("chunk too big: %q", c)
		}
	}
}

func TestChunkAtPreservesShort(t *testing.T) {
	got := chunkAt("hi", 380)
	if len(got) != 1 || got[0] != "hi" {
		t.Fatalf("got %v", got)
	}
}

func TestChunkAtZeroSizeReturnsSingle(t *testing.T) {
	got := chunkAt("hello", 0)
	if len(got) != 1 || got[0] != "hello" {
		t.Fatalf("got %v", got)
	}
}
```

- [ ] **Step 2: Run to fail**

Run: `go test ./exec/ -run TestChunkAt`
Expected: FAIL.

- [ ] **Step 3: Implement chunkAt**

```go
package exec

// chunkAt splits s into pieces of at most n bytes. n<=0 returns [s].
//
// This is byte-naive on purpose: IRC PRIVMSG has a byte-length cap (~512
// including envelope). A UTF-8-aware splitter is a future refinement; for
// v1, sub-project A is expected to either reject malformed chunks or guard
// at the wire boundary.
func chunkAt(s string, n int) []string {
	if n <= 0 || len(s) <= n {
		return []string{s}
	}
	var out []string
	for len(s) > n {
		out = append(out, s[:n])
		s = s[n:]
	}
	if len(s) > 0 {
		out = append(out, s)
	}
	return out
}
```

- [ ] **Step 4: Run to pass**

Run: `go test ./exec/ -run TestChunkAt -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add go/exec/delivery.go go/exec/delivery_test.go
git commit -m "feat(go): add exec.chunkAt helper"
```

---

### Task 21: `exec/loop.go` — tool-call loop

**Files:**
- Create: `go/exec/loop.go`
- Create: `go/exec/loop_test.go`

- [ ] **Step 1: Write the failing test**

```go
package exec

import (
	"context"
	"errors"
	"testing"

	"github.com/rdrake/vibebot-v8/v9/llmcore"
	llmfake "github.com/rdrake/vibebot-v8/v9/llmcore/fake"
	"github.com/rdrake/vibebot-v8/v9/tooling"
	toolfake "github.com/rdrake/vibebot-v8/v9/tooling/fake"
)

func TestRunLoopNoToolCallsReturnsImmediately(t *testing.T) {
	llm := llmfake.New()
	llm.Script = []llmcore.Completion{{Text: "hello"}}
	td := toolfake.New()
	text, err := runLoop(context.Background(), runLoopArgs{
		LLM: llm, Tools: td, MaxIters: 3,
		Messages: []llmcore.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if text != "hello" {
		t.Fatalf("text %q", text)
	}
	if got := len(llm.CompleteCalls); got != 1 {
		t.Fatalf("Complete calls: %d", got)
	}
}

func TestRunLoopOneToolCallThenReturn(t *testing.T) {
	llm := llmfake.New()
	llm.Script = []llmcore.Completion{
		{ToolCalls: []llmcore.ToolCall{{ID: "1", Name: "save_memory", Arguments: `{}`}}},
		{Text: "saved!"},
	}
	td := toolfake.New()
	td.Results = map[string]tooling.ToolResult{"save_memory": {Content: `{"ok":true}`}}

	text, err := runLoop(context.Background(), runLoopArgs{
		LLM: llm, Tools: td, MaxIters: 3,
		Messages: []llmcore.Message{{Role: "user", Content: "remember x"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if text != "saved!" {
		t.Fatalf("text %q", text)
	}
	if got := len(llm.CompleteCalls); got != 2 {
		t.Fatalf("Complete calls: %d", got)
	}
	// Second call should have the tool result appended.
	second := llm.CompleteCalls[1].Messages
	if len(second) < 2 || second[len(second)-1].Role != "tool" {
		t.Fatalf("last msg not tool result: %+v", second)
	}
}

func TestRunLoopHitsMaxIters(t *testing.T) {
	llm := llmfake.New()
	llm.Script = []llmcore.Completion{
		{ToolCalls: []llmcore.ToolCall{{ID: "1", Name: "a", Arguments: "{}"}}},
		{ToolCalls: []llmcore.ToolCall{{ID: "2", Name: "a", Arguments: "{}"}}},
	}
	td := toolfake.New()
	td.Results = map[string]tooling.ToolResult{"a": {Content: "{}"}}

	text, err := runLoop(context.Background(), runLoopArgs{
		LLM: llm, Tools: td, MaxIters: 2, FallbackText: "i'm in a loop",
		Messages: []llmcore.Message{{Role: "user", Content: "go"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if text != "i'm in a loop" {
		t.Fatalf("text %q", text)
	}
}

func TestRunLoopPropagatesLLMError(t *testing.T) {
	llm := llmfake.New()
	llm.CompleteErr = errors.New("boom")
	td := toolfake.New()
	_, err := runLoop(context.Background(), runLoopArgs{
		LLM: llm, Tools: td, MaxIters: 2,
		Messages: []llmcore.Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}
```

- [ ] **Step 2: Run to fail**

Run: `go test ./exec/ -run TestRunLoop`
Expected: FAIL.

- [ ] **Step 3: Implement runLoop**

```go
package exec

import (
	"context"
	"encoding/json"

	"github.com/rdrake/vibebot-v8/v9/llmcore"
	"github.com/rdrake/vibebot-v8/v9/tooling"
)

type runLoopArgs struct {
	LLM          llmcore.Client
	Tools        tooling.Dispatcher
	MaxIters     int
	Messages     []llmcore.Message
	ToolsList    []string // for Complete()
	Model        string
	CacheName    string
	FallbackText string
}

// runLoop runs the multi-step tool-call loop. Returns the final assistant
// text or the fallback if MaxIters is reached.
func runLoop(ctx context.Context, args runLoopArgs) (string, error) {
	messages := append([]llmcore.Message(nil), args.Messages...)
	for i := 1; i <= args.MaxIters; i++ {
		completion, err := args.LLM.Complete(ctx, messages, args.ToolsList, args.Model, args.CacheName)
		if err != nil {
			return "", err
		}
		if len(completion.ToolCalls) == 0 {
			return completion.Text, nil
		}
		for _, tc := range completion.ToolCalls {
			result := args.Tools.Dispatch(ctx, tooling.ToolCall{
				ID: tc.ID, Name: tc.Name, Arguments: tc.Arguments,
			})
			messages = append(messages, llmcore.Message{
				Role:       "tool",
				ToolCallID: tc.ID,
				Content:    toolResultPayload(result),
			})
		}
	}
	fallback := args.FallbackText
	if fallback == "" {
		fallback = "I appear to be thinking in circles. Try rephrasing?"
	}
	return fallback, nil
}

func toolResultPayload(r tooling.ToolResult) string {
	type wire struct {
		Content string `json:"content"`
		Error   string `json:"error,omitempty"`
		Denied  bool   `json:"denied,omitempty"`
	}
	w := wire{Content: r.Content}
	if r.Err != nil {
		w.Error = r.Err.Error()
	}
	w.Denied = r.Denied
	b, _ := json.Marshal(w)
	return string(b)
}
```

- [ ] **Step 4: Run to pass**

Run: `go test ./exec/ -run TestRunLoop -v`
Expected: PASS for all four tests.

- [ ] **Step 5: Commit**

```bash
git add go/exec/loop.go go/exec/loop_test.go
git commit -m "feat(go): add exec.runLoop tool-call loop"
```

---

### Task 22: `exec/executor.go` — Executor.Run wiring

**Files:**
- Create: `go/exec/executor.go`
- Create: `go/exec/executor_test.go`

- [ ] **Step 1: Write the failing test**

```go
package exec_test

import (
	"context"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/rdrake/vibebot-v8/v9/exec"
	"github.com/rdrake/vibebot-v8/v9/ircout"
	ircfake "github.com/rdrake/vibebot-v8/v9/ircout/fake"
	"github.com/rdrake/vibebot-v8/v9/llmcore"
	llmfake "github.com/rdrake/vibebot-v8/v9/llmcore/fake"
	"github.com/rdrake/vibebot-v8/v9/overlay"
	overlayfake "github.com/rdrake/vibebot-v8/v9/overlay/fake"
	persistfake "github.com/rdrake/vibebot-v8/v9/persist/fake"
	"github.com/rdrake/vibebot-v8/v9/router"
	"github.com/rdrake/vibebot-v8/v9/router/profile"
	"github.com/rdrake/vibebot-v8/v9/tooling"
	toolfake "github.com/rdrake/vibebot-v8/v9/tooling/fake"
)

func newExecutor(t *testing.T) (*exec.Executor, *llmfake.Client, *ircfake.Sender, *toolfake.Dispatcher, *overlayfake.Resolver) {
	t.Helper()
	llm := llmfake.New()
	td := toolfake.New()
	td.Schemas = map[string][]byte{
		"clear_instruction": []byte(`{"name":"clear_instruction"}`),
		"get_instruction":   []byte(`{"name":"get_instruction"}`),
		"list_memories":     []byte(`{"name":"list_memories"}`),
		"save_memory":       []byte(`{"name":"save_memory"}`),
		"set_instruction":   []byte(`{"name":"set_instruction"}`),
		"delete_memory":     []byte(`{"name":"delete_memory"}`),
		"update_memory":     []byte(`{"name":"update_memory"}`),
		"search_web":        []byte(`{"name":"search_web"}`),
		"fetch_url":         []byte(`{"name":"fetch_url"}`),
		"generate_image":    []byte(`{"name":"generate_image"}`),
	}
	or := overlayfake.New()
	st := persistfake.New()
	sender := ircfake.New()
	e := exec.New(exec.Config{
		LLM:      llm,
		Tools:    td,
		Overlay:  or,
		Store:    st,
		IRC:      sender,
		Log:      slog.New(slog.NewTextHandler(os.Stderr, nil)),
		Registry: builtinRegistry(),
	})
	return e, llm, sender, td, or
}

func builtinRegistry() *profile.Registry {
	r := profile.NewRegistry()
	profile.RegisterBuiltins(r)
	return r
}

func sampleDecision() router.RouteDecision {
	return router.RouteDecision{
		Action:     router.RespondChat,
		Profile:    "chat",
		Tools:      []string{"clear_instruction", "delete_memory", "fetch_url", "generate_image", "get_instruction", "list_memories", "save_memory", "search_web", "set_instruction", "update_memory"},
		CacheScope: router.CacheScope{Network: "afternet", Channel: "#x", Profile: "chat", OverlayHash: "abc"},
		Model:      "gemini-2.5-flash",
		Prompt:     router.PromptSpec{HistoryLimit: 5, UserText: "hi", UserNick: "alice"},
		Delivery:   router.DeliverySpec{Typing: true, ChunkSize: 380, MaxIters: 2},
		Event:      router.IRCEvent{Network: "afternet", Channel: "#x", Nick: "alice", Text: "hi", ReceivedAt: time.Now()},
	}
}

func TestRunSendsResponse(t *testing.T) {
	e, llm, sender, _, _ := newExecutor(t)
	llm.Script = []llmcore.Completion{{Text: "hello alice"}}
	if err := e.Run(context.Background(), sampleDecision()); err != nil {
		t.Fatal(err)
	}
	if len(sender.Sent) == 0 {
		t.Fatal("no irc send")
	}
	if sender.Sent[0].Text != "hello alice" {
		t.Fatalf("sent: %q", sender.Sent[0].Text)
	}
}

func TestRunSendsTypingActiveThenDone(t *testing.T) {
	e, llm, sender, _, _ := newExecutor(t)
	llm.Script = []llmcore.Completion{{Text: "hi"}}
	if err := e.Run(context.Background(), sampleDecision()); err != nil {
		t.Fatal(err)
	}
	if len(sender.TypingStates) < 2 {
		t.Fatalf("typing: %v", sender.TypingStates)
	}
	if sender.TypingStates[0].State != ircout.TypingActive {
		t.Errorf("first: %v", sender.TypingStates[0].State)
	}
	if sender.TypingStates[len(sender.TypingStates)-1].State != ircout.TypingDone {
		t.Errorf("last: %v", sender.TypingStates[len(sender.TypingStates)-1].State)
	}
}

func TestRunTypingDoneFiresOnPanicViaDefer(t *testing.T) {
	e, _, sender, _, _ := newExecutor(t)
	d := sampleDecision()
	// Force a panic by passing a nil registry-resolved tool the dispatcher will
	// fail on for schema lookup. Simulate by setting Tools to one with no schema.
	d.Tools = []string{"nonexistent_tool"}
	// Run should return an error from BuildCachedPrefix, NOT panic — but typing=done
	// must still fire.
	err := e.Run(context.Background(), d)
	if err == nil {
		t.Fatal("expected error")
	}
	if len(sender.TypingStates) == 0 || sender.TypingStates[len(sender.TypingStates)-1].State != ircout.TypingDone {
		t.Fatalf("typing done not sent: %v", sender.TypingStates)
	}
}

func TestRunRecordsUsage(t *testing.T) {
	e, llm, _, _, _ := newExecutor(t)
	llm.Script = []llmcore.Completion{{Text: "hi", CachedTokens: 100}}
	store := e.Store().(*persistfake.Store)
	if err := e.Run(context.Background(), sampleDecision()); err != nil {
		t.Fatal(err)
	}
	if len(store.Usage) != 1 {
		t.Fatalf("usage: %d", len(store.Usage))
	}
	if store.Usage[0].CachedTokens != 100 {
		t.Errorf("cached tokens: %d", store.Usage[0].CachedTokens)
	}
}

var _ = overlay.Scope{} // imported
var _ = tooling.ToolCall{} // imported
```

- [ ] **Step 2: Run to fail**

Run: `go test ./exec/ -run TestRun -v`
Expected: FAIL — `exec.New`, `exec.Executor`, `exec.Config` don't exist.

- [ ] **Step 3: Implement Executor**

```go
package exec

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/rdrake/vibebot-v8/v9/ircout"
	"github.com/rdrake/vibebot-v8/v9/llmcore"
	"github.com/rdrake/vibebot-v8/v9/overlay"
	"github.com/rdrake/vibebot-v8/v9/persist"
	"github.com/rdrake/vibebot-v8/v9/router"
	"github.com/rdrake/vibebot-v8/v9/router/profile"
	"github.com/rdrake/vibebot-v8/v9/tooling"
)

// Config bundles the executor's dependencies.
type Config struct {
	LLM      llmcore.Client
	Tools    tooling.Dispatcher
	Overlay  overlay.Resolver
	Store    persist.Store
	IRC      ircout.Sender
	Log      *slog.Logger
	Registry *profile.Registry
}

// Executor runs RouteDecisions.
type Executor struct {
	cfg Config
}

// New constructs an Executor.
func New(cfg Config) *Executor { return &Executor{cfg: cfg} }

// Store returns the configured persist.Store. Test-only convenience.
func (e *Executor) Store() persist.Store { return e.cfg.Store }

// Run executes one routing decision end-to-end.
func (e *Executor) Run(ctx context.Context, d router.RouteDecision) (err error) {
	if d.Action == router.Ignore {
		return nil
	}

	// Typing on, with deferred off (fires on every exit including panic).
	if d.Delivery.Typing {
		e.cfg.IRC.SendTyping(ctx, d.Event.Network, deliveryTarget(d), ircout.TypingActive)
		defer func() {
			e.cfg.IRC.SendTyping(context.Background(), d.Event.Network, deliveryTarget(d), ircout.TypingDone)
		}()
	}

	prof, _ := e.cfg.Registry.Get(d.Profile)

	// Resolve overlay via D.
	ovText, err := e.cfg.Overlay.Get(ctx, overlay.Scope{
		Network: d.CacheScope.Network, Channel: d.CacheScope.Channel,
		Profile: d.CacheScope.Profile, OverlayHash: d.CacheScope.OverlayHash,
	})
	if err != nil {
		return fmt.Errorf("overlay get: %w", err)
	}

	// Build cached prefix.
	prefix, err := BuildCachedPrefix(BuildPrefixArgs{
		Framework: prof.FrameworkPrompt,
		Overlay:   ovText,
		Tools:     d.Tools,
		Network:   d.CacheScope.Network,
		Channel:   d.CacheScope.Channel,
		Profile:   d.CacheScope.Profile,
		Dispatch:  e.cfg.Tools,
	})
	if err != nil {
		return fmt.Errorf("build prefix: %w", err)
	}

	// Pre-warm cache via B.
	cacheName, err := e.cfg.LLM.EnsureCache(ctx, llmcore.Scope{
		Network: d.CacheScope.Network, Channel: d.CacheScope.Channel,
		Profile: d.CacheScope.Profile, OverlayHash: d.CacheScope.OverlayHash,
	}, prefix)
	if err != nil {
		return fmt.Errorf("ensure cache: %w", err)
	}

	// Hydrate uncached tail.
	history, err := e.cfg.Store.History(ctx, d.Event.Network, d.Event.Channel, d.Prompt.HistoryLimit)
	if err != nil {
		return fmt.Errorf("history: %w", err)
	}
	memories, err := e.cfg.Store.Memories(ctx, d.Event.Network, d.Event.Channel, d.Prompt.UserNick)
	if err != nil {
		return fmt.Errorf("memories: %w", err)
	}
	messages := buildTail(history, memories, d.Prompt.UserText, d.Prompt.UserNick)

	// Capture cached token count from the LLM response stream.
	var lastCached int
	wrapped := &cacheCounter{Client: e.cfg.LLM, last: &lastCached}

	text, err := runLoop(ctx, runLoopArgs{
		LLM:       wrapped,
		Tools:     e.cfg.Tools,
		MaxIters:  d.Delivery.MaxIters,
		Messages:  messages,
		ToolsList: d.Tools,
		Model:     d.Model,
		CacheName: cacheName,
	})
	if err != nil {
		_ = e.cfg.Store.RecordUsage(ctx, persist.UsageRow{
			Timestamp: time.Now(),
			Network:   d.Event.Network, Channel: d.Event.Channel, Nick: d.Event.Nick,
			Profile: d.Profile, Model: d.Model, Status: "fatal_fail",
			ErrorDetail: err.Error(),
		})
		return err
	}

	// Deliver.
	if err := e.deliver(ctx, d, text); err != nil {
		return fmt.Errorf("deliver: %w", err)
	}

	// Persist.
	if err := e.cfg.Store.AppendTurn(ctx, d.Event.Network, d.Event.Channel, d.Event.Nick, text); err != nil {
		e.cfg.Log.Warn("append turn failed", "err", err)
	}
	_ = e.cfg.Store.RecordUsage(ctx, persist.UsageRow{
		Timestamp:    time.Now(),
		Network:      d.Event.Network,
		Channel:      d.Event.Channel,
		Nick:         d.Event.Nick,
		Profile:      d.Profile,
		Model:        d.Model,
		CachedTokens: lastCached,
		Status:       "success",
	})

	return nil
}

func deliveryTarget(d router.RouteDecision) string {
	if d.Event.Channel != "" {
		return d.Event.Channel
	}
	return d.Event.Nick
}

func buildTail(history []persist.HistoryEntry, memories []persist.Memory, userText, userNick string) []llmcore.Message {
	msgs := make([]llmcore.Message, 0, len(history)+len(memories)+1)
	for _, h := range history {
		msgs = append(msgs, llmcore.Message{Role: h.Role, Content: h.Content})
	}
	if len(memories) > 0 {
		var content string
		for _, m := range memories {
			content += "- " + m.Fact + "\n"
		}
		msgs = append(msgs, llmcore.Message{Role: "system", Content: "Memories about " + userNick + ":\n" + content})
	}
	msgs = append(msgs, llmcore.Message{Role: "user", Content: userText})
	return msgs
}

func (e *Executor) deliver(ctx context.Context, d router.RouteDecision, text string) error {
	chunks := chunkAt(text, d.Delivery.ChunkSize)
	for i, c := range chunks {
		opts := ircout.SendOpts{
			Network: d.Event.Network,
			Target:  deliveryTarget(d),
			Text:    c,
		}
		if i == 0 {
			if e.cfg.IRC.HasReplyCAP(d.Event.Network) && d.Delivery.ReplyToID != "" {
				opts.ReplyTo = d.Delivery.ReplyToID
			} else if d.Delivery.NickPrefixOnFallback && d.Event.Channel != "" {
				opts.NickPrefix = d.Event.Nick + ": "
			}
		}
		if err := e.cfg.IRC.Send(ctx, opts); err != nil {
			return err
		}
	}
	return nil
}

// cacheCounter wraps a Client to record the most recent CachedTokens.
type cacheCounter struct {
	llmcore.Client
	last *int
}

func (c *cacheCounter) Complete(ctx context.Context, messages []llmcore.Message, tools []string, model, cacheName string) (llmcore.Completion, error) {
	out, err := c.Client.Complete(ctx, messages, tools, model, cacheName)
	if err == nil {
		*c.last = out.CachedTokens
	}
	return out, err
}
```

- [ ] **Step 4: Run to pass**

Run: `go test ./exec/...`
Expected: PASS for all executor tests.

- [ ] **Step 5: Commit**

```bash
git add go/exec/executor.go go/exec/executor_test.go
git commit -m "feat(go): add exec.Executor.Run end-to-end wiring"
```

---

## Phase 7 — Demo

### Task 23: `cmd/routerdemo/main.go` — wires everything

**Files:**
- Create: `go/cmd/routerdemo/main.go`

- [ ] **Step 1: Write the demo**

```go
// Command routerdemo wires the routing/intent layer with in-memory fakes
// and runs one chat turn end-to-end. Useful for manual smoke and for
// confirming the slice is internally consistent.
package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"time"

	"github.com/rdrake/vibebot-v8/v9/exec"
	ircfake "github.com/rdrake/vibebot-v8/v9/ircout/fake"
	"github.com/rdrake/vibebot-v8/v9/llmcore"
	llmfake "github.com/rdrake/vibebot-v8/v9/llmcore/fake"
	"github.com/rdrake/vibebot-v8/v9/overlay"
	overlayfake "github.com/rdrake/vibebot-v8/v9/overlay/fake"
	persistfake "github.com/rdrake/vibebot-v8/v9/persist/fake"
	"github.com/rdrake/vibebot-v8/v9/router"
	"github.com/rdrake/vibebot-v8/v9/router/profile"
	toolfake "github.com/rdrake/vibebot-v8/v9/tooling/fake"
)

func main() {
	log := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))

	reg := profile.NewRegistry()
	profile.RegisterBuiltins(reg)

	llm := llmfake.New()
	llm.CacheName = "fake-cache-1"
	llm.Script = []llmcore.Completion{{Text: "hi alice, how can i help?", CachedTokens: 0}}

	td := toolfake.New()
	for _, n := range []string{"clear_instruction", "delete_memory", "fetch_url", "generate_image", "get_instruction", "list_memories", "save_memory", "search_web", "set_instruction", "update_memory"} {
		td.Schemas[n] = []byte(`{"name":"` + n + `"}`)
	}

	or := overlayfake.New()
	or.Texts[overlay.Scope{Network: "afternet", Channel: "#demo", Profile: "chat", OverlayHash: ""}] = "be friendly"

	st := persistfake.New()
	sender := ircfake.New()

	e := exec.New(exec.Config{
		LLM: llm, Tools: td, Overlay: or, Store: st, IRC: sender, Log: log, Registry: reg,
	})

	evt := router.IRCEvent{
		Network: "afternet", Channel: "#demo", Nick: "alice",
		Text: "vibebot: hi", MessageID: "m-1", ReceivedAt: time.Now(),
	}
	state := router.ChannelState{Profile: "chat", Overlay: "be friendly"}
	bot := router.BotState{SelfNick: "vibebot", Now: time.Now()}

	d := router.Route(evt, state, bot, reg)
	fmt.Printf("Decision: action=%s profile=%s tools=%d model=%s\n", d.Action, d.Profile, len(d.Tools), d.Model)

	if err := e.Run(context.Background(), d); err != nil {
		log.Error("run failed", "err", err)
		os.Exit(1)
	}

	fmt.Printf("Sent %d msg(s); first: %q\n", len(sender.Sent), sender.Sent[0].Text)
	fmt.Printf("Typing events: %d\n", len(sender.TypingStates))
	fmt.Printf("Usage rows: %d\n", len(st.Usage))
}
```

- [ ] **Step 2: Build the demo**

Run from `go/`: `go build ./cmd/routerdemo/`
Expected: success.

- [ ] **Step 3: Run the demo**

Run from `go/`: `go run ./cmd/routerdemo/`
Expected output (approximately):
```
Decision: action=RespondChat profile=chat tools=10 model=gemini-2.5-flash
Sent 1 msg(s); first: "hi alice, how can i help?"
Typing events: 2
Usage rows: 1
```

- [ ] **Step 4: Commit**

```bash
git add go/cmd/routerdemo/main.go
git commit -m "feat(go): add routerdemo wiring sample for manual smoke"
```

---

### Task 24: Full test sweep + plan checkpoint

- [ ] **Step 1: Run all Go tests**

Run from `go/`: `make test`
Expected: ALL packages PASS, race detector clean, no flakes.

- [ ] **Step 2: Run vet**

Run from `go/`: `make vet`
Expected: no warnings.

- [ ] **Step 3: Run format check**

Run from `go/`: `gofmt -l .`
Expected: empty output (everything formatted).

- [ ] **Step 4: Update repo-level docs**

Append the following line to `/Users/rdrake/workspace/afternet/vibebot-v8/AGENTS.md` under a new "Go rewrite (v9)" section (or create the section if absent):

```markdown
## Go rewrite (v9)

The Go rewrite lives in `go/`. Sub-project E (routing/intent layer) is the first slice — see `docs/superpowers/plans/2026-05-11-routing-intent-layer.md` for the implementation plan and `docs/superpowers/specs/2026-05-11-routing-intent-layer-design.md` for the design. All other sub-projects (A IRC, B LLM, C tools, D overlay, F persistence, G deploy) ship as separate specs/plans.

Build/test: `cd go && make all`.
```

- [ ] **Step 5: Commit**

```bash
git add AGENTS.md
git commit -m "docs: note Go rewrite v9 sub-project E landed"
```

---

## Out of scope for this plan (deferred to sibling sub-projects)

- **Sub-project A** — Real IRC client (ergochat/irc-go), CAP negotiation, SASL, multi-network, reconnect. Includes the **per-channel turn-lock dispatcher** that takes the `LastAmbientAt` snapshot, calls `router.Route`, atomically commits the ambient timestamp, and invokes `Executor.Run`. The spec's atomicity requirement lives at the dispatch boundary, which is in A.
- **Sub-project B** — Real LLM client (openai-go + Gemini OpenAI-compat for the hot path; `google.golang.org/genai` for `CachedContent` lifecycle). Provider abstraction for tool-call gaps. Emits typed errors (`ErrLLMTransient`, `ErrLLMFatal`, `ErrCacheStale`, `ErrBudgetExceeded`) that the executor can categorize for recovery.
- **Sub-project C** — Real tool implementations (memory, instruction, search, fetch, draw, etc.) and rate-bucket dispatcher with per-tool timeout enforcement.
- **Sub-project D** — Real overlay resolver (channel `assistantSystemPrompt`, scene/loom overlay layering). Includes the property test asserting `Get(scope)` is byte-stable across N calls and N goroutines.
- **Sub-project F** — Real persistence (SQLite or similar) for history, memories, usage.
- **Sub-project G** — Single-binary build, config loader, logging sink, optional Prometheus metrics, deploy scripts.
- **Error-category recovery in executor** — The current `Executor.Run` returns raw wrapped errors. Once B emits typed `ErrLLMTransient` / `ErrLLMFatal` / `ErrCacheStale`, a follow-up task adds: retry-with-backoff for transient (3 tries, 1s/3s/9s), inline cache rebuild for stale, user-visible apology for fatal. Deferred because it can't be meaningfully tested against the current fake (fake doesn't emit categorized errors).
- **Tier 3 integration** (real ergochat against local Ergo IRCd) — added in sub-project A's plan.
- **Tier 4 live-Gemini smoke** — added in sub-project B's plan.
- **Admin `@cmd` dispatcher** — separate spec ("admin commands").
- **Mid-message mention detection** — deferred to a future "natural addressed detection" iteration per spec.

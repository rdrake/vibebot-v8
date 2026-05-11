// Package llmcore defines the LLM client contract consumed by the executor.
// The real implementation lives in sub-project B.
package llmcore

import (
	"bytes"
	"context"
	"time"
)

// CachedPrefix is the deterministic content that goes into the LLM cache.
// The four fields are serialized by Canonical into a single byte block.
// E populates this struct; B passes Canonical(cp) to the provider.
//
// Schema canonicalization is sub-project C's responsibility: the bytes in
// ToolSchemasJSON MUST already be canonical (sorted keys, no insignificant
// whitespace). E does not re-canonicalize.
type CachedPrefix struct {
	FrameworkPrompt string
	Overlay         string
	ToolSchemasJSON []byte
	ChannelContext  string
}

// Canonical returns the byte representation of cp that the cache key
// must agree with byte-for-byte. The exact ordering and `\n\n---\n\n`
// delimiter are part of the contract with sub-project B.
func Canonical(cp CachedPrefix) []byte {
	const sep = "\n\n---\n\n"
	var buf bytes.Buffer
	buf.Grow(len(cp.FrameworkPrompt) + len(cp.Overlay) + len(cp.ToolSchemasJSON) + len(cp.ChannelContext) + 3*len(sep))
	buf.WriteString(cp.FrameworkPrompt)
	buf.WriteString(sep)
	buf.WriteString(cp.Overlay)
	buf.WriteString(sep)
	buf.Write(cp.ToolSchemasJSON)
	buf.WriteString(sep)
	buf.WriteString(cp.ChannelContext)
	return buf.Bytes()
}

// ToolSchema is one tool's canonical schema, ready for both cache prefix
// composition and Complete calls. Pre-resolved by exec.ResolveSchemas so
// B never needs a back-channel to C.
type ToolSchema struct {
	Name       string
	SchemaJSON []byte
}

// CacheHandle is the opaque reference to a CachedContent (or equivalent).
// Zero value means uncached path (e.g. xAI, or provider without cache).
type CacheHandle struct {
	Name      string
	Provider  string    // "gemini", "xai", ""
	ExpiresAt time.Time // zero when no cache
}

// IsCached reports whether the handle refers to a live cache entry.
func (h CacheHandle) IsCached() bool { return h.Name != "" }

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
	CachedTokens int
}

// Scope identifies a cache lane. Mirror of router.CacheScope; lives here
// to avoid import cycles (llmcore must not import router).
type Scope struct {
	Network     string
	Channel     string
	Profile     string
	OverlayHash string
}

// Client is the contract.
type Client interface {
	// EnsureCache creates or refreshes a CachedContent for scope+prefix.
	// Returns a CacheHandle; zero value indicates no cache (provider does
	// not support, or scope already has live coverage cheaper than refresh).
	EnsureCache(ctx context.Context, scope Scope, prefix CachedPrefix) (CacheHandle, error)

	// Complete runs one LLM completion. The prefix is passed every call so
	// B can choose the right mode:
	//   - if !cache.IsCached(): B MUST inline Canonical(prefix) ahead of
	//     messages (uncached path; e.g. xAI, or Gemini with no live cache).
	//   - if cache.IsCached(): B MUST reference cache by handle and ignore
	//     the prefix bytes (they were already cached when EnsureCache ran).
	// E never branches on provider; B is the only layer that knows mode.
	Complete(ctx context.Context, prefix CachedPrefix, messages []Message, tools []ToolSchema, model string, cache CacheHandle) (Completion, error)
}

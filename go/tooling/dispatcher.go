// Package tooling defines the tool-dispatch contract consumed by the executor.
// The real implementation lives in sub-project C.
package tooling

import "context"

// ToolCall is what the LLM asked the bot to run.
type ToolCall struct {
	ID        string
	Name      string
	Arguments string
}

// ToolResult is what to feed back to the LLM.
type ToolResult struct {
	Content string
	Err     error
	Denied  bool // true if profile/rate disallowed; Err must also be non-nil
}

// Dispatcher resolves a tool name to a handler.
type Dispatcher interface {
	// SchemaJSON returns the canonical JSON schema for a tool name.
	// MUST be byte-identical across calls (sorted keys, no insignificant
	// whitespace). Used by exec.ResolveSchemas to populate llmcore.ToolSchema.
	SchemaJSON(name string) ([]byte, error)

	// Dispatch runs a tool call. Honors ctx cancellation. The per-tool
	// timeout is enforced inside Dispatch (sub-project C's responsibility).
	Dispatch(ctx context.Context, call ToolCall) ToolResult
}

// Package persist defines the persistence contract consumed by the executor.
// The real implementation lives in sub-project F.
package persist

import (
	"context"
	"time"
)

type HistoryEntry struct {
	Role      string // "user" or "assistant"
	Nick      string
	Content   string
	Timestamp time.Time
}

type Memory struct {
	ID      int64
	Nick    string
	Fact    string
	Channel string
}

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
	Status           string
	ErrorDetail      string
}

type Store interface {
	History(ctx context.Context, network, channel string, limit int) ([]HistoryEntry, error)
	Memories(ctx context.Context, network, channel, nick string) ([]Memory, error)
	AppendTurn(ctx context.Context, network, channel, nick, assistantText string) error
	RecordUsage(ctx context.Context, row UsageRow) error
}

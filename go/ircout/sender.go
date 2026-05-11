// Package ircout defines the IRC output contract consumed by the executor.
// The real implementation lives in sub-project A.
package ircout

import "context"

type SendOpts struct {
	Network    string
	Target     string
	Text       string
	ReplyTo    string // IRCv3 +draft/reply target message-id; "" to omit
	NickPrefix string // fallback prefix when ReplyTo CAP unavailable; "" to omit
}

type TypingState string

const (
	TypingActive TypingState = "active"
	TypingDone   TypingState = "done"
)

type Sender interface {
	Send(ctx context.Context, opts SendOpts) error

	// SendTyping is best-effort: no error returned, failures logged at sink.
	SendTyping(ctx context.Context, network, target string, state TypingState)

	// HasReplyCAP reports whether IRCv3 draft/reply was negotiated.
	HasReplyCAP(network string) bool
}

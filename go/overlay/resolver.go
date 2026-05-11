// Package overlay defines the persona/overlay resolution contract consumed
// by the executor. The real implementation lives in sub-project D.
package overlay

import "context"

// Scope identifies an overlay lookup. NO OverlayHash — the hash is derived
// FROM the returned text, computed by the caller. Including it here would
// be circular.
type Scope struct {
	Network string
	Channel string
	Profile string
}

// Resolver returns the overlay text for a scope. MUST be a pure function:
// same Scope → byte-identical text, across goroutines and calls.
//
// MAY NOT inject user-specific text (nick, account, timestamps). User-
// specific layering happens in the uncached tail.
type Resolver interface {
	Get(ctx context.Context, scope Scope) (text string, err error)
}

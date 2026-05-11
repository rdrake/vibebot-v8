package exec

import "errors"

var (
	ErrLLMTransient   = errors.New("exec: llm transient failure")
	ErrLLMFatal       = errors.New("exec: llm fatal failure")
	ErrToolDenied     = errors.New("exec: tool denied")
	ErrToolFailed     = errors.New("exec: tool execution failed")
	ErrIRCSend        = errors.New("exec: irc send failed")
	ErrCacheStale     = errors.New("exec: cache stale")
	ErrBudgetExceeded = errors.New("exec: budget exceeded")
)

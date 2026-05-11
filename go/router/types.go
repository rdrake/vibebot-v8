package router

import "time"

type IRCEvent struct {
	Network    string
	Channel    string
	Nick       string
	Account    string
	Text       string
	IsAction   bool
	Tags       map[string]string
	MessageID  string
	ReceivedAt time.Time
}

type SceneRef struct{ ID string }
type LoomRef struct{ ID string }

type ChannelKey struct {
	Network string
	Channel string
}

// ChannelState is hydrated by the dispatcher BEFORE Route is called. The
// Overlay field is the pre-resolved overlay text from D's Resolver.Get —
// Route hashes it to build CacheScope.OverlayHash. Executor will re-resolve
// via D for determinism, but D's purity contract guarantees the same text.
type ChannelState struct {
	Profile         string
	Overlay         string
	AmbientEnabled  bool
	AmbientCooldown time.Duration
	SceneActive     *SceneRef
	LoomActive      *LoomRef
}

type BotState struct {
	SelfNick      string
	LastAmbientAt map[ChannelKey]time.Time
	Now           time.Time
	RecentSentIDs []string
}

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

type CacheScope struct {
	Network     string
	Channel     string
	Profile     string
	OverlayHash string
}

type PromptSpec struct {
	HistoryLimit int
	UserText     string
	UserNick     string
}

type DeliverySpec struct {
	Typing               bool
	ChunkSize            int
	ReplyToID            string
	NickPrefixOnFallback bool
	MaxIters             int
}

type RouteDecision struct {
	Action     Action
	Profile    string
	Tools      []string
	CacheScope CacheScope
	Model      string
	Prompt     PromptSpec
	Delivery   DeliverySpec
	Event      IRCEvent
}

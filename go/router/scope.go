package router

import (
	"crypto/sha256"
	"encoding/hex"
)

// overlayHash is hex sha256 truncated to 16 bytes (32 hex chars).
// Same input → same hash, byte-identical.
func overlayHash(text string) string {
	sum := sha256.Sum256([]byte(text))
	return hex.EncodeToString(sum[:16])
}

func buildCacheScope(network, channel, profile, overlay string) CacheScope {
	return CacheScope{
		Network:     network,
		Channel:     channel,
		Profile:     profile,
		OverlayHash: overlayHash(overlay),
	}
}

package router

import "testing"

func TestOverlayHashStable(t *testing.T) {
	a := overlayHash("hello world")
	b := overlayHash("hello world")
	if a != b {
		t.Fatalf("not stable")
	}
	if len(a) != 32 {
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
		t.Fatalf("%+v", got)
	}
	if len(got.OverlayHash) != 32 {
		t.Fatalf("hash: %q", got.OverlayHash)
	}
}

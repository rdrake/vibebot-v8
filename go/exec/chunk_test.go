package exec

import "testing"

func TestChunkAtSplits(t *testing.T) {
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
		t.Fatal(got)
	}
}

func TestChunkAtZeroSizeReturnsSingle(t *testing.T) {
	got := chunkAt("hello", 0)
	if len(got) != 1 || got[0] != "hello" {
		t.Fatal(got)
	}
}

func TestChunkAtExactBoundary(t *testing.T) {
	got := chunkAt("hello", 5)
	if len(got) != 1 || got[0] != "hello" {
		t.Fatalf("got %v", got)
	}
}

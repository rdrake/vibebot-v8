package router

import "testing"

func TestAddressedDM(t *testing.T) {
	if !addressed(IRCEvent{Channel: "", Text: "anything"}, "vibebot", nil) {
		t.Fatal("DM should be addressed")
	}
}

func TestAddressedFirstTokenMatch(t *testing.T) {
	cases := []struct {
		text string
		want bool
	}{
		{"vibebot: hello", true},
		{"vibebot, hi", true},
		{"vibebot; what?", true},
		{"VIBEBOT hi", true},
		{"  vibebot   hi", true},
		{"vibebot:\nhello", true},     // newline whitespace
		{"vibebot:\thi", true},        // tab
		{"hi vibebot", false},         // mid-message — out of scope for v1
		{"vibebotsomething hi", false},
		{"", false},
	}
	for _, c := range cases {
		got := addressed(IRCEvent{Channel: "#x", Text: c.text}, "vibebot", nil)
		if got != c.want {
			t.Errorf("addressed(%q) = %v want %v", c.text, got, c.want)
		}
	}
}

func TestAddressedReplyTag(t *testing.T) {
	recent := []string{"msg-1", "msg-2"}
	evt := IRCEvent{Channel: "#x", Text: "yes", Tags: map[string]string{"+draft/reply": "msg-1"}}
	if !addressed(evt, "vibebot", recent) {
		t.Fatal("reply to recent bot msg should be addressed")
	}
	evt.Tags["+draft/reply"] = "unknown-id"
	if addressed(evt, "vibebot", recent) {
		t.Fatal("reply to non-bot msg should not be addressed")
	}
}

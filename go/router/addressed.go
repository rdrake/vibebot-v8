package router

import "strings"

// addressed reports whether evt is addressed to selfNick.
//
// True iff ANY of:
//   - evt.Channel == "" (DM)
//   - evt.Tags["+draft/reply"] is in recentSentIDs
//   - first whitespace-delimited token of evt.Text, lowercased and stripped
//     of trailing punctuation in [:,;.!?], equals strings.ToLower(selfNick)
//
// Whitespace is any unicode whitespace (via strings.Fields).
// Mid-message mention is intentionally NOT addressed for v1.
func addressed(evt IRCEvent, selfNick string, recentSentIDs []string) bool {
	if evt.Channel == "" {
		return true
	}
	if replyID := evt.Tags["+draft/reply"]; replyID != "" {
		for _, id := range recentSentIDs {
			if id == replyID {
				return true
			}
		}
	}
	fields := strings.Fields(evt.Text)
	if len(fields) == 0 {
		return false
	}
	return strings.EqualFold(stripTrailingPunct(fields[0]), selfNick)
}

func stripTrailingPunct(s string) string {
	return strings.TrimRight(s, ":,;.!?")
}

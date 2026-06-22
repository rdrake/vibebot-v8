from llm.verse.taste_mine import Msg, iter_messages


def test_iter_messages_parses_privmsg_action_skips_rest():
    lines = [
        "2026-06-22T00:01:16  <fc42> a normal message",
        "2026-06-22T00:02:00  * fc42 does a thing",  # CTCP ACTION / /me
        "2026-06-22T00:03:00  *** vibebot has joined #afternet",  # system -> skip
        "2026-06-22T00:04:00  -ChanServ- a notice",  # notice -> skip
        "2026-06-22T00:05:00  <fc42> ",  # empty body -> skip
        "garbage no double-space sep",  # malformed -> skip
        "2026-06-22T00:06:00  <rdrake> hi th�ere",  # garbled char -> kept
    ]
    assert list(iter_messages(lines)) == [
        Msg("fc42", "a normal message"),
        Msg("fc42", "does a thing"),
        Msg("rdrake", "hi th�ere"),
    ]

"""Tests for ConversationContext."""

from __future__ import annotations

import threading
import time

from llm.context import ContextConfig, ConversationContext
from llm.persistence import LLMDatabase


class TestConversationContext:
    """Test conversation context functionality."""

    def test_context_add_and_get_messages(self) -> None:
        """GIVEN context WHEN add messages THEN get returns them."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_message("user1", "#channel", "user", "Hello")
        ctx.add_message("user1", "#channel", "assistant", "Hi there!")

        messages = ctx.get_messages("user1", "#channel")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi there!"

    def test_context_per_user_isolation(self) -> None:
        """GIVEN context WHEN different users THEN contexts are isolated."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_message("user1", "#channel", "user", "Hello from user1")
        ctx.add_message("user2", "#channel", "user", "Hello from user2")

        messages1 = ctx.get_messages("user1", "#channel")
        messages2 = ctx.get_messages("user2", "#channel")

        assert len(messages1) == 1
        assert messages1[0]["content"] == "Hello from user1"
        assert len(messages2) == 1
        assert messages2[0]["content"] == "Hello from user2"

    def test_context_per_channel_isolation(self) -> None:
        """GIVEN context WHEN same user different channels THEN contexts are isolated."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_message("user1", "#channel1", "user", "Hello in channel1")
        ctx.add_message("user1", "#channel2", "user", "Hello in channel2")

        messages1 = ctx.get_messages("user1", "#channel1")
        messages2 = ctx.get_messages("user1", "#channel2")

        assert len(messages1) == 1
        assert messages1[0]["content"] == "Hello in channel1"
        assert len(messages2) == 1
        assert messages2[0]["content"] == "Hello in channel2"

    def test_context_case_insensitive(self) -> None:
        """GIVEN context WHEN different case nick/channel THEN same context."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_message("User1", "#Channel", "user", "Hello")
        ctx.add_message("user1", "#channel", "user", "World")

        messages = ctx.get_messages("USER1", "#CHANNEL")
        assert len(messages) == 2

    def test_context_max_messages_limit(self) -> None:
        """GIVEN context with max_messages WHEN exceed limit THEN oldest removed."""
        config = ContextConfig(max_messages=4, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        # Add 6 messages (exceeds limit of 4)
        for i in range(6):
            ctx.add_message("user1", "#channel", "user", f"Message {i}")

        messages = ctx.get_messages("user1", "#channel")
        assert len(messages) == 4
        # Should have messages 2-5, not 0-3
        assert messages[0]["content"] == "Message 2"
        assert messages[3]["content"] == "Message 5"

    def test_context_time_expiry(self) -> None:
        """GIVEN context WHEN timeout expires THEN context cleared."""
        config = ContextConfig(max_messages=20, timeout_minutes=1, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_message("user1", "#channel", "user", "Hello")

        # Backdate last_activity so it appears expired
        key = ("user1", "#channel")
        ctx._conversations[key].last_activity -= 120

        messages = ctx.get_messages("user1", "#channel")
        assert len(messages) == 0

    def test_context_clear_specific_user(self) -> None:
        """GIVEN context WHEN clear specific user THEN only that context cleared."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_message("user1", "#channel", "user", "Hello from user1")
        ctx.add_message("user2", "#channel", "user", "Hello from user2")

        cleared = ctx.clear("user1", "#channel")
        assert cleared is True

        # User1 context should be cleared
        assert len(ctx.get_messages("user1", "#channel")) == 0
        # User2 context should remain
        assert len(ctx.get_messages("user2", "#channel")) == 1

    def test_context_clear_nonexistent(self) -> None:
        """GIVEN context WHEN clear nonexistent user THEN returns False."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        cleared = ctx.clear("nonexistent", "#channel")
        assert cleared is False

    def test_context_clear_all(self) -> None:
        """GIVEN context WHEN clear all THEN all contexts cleared."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_message("user1", "#channel1", "user", "Hello")
        ctx.add_message("user2", "#channel2", "user", "World")

        count = ctx.clear_all()
        assert count == 2

        assert len(ctx.get_messages("user1", "#channel1")) == 0
        assert len(ctx.get_messages("user2", "#channel2")) == 0

    def test_context_disabled(self) -> None:
        """GIVEN context disabled WHEN operations THEN no storage."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=False)
        ctx = ConversationContext(config)

        ctx.add_message("user1", "#channel", "user", "Hello")
        messages = ctx.get_messages("user1", "#channel")

        assert len(messages) == 0

    def test_context_stats(self) -> None:
        """GIVEN context WHEN get stats THEN accurate counts returned."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_message("user1", "#channel", "user", "Hello")
        ctx.add_message("user1", "#channel", "assistant", "Hi")
        ctx.add_message("user2", "#channel", "user", "Hey")

        stats = ctx.get_stats()
        assert stats["active_conversations"] == 2
        assert stats["total_messages"] == 3
        assert stats["max_messages_per_conv"] == 20
        assert stats["timeout_minutes"] == 30
        assert stats["enabled"] is True

    def test_user_stats_with_messages(self) -> None:
        """GIVEN user with messages WHEN get_user_stats THEN returns count and expiry."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_message("user1", "#channel", "user", "Hello")
        ctx.add_message("user1", "#channel", "assistant", "Hi")

        stats = ctx.get_user_stats("user1", "#channel")
        assert stats["message_count"] == 2
        assert stats["max_messages"] == 20
        assert stats["seconds_until_expiry"] > 0
        assert stats["enabled"] is True

    def test_user_stats_empty(self) -> None:
        """GIVEN user with no messages WHEN get_user_stats THEN returns zero count."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        stats = ctx.get_user_stats("user1", "#channel")
        assert stats["message_count"] == 0
        assert stats["max_messages"] == 20
        assert stats["seconds_until_expiry"] == 0
        assert stats["enabled"] is True

    def test_user_stats_disabled(self) -> None:
        """GIVEN context disabled WHEN get_user_stats THEN returns disabled."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=False)
        ctx = ConversationContext(config)

        stats = ctx.get_user_stats("user1", "#channel")
        assert stats["message_count"] == 0
        assert stats["enabled"] is False

    def test_context_thread_safe(self) -> None:
        """GIVEN context WHEN concurrent operations THEN thread-safe."""
        config = ContextConfig(max_messages=1000, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        errors: list[Exception] = []
        lock = threading.Lock()

        def add_messages(user_id: str) -> None:
            try:
                for i in range(100):
                    ctx.add_message(user_id, "#channel", "user", f"Message {i}")
                    ctx.get_messages(user_id, "#channel")
            except Exception as e:
                with lock:
                    errors.append(e)

        # Run 10 threads, each adding 100 messages
        threads = []
        for i in range(10):
            t = threading.Thread(target=add_messages, args=(f"user{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should complete without errors
        assert len(errors) == 0

        # Each user should have their messages
        stats = ctx.get_stats()
        assert stats["active_conversations"] == 10


class TestChannelContext:
    """Test shared channel context functionality."""

    def test_channel_context_add_and_get(self) -> None:
        """GIVEN channel context WHEN add messages THEN get returns them."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_channel_message("#channel", "alice", "user", "Hello everyone")
        ctx.add_channel_message("#channel", "bot", "assistant", "Hi Alice!")

        messages = ctx.get_channel_messages("#channel")
        assert len(messages) == 2
        assert messages[0]["nick"] == "alice"
        assert messages[0]["content"] == "Hello everyone"
        assert messages[1]["nick"] == "bot"
        assert messages[1]["content"] == "Hi Alice!"

    def test_channel_context_shared_across_users(self) -> None:
        """GIVEN channel context WHEN different users THEN same context."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_channel_message("#channel", "alice", "user", "What's 2+2?")
        ctx.add_channel_message("#channel", "bot", "assistant", "4")

        # Both alice and bob should see the same channel context
        alice_view = ctx.get_channel_messages("#channel")
        bob_view = ctx.get_channel_messages("#channel")

        assert len(alice_view) == 2
        assert len(bob_view) == 2
        assert alice_view == bob_view

    def test_channel_context_exclude_nick(self) -> None:
        """GIVEN channel context WHEN exclude nick THEN filters correctly."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_channel_message("#channel", "alice", "user", "Question from Alice")
        ctx.add_channel_message("#channel", "bot", "assistant", "Answer to Alice")
        ctx.add_channel_message("#channel", "bob", "user", "Question from Bob")
        ctx.add_channel_message("#channel", "bot", "assistant", "Answer to Bob")

        # Bob's view should exclude his own messages
        bob_view = ctx.get_channel_messages("#channel", exclude_nick="bob")

        assert len(bob_view) == 3
        nicks = [m["nick"] for m in bob_view]
        assert "bob" not in nicks
        assert "alice" in nicks
        assert "bot" in nicks

    def test_channel_context_exclude_nick_case_insensitive(self) -> None:
        """GIVEN channel context WHEN exclude nick with different case THEN filters."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_channel_message("#channel", "Alice", "user", "From Alice")
        ctx.add_channel_message("#channel", "bob", "user", "From Bob")

        # Exclude with different case
        view = ctx.get_channel_messages("#channel", exclude_nick="ALICE")

        assert len(view) == 1
        assert view[0]["nick"] == "bob"

    def test_channel_context_max_messages_limit(self) -> None:
        """GIVEN channel context with limit WHEN exceed THEN oldest removed."""
        config = ContextConfig(
            max_messages=20, timeout_minutes=30, enabled=True, channel_max_messages=3
        )
        ctx = ConversationContext(config)

        # Add 5 messages (exceeds limit of 3)
        for i in range(5):
            ctx.add_channel_message("#channel", f"user{i}", "user", f"Message {i}")

        messages = ctx.get_channel_messages("#channel")
        assert len(messages) == 3
        # Should have messages 2-4, not 0-1
        assert messages[0]["content"] == "Message 2"
        assert messages[2]["content"] == "Message 4"

    def test_channel_context_isolated_by_channel(self) -> None:
        """GIVEN channel context WHEN different channels THEN isolated."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_channel_message("#channel1", "alice", "user", "In channel 1")
        ctx.add_channel_message("#channel2", "alice", "user", "In channel 2")

        ch1_messages = ctx.get_channel_messages("#channel1")
        ch2_messages = ctx.get_channel_messages("#channel2")

        assert len(ch1_messages) == 1
        assert ch1_messages[0]["content"] == "In channel 1"
        assert len(ch2_messages) == 1
        assert ch2_messages[0]["content"] == "In channel 2"

    def test_channel_context_expiry(self) -> None:
        """GIVEN channel context WHEN timeout expires THEN cleared."""
        config = ContextConfig(
            max_messages=20,
            timeout_minutes=1,
            enabled=True,
        )
        ctx = ConversationContext(config)

        ctx.add_channel_message("#channel", "alice", "user", "Hello")

        # Backdate last_activity so it appears expired
        ctx._channel_contexts["#channel"].last_activity -= 120

        messages = ctx.get_channel_messages("#channel")
        assert len(messages) == 0

    def test_channel_context_disabled(self) -> None:
        """GIVEN context disabled WHEN channel operations THEN no storage."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=False)
        ctx = ConversationContext(config)

        ctx.add_channel_message("#channel", "alice", "user", "Hello")
        messages = ctx.get_channel_messages("#channel")

        assert len(messages) == 0

    def test_channel_context_in_stats(self) -> None:
        """GIVEN channel context WHEN get stats THEN includes channel stats."""
        config = ContextConfig(
            max_messages=20, timeout_minutes=30, enabled=True, channel_max_messages=10
        )
        ctx = ConversationContext(config)

        # Add personal context
        ctx.add_message("user1", "#channel", "user", "Personal message")

        # Add channel context
        ctx.add_channel_message("#channel", "alice", "user", "Channel msg 1")
        ctx.add_channel_message("#channel", "bob", "user", "Channel msg 2")

        stats = ctx.get_stats()
        assert stats["active_channels"] == 1
        assert stats["channel_messages"] == 2
        assert stats["channel_max_messages"] == 10

    def test_channel_context_clear_all(self) -> None:
        """GIVEN channel context WHEN clear all THEN channel context also cleared."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_message("user1", "#channel", "user", "Personal")
        ctx.add_channel_message("#channel", "alice", "user", "Channel")

        ctx.clear_all()

        assert len(ctx.get_messages("user1", "#channel")) == 0
        assert len(ctx.get_channel_messages("#channel")) == 0

    def test_channel_context_returns_copies(self) -> None:
        """GIVEN channel context WHEN get messages THEN returns copies."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_channel_message("#channel", "alice", "user", "Hello")

        messages1 = ctx.get_channel_messages("#channel")
        messages2 = ctx.get_channel_messages("#channel")

        # Should be different list objects
        assert messages1 is not messages2
        # But with same content
        assert messages1 == messages2

        # Modifying one should not affect the other or the stored data
        messages1[0]["content"] = "Modified"
        messages3 = ctx.get_channel_messages("#channel")
        assert messages3[0]["content"] == "Hello"


class TestPersistentContext:
    """Test conversation context with SQLite persistence."""

    def _make_ctx(self, db: LLMDatabase) -> tuple[ConversationContext, LLMDatabase]:
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config, db=db)
        return ctx, db

    def test_add_message_persists_to_db(self, test_db: LLMDatabase) -> None:
        """GIVEN context with db WHEN add_message THEN conversation is in SQLite."""
        ctx, db = self._make_ctx(test_db)
        ctx.add_message("user1", "#chan", "user", "Hello")

        loaded = db.load_conversations()
        assert len(loaded) == 1
        assert loaded[0][2] == [{"role": "user", "content": "Hello"}]

    def test_add_message_persist_false_skips_db(self, test_db: LLMDatabase) -> None:
        """GIVEN context with db WHEN add_message(persist=False) THEN not in SQLite."""
        ctx, db = self._make_ctx(test_db)
        ctx.add_message("user1", "#chan", "user", "Hello", persist=False)

        loaded = db.load_conversations()
        assert len(loaded) == 0

        # But still in memory
        msgs = ctx.get_messages("user1", "#chan")
        assert len(msgs) == 1

    def test_clear_deletes_from_db(self, test_db: LLMDatabase) -> None:
        """GIVEN persisted conversation WHEN clear THEN removed from SQLite."""
        ctx, db = self._make_ctx(test_db)
        ctx.add_message("user1", "#chan", "user", "Hello")
        ctx.clear("user1", "#chan")

        assert len(db.load_conversations()) == 0

    def test_clear_all_deletes_from_db(self, test_db: LLMDatabase) -> None:
        """GIVEN persisted conversations WHEN clear_all THEN all removed from SQLite."""
        ctx, db = self._make_ctx(test_db)
        ctx.add_message("user1", "#chan", "user", "Hello")
        ctx.add_message("user2", "#chan", "user", "Hi")
        ctx.clear_all()

        assert len(db.load_conversations()) == 0

    def test_startup_loads_from_db(self, test_db: LLMDatabase) -> None:
        """GIVEN conversations in db WHEN new ConversationContext THEN loaded into memory."""
        test_db.save_conversation(
            "user1",
            "#chan",
            [{"role": "user", "content": "Hello"}],
            time.time(),
        )

        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config, db=test_db)

        msgs = ctx.get_messages("user1", "#chan")
        assert len(msgs) == 1
        assert msgs[0]["content"] == "Hello"

    def test_startup_skips_expired(self, test_db: LLMDatabase) -> None:
        """GIVEN expired conversation in db WHEN new ConversationContext THEN not loaded."""
        old_time = time.time() - 7200  # 2 hours ago
        test_db.save_conversation(
            "user1",
            "#chan",
            [{"role": "user", "content": "Hello"}],
            old_time,
        )

        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config, db=test_db)

        msgs = ctx.get_messages("user1", "#chan")
        assert len(msgs) == 0

    def test_without_db_works_unchanged(self) -> None:
        """GIVEN context without db WHEN operations THEN works as before."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_message("user1", "#chan", "user", "Hello")
        msgs = ctx.get_messages("user1", "#chan")
        assert len(msgs) == 1
        ctx.clear("user1", "#chan")
        assert ctx.get_messages("user1", "#chan") == []

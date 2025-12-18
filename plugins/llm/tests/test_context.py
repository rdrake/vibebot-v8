"""Tests for ConversationContext."""

from __future__ import annotations

import threading
import time

from llm.context import ContextConfig, ConversationContext


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
        # Use very short timeout for testing
        config = ContextConfig(max_messages=20, timeout_minutes=0, enabled=True)
        # timeout_minutes=0 means immediate expiry (0 seconds)
        ctx = ConversationContext(config)

        ctx.add_message("user1", "#channel", "user", "Hello")

        # Small delay to ensure expiry
        time.sleep(0.1)

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

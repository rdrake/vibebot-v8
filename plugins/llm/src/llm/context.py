"""Conversation context management for LLM commands."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


class Role:
    """Message role constants for LLM conversations."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ContextConfig:
    """Configuration for conversation context."""

    max_messages: int  # Maximum messages to keep per conversation
    timeout_minutes: int  # Clear context after this many minutes of inactivity
    enabled: bool = True  # Whether context is enabled
    channel_max_messages: int = 10  # Maximum messages in shared channel context


@dataclass
class Conversation:
    """A conversation's state.

    For personal context: messages have {"role": str, "content": str}
    For channel context: messages have {"nick": str, "role": str, "content": str}
    """

    messages: list[dict[str, str]] = field(default_factory=list)
    last_activity: float = field(default_factory=time.time)


class ConversationContext:
    """Thread-safe conversation context manager.

    Stores conversation history per user per channel with automatic expiry.
    """

    def __init__(self, config: ContextConfig) -> None:
        """Initialize context manager.

        Args:
            config: Context configuration
        """
        self.config = config
        self._lock = Lock()
        # Key: (nick, channel) -> Conversation
        self._conversations: dict[tuple[str, str], Conversation] = {}
        # Key: channel -> Conversation (shared across all users)
        self._channel_contexts: dict[str, Conversation] = {}

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return (
            f"ConversationContext(conversations={len(self._conversations)}, "
            f"channels={len(self._channel_contexts)}, enabled={self.config.enabled})"
        )

    def _get_key(self, nick: str, channel: str) -> tuple[str, str]:
        """Generate conversation key.

        Args:
            nick: User's IRC nick
            channel: IRC channel

        Returns:
            Tuple key for the conversation
        """
        return (nick.lower(), channel.lower())

    def _is_expired(self, conversation: Conversation) -> bool:
        """Check if a conversation has expired.

        Args:
            conversation: Conversation to check

        Returns:
            True if expired
        """
        if not self.config.enabled:
            return True
        timeout_seconds = self.config.timeout_minutes * 60
        return time.time() - conversation.last_activity > timeout_seconds

    def _prune_expired(self) -> None:
        """Remove expired conversations. Must be called with lock held."""
        expired_keys = [key for key, conv in self._conversations.items() if self._is_expired(conv)]
        for key in expired_keys:
            del self._conversations[key]

        # Also prune expired channel contexts
        expired_channels = [
            ch for ch, ctx in self._channel_contexts.items() if self._is_expired(ctx)
        ]
        for ch in expired_channels:
            del self._channel_contexts[ch]

    def add_message(self, nick: str, channel: str, role: str, content: str) -> None:
        """Add a message to conversation history.

        Args:
            nick: User's IRC nick
            channel: IRC channel
            role: Message role ("user" or "assistant")
            content: Message content
        """
        if not self.config.enabled:
            return

        with self._lock:
            self._prune_expired()

            key = self._get_key(nick, channel)
            if key not in self._conversations:
                self._conversations[key] = Conversation()

            conv = self._conversations[key]
            conv.messages.append({"role": role, "content": content})
            conv.last_activity = time.time()

            # Trim to max messages (keep pairs to maintain context coherence)
            # Each exchange is 2 messages (user + assistant)
            if len(conv.messages) > self.config.max_messages:
                # Remove oldest messages, keeping most recent
                conv.messages = conv.messages[-self.config.max_messages :]

    def get_messages(self, nick: str, channel: str) -> list[dict[str, str]]:
        """Get conversation history for LiteLLM.

        Args:
            nick: User's IRC nick
            channel: IRC channel

        Returns:
            List of message dicts for LiteLLM
        """
        if not self.config.enabled:
            return []

        with self._lock:
            self._prune_expired()

            key = self._get_key(nick, channel)
            conv = self._conversations.get(key)

            if conv is None or self._is_expired(conv):
                return []

            # Return deep copies to prevent external modification
            return [dict(msg) for msg in conv.messages]

    def add_channel_message(self, channel: str, nick: str, role: str, content: str) -> None:
        """Add a message to shared channel context.

        Channel context is shared across all users in a channel, allowing
        the bot to follow group conversations.

        Args:
            channel: IRC channel
            nick: Nick of who sent the message
            role: Message role ("user" or "assistant")
            content: Message content
        """
        if not self.config.enabled:
            return

        with self._lock:
            self._prune_expired()

            ch_key = channel.lower()
            if ch_key not in self._channel_contexts:
                self._channel_contexts[ch_key] = Conversation()

            ctx = self._channel_contexts[ch_key]
            ctx.messages.append({"nick": nick, "role": role, "content": content})
            ctx.last_activity = time.time()

            # Trim to max messages
            if len(ctx.messages) > self.config.channel_max_messages:
                ctx.messages = ctx.messages[-self.config.channel_max_messages :]

    def get_channel_messages(
        self, channel: str, exclude_nick: str | None = None
    ) -> list[dict[str, str]]:
        """Get shared channel context.

        Args:
            channel: IRC channel
            exclude_nick: Optional nick to exclude (typically the current user,
                since their messages are in personal context)

        Returns:
            List of channel messages with nick, role, and content
        """
        if not self.config.enabled:
            return []

        with self._lock:
            self._prune_expired()

            ch_key = channel.lower()
            ctx = self._channel_contexts.get(ch_key)

            if ctx is None or self._is_expired(ctx):
                return []

            # Filter out excluded nick if specified
            if exclude_nick:
                exclude_lower = exclude_nick.lower()
                return [
                    dict(msg)
                    for msg in ctx.messages
                    if msg.get("nick", "").lower() != exclude_lower
                ]

            return [dict(msg) for msg in ctx.messages]

    def clear(self, nick: str, channel: str) -> bool:
        """Clear conversation context for a specific user.

        Args:
            nick: User's IRC nick
            channel: IRC channel

        Returns:
            True if context was cleared, False if none existed
        """
        with self._lock:
            key = self._get_key(nick, channel)
            if key in self._conversations:
                del self._conversations[key]
                return True
            return False

    def clear_all(self) -> int:
        """Clear all conversation and channel contexts.

        Returns:
            Number of conversations cleared (not including channel contexts)
        """
        with self._lock:
            count = len(self._conversations)
            self._conversations.clear()
            self._channel_contexts.clear()
            return count

    def get_stats(self) -> dict[str, Any]:
        """Get context manager statistics.

        Returns:
            Dictionary with current statistics
        """
        with self._lock:
            self._prune_expired()

            total_messages = sum(len(conv.messages) for conv in self._conversations.values())
            channel_messages = sum(len(ctx.messages) for ctx in self._channel_contexts.values())

            return {
                "active_conversations": len(self._conversations),
                "total_messages": total_messages,
                "max_messages_per_conv": self.config.max_messages,
                "timeout_minutes": self.config.timeout_minutes,
                "enabled": self.config.enabled,
                "active_channels": len(self._channel_contexts),
                "channel_messages": channel_messages,
                "channel_max_messages": self.config.channel_max_messages,
            }

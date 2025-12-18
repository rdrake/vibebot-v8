"""Rate limiting for LLM commands."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    max_requests: int  # Maximum requests per window
    window_seconds: int  # Time window in seconds
    enabled: bool = True  # Whether rate limiting is enabled


class RateLimiter:
    """Thread-safe rate limiter using sliding window."""

    def __init__(self, config: RateLimitConfig) -> None:
        """Initialize rate limiter.

        Args:
            config: Rate limiting configuration
        """
        self.config = config
        self._lock = Lock()
        # Track timestamps of requests per identifier (user@channel)
        self._requests: defaultdict[str, list[float]] = defaultdict(list)

    def check_rate_limit(self, identifier: str) -> tuple[bool, str]:
        """Check if request is allowed under rate limit.

        Args:
            identifier: Unique identifier (e.g., "nick!user@host#channel")

        Returns:
            Tuple of (allowed: bool, message: str)
            If allowed=False, message contains error for user
        """
        if not self.config.enabled:
            return True, ""

        with self._lock:
            current_time = time.time()
            window_start = current_time - self.config.window_seconds

            # Get request history for this identifier
            request_times = self._requests[identifier]

            # Remove requests outside the current window
            request_times[:] = [t for t in request_times if t > window_start]

            # Check if limit exceeded
            if len(request_times) >= self.config.max_requests:
                # Calculate when the oldest request will expire
                oldest_request = min(request_times)
                wait_seconds = int(oldest_request + self.config.window_seconds - current_time)
                return False, (
                    f"Rate limit exceeded. "
                    f"You can make {self.config.max_requests} requests per "
                    f"{self.config.window_seconds}s. "
                    f"Try again in {wait_seconds}s."
                )

            # Record this request
            request_times.append(current_time)
            return True, ""

    def reset(self, identifier: str | None = None) -> None:
        """Reset rate limit for an identifier or all identifiers.

        Args:
            identifier: Specific identifier to reset, or None for all
        """
        with self._lock:
            if identifier is None:
                self._requests.clear()
            elif identifier in self._requests:
                del self._requests[identifier]

    def get_stats(self, identifier: str) -> dict[str, int | float]:
        """Get rate limit statistics for an identifier.

        Args:
            identifier: Identifier to get stats for

        Returns:
            Dictionary with current usage statistics
        """
        with self._lock:
            current_time = time.time()
            window_start = current_time - self.config.window_seconds

            request_times = self._requests.get(identifier, [])
            active_requests = [t for t in request_times if t > window_start]

            remaining = max(0, self.config.max_requests - len(active_requests))

            return {
                "requests_used": len(active_requests),
                "requests_remaining": remaining,
                "window_seconds": self.config.window_seconds,
                "max_requests": self.config.max_requests,
            }

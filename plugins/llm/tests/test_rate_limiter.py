"""Tests for RateLimiter."""

from __future__ import annotations

import threading
import time

from llm.rate_limiter import RateLimitConfig, RateLimiter


class TestRateLimiter:
    """Test rate limiter functionality."""

    def test_rate_limiter_allows_under_limit(self) -> None:
        """GIVEN rate limiter with limit 3/10s WHEN 2 requests THEN both allowed."""
        config = RateLimitConfig(max_requests=3, window_seconds=10, enabled=True)
        limiter = RateLimiter(config)

        # First request
        allowed, msg = limiter.check_rate_limit("user1")
        assert allowed is True
        assert msg == ""

        # Second request
        allowed, msg = limiter.check_rate_limit("user1")
        assert allowed is True
        assert msg == ""

    def test_rate_limiter_blocks_over_limit(self) -> None:
        """GIVEN rate limiter with limit 2/10s WHEN 3 requests THEN third blocked."""
        config = RateLimitConfig(max_requests=2, window_seconds=10, enabled=True)
        limiter = RateLimiter(config)

        # First two requests allowed
        assert limiter.check_rate_limit("user1")[0] is True
        assert limiter.check_rate_limit("user1")[0] is True

        # Third request blocked
        allowed, msg = limiter.check_rate_limit("user1")
        assert allowed is False
        assert "Rate limit exceeded" in msg

    def test_rate_limiter_window_expiry(self) -> None:
        """GIVEN rate limiter WHEN window expires THEN requests allowed again."""
        config = RateLimitConfig(max_requests=2, window_seconds=1, enabled=True)
        limiter = RateLimiter(config)

        # Fill up the limit
        assert limiter.check_rate_limit("user1")[0] is True
        assert limiter.check_rate_limit("user1")[0] is True
        assert limiter.check_rate_limit("user1")[0] is False

        # Wait for window to expire
        time.sleep(1.1)

        # Should allow requests again
        assert limiter.check_rate_limit("user1")[0] is True

    def test_rate_limiter_per_user_isolation(self) -> None:
        """GIVEN rate limiter WHEN different users THEN limits independent."""
        config = RateLimitConfig(max_requests=2, window_seconds=10, enabled=True)
        limiter = RateLimiter(config)

        # User1 fills their limit
        assert limiter.check_rate_limit("user1")[0] is True
        assert limiter.check_rate_limit("user1")[0] is True
        assert limiter.check_rate_limit("user1")[0] is False

        # User2 should still have their full limit
        assert limiter.check_rate_limit("user2")[0] is True
        assert limiter.check_rate_limit("user2")[0] is True

    def test_rate_limiter_disabled(self) -> None:
        """GIVEN rate limiter disabled WHEN requests THEN all allowed."""
        config = RateLimitConfig(max_requests=1, window_seconds=10, enabled=False)
        limiter = RateLimiter(config)

        # Should allow unlimited requests when disabled
        for _ in range(100):
            allowed, msg = limiter.check_rate_limit("user1")
            assert allowed is True
            assert msg == ""

    def test_rate_limiter_reset_specific(self) -> None:
        """GIVEN rate limiter WHEN reset specific user THEN only that user reset."""
        config = RateLimitConfig(max_requests=1, window_seconds=10, enabled=True)
        limiter = RateLimiter(config)

        # Both users hit limit
        assert limiter.check_rate_limit("user1")[0] is True
        assert limiter.check_rate_limit("user1")[0] is False
        assert limiter.check_rate_limit("user2")[0] is True
        assert limiter.check_rate_limit("user2")[0] is False

        # Reset only user1
        limiter.reset("user1")

        # User1 should be reset, user2 still limited
        assert limiter.check_rate_limit("user1")[0] is True
        assert limiter.check_rate_limit("user2")[0] is False

    def test_rate_limiter_reset_all(self) -> None:
        """GIVEN rate limiter WHEN reset all THEN all users reset."""
        config = RateLimitConfig(max_requests=1, window_seconds=10, enabled=True)
        limiter = RateLimiter(config)

        # Both users hit limit
        assert limiter.check_rate_limit("user1")[0] is True
        assert limiter.check_rate_limit("user1")[0] is False
        assert limiter.check_rate_limit("user2")[0] is True
        assert limiter.check_rate_limit("user2")[0] is False

        # Reset all
        limiter.reset()

        # Both should be reset
        assert limiter.check_rate_limit("user1")[0] is True
        assert limiter.check_rate_limit("user2")[0] is True

    def test_rate_limiter_stats(self) -> None:
        """GIVEN rate limiter WHEN get stats THEN accurate counts returned."""
        config = RateLimitConfig(max_requests=5, window_seconds=10, enabled=True)
        limiter = RateLimiter(config)

        # Make 3 requests
        for _ in range(3):
            limiter.check_rate_limit("user1")

        stats = limiter.get_stats("user1")
        assert stats["requests_used"] == 3
        assert stats["requests_remaining"] == 2
        assert stats["max_requests"] == 5
        assert stats["window_seconds"] == 10

    def test_rate_limiter_thread_safe(self) -> None:
        """GIVEN rate limiter WHEN concurrent requests THEN thread-safe."""
        config = RateLimitConfig(max_requests=100, window_seconds=10, enabled=True)
        limiter = RateLimiter(config)

        results: list[bool] = []
        lock = threading.Lock()

        def make_requests(user_id: str) -> None:
            for _ in range(10):
                allowed, _ = limiter.check_rate_limit(user_id)
                with lock:
                    results.append(allowed)

        # Run 10 threads, each making 10 requests
        threads = []
        for i in range(10):
            t = threading.Thread(target=make_requests, args=(f"user{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All 100 requests should be allowed (10 users x 10 requests, limit is 100 per user)
        assert all(results)
        assert len(results) == 100

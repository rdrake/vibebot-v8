"""Request tracing for LLM plugin."""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar

request_id: ContextVar[str] = ContextVar("request_id", default="")


def generate_request_id() -> str:
    """Generate a short unique request ID (8 hex chars)."""
    return uuid.uuid4().hex[:8]


class TraceFilter(logging.Filter):
    """Logging filter that prepends the current request ID to log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        rid = request_id.get()
        if rid:
            record.msg = f"[{rid}] {record.msg}"
        return True


# Headers that identify the backend server handling a request.
SERVER_ID_HEADERS = frozenset(("x-request-id", "cf-ray", "server", "x-server-id", "x-served-by"))


def extract_server_headers(source: object | None) -> dict[str, str]:
    """Extract server-identifying HTTP headers from a LiteLLM response or exception.

    Checks (in order):
    1. source._response_headers  (LiteLLM successful completions)
    2. source.response.headers   (LiteLLM exceptions with httpx.Response)
    3. source.headers            (fallback)

    Args:
        source: A LiteLLM response object, exception, or None.

    Returns:
        Dict of header-name -> value for recognised server headers.
        Empty dict when no headers are available.
    """
    if source is None:
        return {}

    raw: object | None = None

    # 1. LiteLLM response objects (_response_headers attribute)
    raw = getattr(source, "_response_headers", None)

    # 2. LiteLLM exceptions wrap httpx.Response on .response
    if raw is None:
        resp = getattr(source, "response", None)
        if resp is not None:
            raw = getattr(resp, "headers", None)

    # 3. Direct .headers fallback
    if raw is None:
        raw = getattr(source, "headers", None)

    if raw is None:
        return {}

    # raw may be dict, httpx.Headers, or similar mapping
    try:
        items = raw.items() if hasattr(raw, "items") else []
        return {k.lower(): v for k, v in items if k.lower() in SERVER_ID_HEADERS}
    except Exception:
        return {}

"""In-memory sliding-window rate limit, per client IP.

Opt-in via HUMANIZER_RATE_LIMIT="60/minute" or "1000/hour" or "10/second".
Returns 429 with Retry-After when exceeded. No external deps (no Redis,
no slowapi). Per-worker — for multi-worker deploys, prefer a shared
backend like Redis (this module would need to be swapped out).

Why a hand-rolled limiter: the only state we need is "for this IP,
when did the last N requests arrive?" — that fits in a `deque[float]`
per IP, ~100 bytes. The whole class is 60 lines and easy to audit.
For abuse, you'd add CDN-side limits (Cloudflare etc.) in production
anyway; this is the application-side floor.
"""
from __future__ import annotations

import os
import re
import threading
import time
from collections import defaultdict, deque
from typing import Deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


_UNIT_TO_SECONDS = {
    "second": 1, "seconds": 1, "s": 1,
    "minute": 60, "minutes": 60, "m": 60, "min": 60,
    "hour": 3600, "hours": 3600, "h": 3600, "hr": 3600,
}


def parse_rate(spec: str) -> tuple[int, int]:
    """Parse "60/minute" -> (60, 60). Returns (max_requests, window_seconds)."""
    m = re.match(r"\s*(\d+)\s*/\s*(\w+)\s*", spec)
    if not m:
        raise ValueError(f"bad rate spec {spec!r} — expected 'N/unit', e.g. '60/minute'")
    count = int(m.group(1))
    unit = m.group(2).lower()
    if unit not in _UNIT_TO_SECONDS:
        raise ValueError(f"bad rate unit {unit!r}; allowed: second, minute, hour")
    return count, _UNIT_TO_SECONDS[unit]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limit. One bucket per client IP."""

    def __init__(
        self,
        app,
        rate_spec: str | None = None,
        excluded_paths: tuple[str, ...] = ("/health", "/version"),
    ):
        super().__init__(app)
        self.rate_spec = rate_spec or os.environ.get("HUMANIZER_RATE_LIMIT", "")
        self.max_requests = 0
        self.window_seconds = 0
        if self.rate_spec:
            self.max_requests, self.window_seconds = parse_rate(self.rate_spec)
        self.excluded_paths = excluded_paths
        self._lock = threading.Lock()
        self._buckets: dict[str, Deque[float]] = defaultdict(deque)

    def _client_key(self, request: Request) -> str:
        # Prefer X-Forwarded-For when behind a proxy, falling back to direct.
        # In production, ensure your proxy strips/rewrites this header so
        # clients can't spoof it.
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next):
        if not self.rate_spec or request.url.path in self.excluded_paths:
            return await call_next(request)

        now = time.monotonic()
        key = self._client_key(request)
        with self._lock:
            bucket = self._buckets[key]
            cutoff = now - self.window_seconds
            # Drop entries older than the window.
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                # Compute Retry-After: seconds until oldest entry falls out.
                retry_after = max(1, int(self.window_seconds - (now - bucket[0])))
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": (
                            f"rate limit exceeded ({self.rate_spec}); "
                            f"retry in {retry_after}s"
                        ),
                    },
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(self.max_requests),
                        "X-RateLimit-Remaining": "0",
                    },
                )
            bucket.append(now)
            remaining = self.max_requests - len(bucket)

        response: Response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

"""
Lightweight API protections (Phase 2)
=======================================
Deliberately small and dependency-free. This is demo-level hardening, NOT a
substitute for production controls:

  * BodySizeLimitMiddleware -- rejects oversized request bodies (413) BEFORE any
    parsing, including chunked uploads that carry no Content-Length. It first
    drains a bounded amount of the rejected body and only then replies, so a client
    that is still uploading receives the 413 instead of a connection reset.
  * RateLimiter              -- in-process sliding-window budget per client IP
    for the expensive endpoints (429 + Retry-After).
  * cors_settings            -- explicit allow-list of frontend origins.

Production would additionally need real authentication/authorisation, TLS,
distributed rate limiting (shared across workers/instances), per-user quotas,
and audit logging. The in-process limiter resets on restart and is per-process.
"""

import asyncio
import json
import threading
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from fastapi import HTTPException, Request

from .config import CORS_ORIGINS, RATE_LIMITS, RATE_LIMIT_ENABLED


# ------------------------------------------------------------------- CORS
def cors_settings(origins: Optional[List[str]] = None) -> dict:
    """Keyword arguments for CORSMiddleware. Credentials are NEVER enabled: the
    dashboard uses no cookies/auth headers, and a wildcard origin must not be
    combined with credentials."""
    origins = list(CORS_ORIGINS if origins is None else origins)
    return {
        "allow_origins": origins,
        "allow_credentials": False,
        "allow_methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"],
        "max_age": 600,
    }


# ------------------------------------------------------- body size limit
class BodySizeLimitMiddleware:
    """Pure-ASGI middleware: 413 for request bodies above a per-path ceiling.
    The Content-Length header is checked first (cheap); a counting wrapper
    around `receive` also catches chunked bodies that omit it."""

    def __init__(self, app, limits: Dict[str, int], default_limit: int):
        self.app = app
        self.limits = limits
        self.default_limit = default_limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH"):
            await self.app(scope, receive, send)
            return

        limit = self.limits.get(scope["path"], self.default_limit)
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                if int(declared) > limit:
                    await self._drain(receive, limit)      # drain FIRST: the server closes the socket the moment
                    await self._reject(send, limit)        # a response completes while the request body is unfinished
                    return
            except ValueError:
                await self._reject(send, limit, status=400, detail="Invalid Content-Length header.")
                return

        received = 0
        rejected = False

        async def limited_receive():
            # Chunked bodies carry no Content-Length, so count what actually arrives. Raising here would be
            # swallowed by FastAPI's body parsing and turned into a generic 400, so instead drain, answer 413
            # ourselves, then present the app with a disconnect and swallow whatever it tries to send.
            nonlocal received, rejected
            if rejected:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    rejected = True
                    if message.get("more_body", False):
                        await self._drain(receive, limit)
                    await self._reject(send, limit)
                    return {"type": "http.disconnect"}
            return message

        async def tracked_send(message):
            if rejected:
                return
            await send(message)

        await self.app(scope, limited_receive, tracked_send)

    DRAIN_FACTOR = 8              # read and discard at most this many x the limit after rejecting...
    MIN_DRAIN_BYTES = 4_000_000   # ...but never less than this (a 16 KB-limit endpoint still absorbs a few MB cleanly)
    DRAIN_SECONDS = 5.0           # ...and never wait longer than this

    @classmethod
    async def _drain(cls, receive, limit: int) -> None:
        """Consume (and throw away) the rest of a rejected request body, bounded in size and time.
        Without this the server closes the socket while the client is still sending (uvicorn closes
        as soon as a response completes before the request body has), and the client sees a
        connection reset instead of the 413 response."""
        cap = max(limit * cls.DRAIN_FACTOR, cls.MIN_DRAIN_BYTES)

        async def consume():
            drained = 0
            while drained < cap:
                message = await receive()
                if message["type"] != "http.request":
                    return
                drained += len(message.get("body", b""))
                if not message.get("more_body", False):
                    return
        try:
            await asyncio.wait_for(consume(), timeout=cls.DRAIN_SECONDS)
        except (asyncio.TimeoutError, Exception):
            pass

    @staticmethod
    async def _reject(send, limit: int, status: int = 413, detail: Optional[str] = None):
        body = json.dumps({"detail": detail or f"Request body too large (maximum {limit} bytes)."}).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


# ------------------------------------------------------------ rate limit
class RateLimiter:
    """Sliding-window limiter: at most `max_requests` per `window_s` per
    (bucket, client). Thread-safe; in-process only."""

    def __init__(self, limits: Dict[str, Tuple[int, int]], enabled: bool = True):
        self.limits = dict(limits)
        self.enabled = enabled
        self._hits: Dict[Tuple[str, str], deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def configure(self, bucket: str, max_requests: int, window_s: int = 60) -> None:
        self.limits[bucket] = (max_requests, window_s)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()

    def check(self, bucket: str, client: str) -> Optional[int]:
        """Records a hit. Returns None if allowed, else the seconds to wait."""
        limit = self.limits.get(bucket)
        if not self.enabled or not limit:
            return None
        max_requests, window = limit
        now = time.monotonic()
        with self._lock:
            hits = self._hits[(bucket, client)]
            while hits and now - hits[0] >= window:
                hits.popleft()
            if len(hits) >= max_requests:
                return max(1, int(window - (now - hits[0])) + 1)
            hits.append(now)
            if len(self._hits) > 10_000:   # keep memory bounded under many distinct clients
                for key in [k for k, v in self._hits.items() if not v][:5_000]:
                    del self._hits[key]
        return None


rate_limiter = RateLimiter(RATE_LIMITS, enabled=RATE_LIMIT_ENABLED)


def client_key(request: Request) -> str:
    # The socket peer only. X-Forwarded-For is client-controlled and is NOT trusted here.
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(request: Request, bucket: str) -> None:
    retry_after = rate_limiter.check(bucket, client_key(request))
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests; try again in {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )


def rate_limit(bucket: str):
    """FastAPI dependency factory: Depends(rate_limit("predict"))."""
    def dependency(request: Request):
        enforce_rate_limit(request, bucket)
    return dependency

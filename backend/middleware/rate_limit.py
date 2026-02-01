"""Rate limiting middleware using Redis."""

import time
import logging
from typing import Callable
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from backend.utils.cache import get_cache

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using Redis.

    Implements token bucket algorithm with per-IP and per-user limits.
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10,
    ):
        """
        Initialize rate limiter.

        Args:
            app: FastAPI application
            requests_per_minute: Maximum requests per minute per client
            requests_per_hour: Maximum requests per hour per client
            burst_size: Maximum burst size
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size
        self.cache = get_cache()

    def get_client_id(self, request: Request) -> str:
        """
        Get unique client identifier.

        Uses user ID if authenticated, otherwise IP address.

        Args:
            request: FastAPI request

        Returns:
            Client identifier string
        """
        # Try to get user ID from request state (set by auth middleware)
        user = getattr(request.state, "user", None)
        if user:
            return f"user:{user.id}"

        # Fallback to IP address
        client_ip = request.client.host
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()

        return f"ip:{client_ip}"

    def is_rate_limited(self, client_id: str) -> tuple[bool, dict]:
        """
        Check if client is rate limited.

        Args:
            client_id: Client identifier

        Returns:
            Tuple of (is_limited, rate_limit_info)
        """
        if not self.cache.enabled:
            # If Redis is down, allow requests but log warning
            logger.warning("Rate limiting disabled: Redis not available")
            return False, {}

        current_time = int(time.time())

        # Check minute limit
        minute_key = f"ratelimit:minute:{client_id}:{current_time // 60}"
        minute_count = self.cache.get(minute_key) or 0

        if minute_count >= self.requests_per_minute:
            retry_after = 60 - (current_time % 60)
            return True, {
                "limit": self.requests_per_minute,
                "remaining": 0,
                "reset": current_time + retry_after,
                "retry_after": retry_after
            }

        # Check hour limit
        hour_key = f"ratelimit:hour:{client_id}:{current_time // 3600}"
        hour_count = self.cache.get(hour_key) or 0

        if hour_count >= self.requests_per_hour:
            retry_after = 3600 - (current_time % 3600)
            return True, {
                "limit": self.requests_per_hour,
                "remaining": 0,
                "reset": current_time + retry_after,
                "retry_after": retry_after
            }

        # Increment counters
        self.cache.increment(minute_key)
        self.cache.set(minute_key, (minute_count or 0) + 1, ttl=60)

        self.cache.increment(hour_key)
        self.cache.set(hour_key, (hour_count or 0) + 1, ttl=3600)

        # Return rate limit info
        minute_remaining = self.requests_per_minute - (minute_count + 1)
        hour_remaining = self.requests_per_hour - (hour_count + 1)

        return False, {
            "limit_minute": self.requests_per_minute,
            "limit_hour": self.requests_per_hour,
            "remaining_minute": max(0, minute_remaining),
            "remaining_hour": max(0, hour_remaining),
            "reset_minute": (current_time // 60 + 1) * 60,
            "reset_hour": (current_time // 3600 + 1) * 3600,
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with rate limiting.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response

        Raises:
            HTTPException: If rate limit exceeded
        """
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        # Get client ID
        client_id = self.get_client_id(request)

        # Check rate limit
        is_limited, rate_info = self.is_rate_limited(client_id)

        if is_limited:
            logger.warning(f"Rate limit exceeded for client {client_id}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "retry_after": rate_info.get("retry_after"),
                },
                headers={
                    "Retry-After": str(rate_info.get("retry_after", 60)),
                    "X-RateLimit-Limit": str(rate_info.get("limit")),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(rate_info.get("reset")),
                }
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit-Minute"] = str(rate_info.get("limit_minute", self.requests_per_minute))
        response.headers["X-RateLimit-Limit-Hour"] = str(rate_info.get("limit_hour", self.requests_per_hour))
        response.headers["X-RateLimit-Remaining-Minute"] = str(rate_info.get("remaining_minute", 0))
        response.headers["X-RateLimit-Remaining-Hour"] = str(rate_info.get("remaining_hour", 0))

        return response

"""Security middleware for ScholarGenie API."""

import logging
import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import Headers

logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware implementing various security best practices.

    - Request logging
    - Security headers
    - Request timing
    - Input validation
    """

    def __init__(self, app):
        """
        Initialize security middleware.

        Args:
            app: FastAPI application
        """
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with security measures.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response with security headers
        """
        # Record start time
        start_time = time.time()

        # Log request
        logger.info(
            f"{request.method} {request.url.path} "
            f"from {request.client.host} "
            f"User-Agent: {request.headers.get('user-agent', 'unknown')}"
        )

        # Process request
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        # Add processing time header
        response.headers["X-Process-Time"] = f"{process_time:.4f}"

        # Remove server header
        response.headers["Server"] = "ScholarGenie"

        # Log response
        logger.info(
            f"{request.method} {request.url.path} "
            f"completed with {response.status_code} "
            f"in {process_time:.4f}s"
        )

        return response


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for validating incoming requests.

    - Content-Type validation
    - Request size limits
    - Header validation
    """

    def __init__(
        self,
        app,
        max_content_length: int = 100 * 1024 * 1024,  # 100 MB
    ):
        """
        Initialize request validation middleware.

        Args:
            app: FastAPI application
            max_content_length: Maximum request content length in bytes
        """
        super().__init__(app)
        self.max_content_length = max_content_length

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Validate and process request.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response
        """
        # Validate content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_content_length:
            logger.warning(
                f"Request too large: {content_length} bytes from {request.client.host}"
            )
            return Response(
                content="Request entity too large",
                status_code=413,
                headers={"Content-Type": "text/plain"}
            )

        # Validate Content-Type for POST/PUT/PATCH
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")

            # Allow JSON and form data
            allowed_types = [
                "application/json",
                "application/x-www-form-urlencoded",
                "multipart/form-data"
            ]

            # Skip validation for file upload endpoints
            if "/upload" not in request.url.path:
                if not any(allowed in content_type for allowed in allowed_types):
                    logger.warning(
                        f"Invalid Content-Type: {content_type} from {request.client.host}"
                    )

        # Process request
        response = await call_next(request)
        return response


class APILoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for detailed API request/response logging.

    Useful for debugging and monitoring.
    """

    def __init__(self, app, log_request_body: bool = False, log_response_body: bool = False):
        """
        Initialize API logging middleware.

        Args:
            app: FastAPI application
            log_request_body: Whether to log request bodies
            log_response_body: Whether to log response bodies
        """
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Log request and response details.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response
        """
        # Log request details
        log_data = {
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_host": request.client.host,
            "headers": {
                k: v for k, v in request.headers.items()
                if k.lower() not in ["authorization", "cookie"]
            }
        }

        # Optionally log request body (be careful with sensitive data)
        if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                log_data["body_size"] = len(body)
                # Don't log actual body content in production for security
            except:
                pass

        logger.debug(f"Request: {log_data}")

        # Process request
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        # Log response details
        response_log = {
            "status_code": response.status_code,
            "duration_ms": round(duration * 1000, 2),
            "headers": dict(response.headers)
        }

        logger.debug(f"Response: {response_log}")

        return response

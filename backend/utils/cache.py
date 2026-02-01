"""Redis caching utilities for ScholarGenie."""

import os
import json
import pickle
from typing import Any, Optional, Callable
from datetime import timedelta
from functools import wraps
import redis
from redis import Redis
import logging

logger = logging.getLogger(__name__)

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_DECODE_RESPONSES = False  # We'll handle encoding/decoding manually

# Default cache TTLs (in seconds)
DEFAULT_TTL = 3600  # 1 hour
PAPER_CACHE_TTL = 86400  # 24 hours
SUMMARY_CACHE_TTL = 43200  # 12 hours
SEARCH_CACHE_TTL = 1800  # 30 minutes


class RedisCache:
    """
    Redis caching service for ScholarGenie.

    Provides caching functionality with automatic serialization/deserialization,
    TTL management, and common cache operations.
    """

    def __init__(
        self,
        host: str = REDIS_HOST,
        port: int = REDIS_PORT,
        db: int = REDIS_DB,
        password: Optional[str] = REDIS_PASSWORD
    ):
        """
        Initialize Redis cache client.

        Args:
            host: Redis server hostname
            port: Redis server port
            db: Redis database number
            password: Redis password (if required)
        """
        try:
            self.client = Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False,  # We handle encoding ourselves
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
            self.enabled = True
        except redis.ConnectionError as e:
            logger.warning(f"Could not connect to Redis: {e}. Caching will be disabled.")
            self.enabled = False
            self.client = None

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        if not self.enabled:
            return None

        try:
            value = self.client.get(key)
            if value is None:
                return None

            # Try to deserialize
            try:
                return pickle.loads(value)
            except:
                # Fallback to JSON
                return json.loads(value.decode('utf-8'))
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = DEFAULT_TTL
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None for no expiration)

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            # Serialize value
            try:
                serialized = pickle.dumps(value)
            except:
                # Fallback to JSON
                serialized = json.dumps(value).encode('utf-8')

            # Set with or without TTL
            if ttl:
                self.client.setex(key, ttl, serialized)
            else:
                self.client.set(key, serialized)

            return True
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False otherwise
        """
        if not self.enabled:
            return False

        try:
            return bool(self.client.delete(key))
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists, False otherwise
        """
        if not self.enabled:
            return False

        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {e}")
            return False

    def clear_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern.

        Args:
            pattern: Key pattern (e.g., "paper:*")

        Returns:
            Number of keys deleted
        """
        if not self.enabled:
            return 0

        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Error clearing cache pattern {pattern}: {e}")
            return 0

    def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment a counter.

        Args:
            key: Cache key
            amount: Amount to increment by

        Returns:
            New value or None on error
        """
        if not self.enabled:
            return None

        try:
            return self.client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Error incrementing cache key {key}: {e}")
            return None

    def get_ttl(self, key: str) -> Optional[int]:
        """
        Get remaining TTL for a key.

        Args:
            key: Cache key

        Returns:
            TTL in seconds, -1 if no TTL, -2 if key doesn't exist, None on error
        """
        if not self.enabled:
            return None

        try:
            return self.client.ttl(key)
        except Exception as e:
            logger.error(f"Error getting TTL for cache key {key}: {e}")
            return None

    def flush_all(self) -> bool:
        """
        Flush entire cache database.

        WARNING: This will delete ALL keys in the current database.

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            self.client.flushdb()
            logger.warning("Flushed entire Redis database")
            return True
        except Exception as e:
            logger.error(f"Error flushing cache: {e}")
            return False


# Global cache instance
_cache_instance: Optional[RedisCache] = None


def get_cache() -> RedisCache:
    """
    Get global cache instance (singleton).

    Returns:
        RedisCache instance
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
    return _cache_instance


def cached(
    ttl: int = DEFAULT_TTL,
    key_prefix: str = "",
    key_func: Optional[Callable] = None
):
    """
    Decorator for caching function results.

    Args:
        ttl: Time-to-live in seconds
        key_prefix: Prefix for cache key
        key_func: Custom function to generate cache key from arguments

    Example:
        @cached(ttl=3600, key_prefix="paper")
        def get_paper(paper_id: str):
            # Expensive operation
            return paper

        @cached(key_func=lambda query, limit: f"search:{query}:{limit}")
        def search_papers(query: str, limit: int = 10):
            return results
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default: use function name and args
                args_str = "_".join(str(arg) for arg in args)
                kwargs_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = f"{key_prefix}:{func.__name__}:{args_str}:{kwargs_str}"

            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_result

            # Execute function
            logger.debug(f"Cache miss for key: {cache_key}")
            result = func(*args, **kwargs)

            # Store in cache
            cache.set(cache_key, result, ttl=ttl)

            return result

        return wrapper
    return decorator


# Convenience functions for common cache operations

def cache_paper(paper_id: str, paper_data: dict, ttl: int = PAPER_CACHE_TTL) -> bool:
    """Cache paper data."""
    cache = get_cache()
    return cache.set(f"paper:{paper_id}", paper_data, ttl=ttl)


def get_cached_paper(paper_id: str) -> Optional[dict]:
    """Get cached paper data."""
    cache = get_cache()
    return cache.get(f"paper:{paper_id}")


def cache_summary(paper_id: str, summary_data: dict, ttl: int = SUMMARY_CACHE_TTL) -> bool:
    """Cache paper summary."""
    cache = get_cache()
    return cache.set(f"summary:{paper_id}", summary_data, ttl=ttl)


def get_cached_summary(paper_id: str) -> Optional[dict]:
    """Get cached paper summary."""
    cache = get_cache()
    return cache.get(f"summary:{paper_id}")


def cache_search_results(query: str, results: list, ttl: int = SEARCH_CACHE_TTL) -> bool:
    """Cache search results."""
    cache = get_cache()
    # Hash the query for consistent key
    import hashlib
    query_hash = hashlib.md5(query.encode()).hexdigest()
    return cache.set(f"search:{query_hash}", results, ttl=ttl)


def get_cached_search_results(query: str) -> Optional[list]:
    """Get cached search results."""
    cache = get_cache()
    import hashlib
    query_hash = hashlib.md5(query.encode()).hexdigest()
    return cache.get(f"search:{query_hash}")


def invalidate_paper_cache(paper_id: str) -> None:
    """Invalidate all cache entries for a paper."""
    cache = get_cache()
    cache.delete(f"paper:{paper_id}")
    cache.delete(f"summary:{paper_id}")
    cache.clear_pattern(f"*:{paper_id}:*")

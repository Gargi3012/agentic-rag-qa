import time
import logging
import threading
from collections import defaultdict
from fastapi import HTTPException, Security, status
from app.config import Config
from app.api.auth import get_api_key

logger = logging.getLogger("agentic_rag.api.rate_limiter")

# Class to manage rate limiting state safely
class InMemoryRateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
        self.lock = threading.Lock()

    def check_rate_limit(self, key: str, limit_rpm: int):
        """
        Validates request counts in a 60-second sliding window.
        Raises 429 Too Many Requests if limit is exceeded.
        """
        now = time.time()
        window_size = 60.0  # seconds

        with self.lock:
            # Get request timestamps for this key
            timestamps = self.requests[key]
            
            # Remove timestamps older than the sliding window
            timestamps = [ts for ts in timestamps if now - ts < window_size]
            self.requests[key] = timestamps
            
            if len(timestamps) >= limit_rpm:
                logger.warning(f"Rate limit exceeded for key {key}: {len(timestamps)} requests in 60s (limit: {limit_rpm})")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Maximum {limit_rpm} requests per minute are allowed."
                )
            
            # Record current request
            self.requests[key].append(now)

# Create a singleton rate limiter instance
_limiter = InMemoryRateLimiter()

def rate_limit_dependency(api_key: str = Security(get_api_key)):
    """
    FastAPI dependency to enforce rate limiting on endpoints.
    Requires API key authentication first.
    """
    limit = Config.RATE_LIMIT_RPM
    _limiter.check_rate_limit(api_key, limit)
    return api_key

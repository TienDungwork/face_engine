import time
from fastapi import Request, HTTPException
from collections import deque
import asyncio


class RateLimiter:
    def __init__(self, requests_per_second: int = 20):
        self.requests_per_second = requests_per_second
        self.requests = deque()
        self._lock = asyncio.Lock()

    async def is_allowed(self) -> bool:
        async with self._lock:
            now = time.time()

            # Remove requests older than 1 second
            while self.requests and self.requests[0] < now - 1:
                self.requests.popleft()

            # Check if we're under the limit
            if len(self.requests) < self.requests_per_second:
                self.requests.append(now)
                return True

            return False


rate_limiter = RateLimiter(20)  # 20 requests per second


async def rate_limit_middleware(request: Request, call_next):
    if not await rate_limiter.is_allowed():
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later."
        )
    return await call_next(request)

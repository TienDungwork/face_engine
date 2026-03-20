from fastapi import Request
import torch
import gc


async def cleanup_memory(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

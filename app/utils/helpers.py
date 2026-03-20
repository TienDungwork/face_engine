import cv2
import numpy as np
import base64
import asyncio
import aiohttp
from typing import List
from app.core.config import app_config


async def base64_to_image(base64_image: str) -> np.ndarray:
    if not base64_image:
        return None
    try:
        loop = asyncio.get_event_loop()
        image_data = await loop.run_in_executor(None, base64.b64decode, base64_image)
        image = await loop.run_in_executor(
            None,
            lambda: cv2.imdecode(np.frombuffer(
                image_data, np.uint8), cv2.IMREAD_COLOR)
        )
        return image
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None


async def download_image(url: str) -> np.ndarray:
    """Download image from URL and convert to OpenCV format asynchronously."""
    try:
        session = aiohttp.ClientSession()
        response = await session.get(url)
        response.raise_for_status()
        content = await response.read()
        await session.close()

        loop = asyncio.get_event_loop()
        img = await loop.run_in_executor(
            None,
            lambda: cv2.imdecode(np.frombuffer(
                content, np.uint8), cv2.IMREAD_COLOR)
        )
        if img is None:
            raise ValueError("Failed to decode image")
        return img
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None


def expand_image(bbox: List[int], image: np.ndarray, ratio: float = 0.2) -> np.ndarray:
    """Expand image to 4 times its original size."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = int(max(1, x1 - (x2 - x1) * ratio))
    y1 = int(max(1, y1 - (y2 - y1) * ratio))
    x2 = int(min(w - 1, x2 + (x2 - x1) * ratio))
    y2 = int(min(h - 1, y2 + (y2 - y1) * ratio))
    return [x1, y1, x2, y2]


def cosine_similarity(embedding_1: np.ndarray, embedding_2: np.ndarray) -> float:
    return float(np.dot(embedding_1, embedding_2) / (
        np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2)))


def encode_embedding(embedding: np.ndarray, secret_key: str = app_config.SECRET_KEY) -> str:
    return secret_key + base64.b64encode(embedding.tobytes()).decode('utf-8')


def decode_embedding(embedding: str, secret_key: str = app_config.SECRET_KEY) -> np.ndarray:
    return np.frombuffer(base64.b64decode(embedding.replace(secret_key, '')), dtype=np.float32)

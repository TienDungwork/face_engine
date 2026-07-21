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
        s = base64_image.strip()
        if s.lower().startswith("data:") and "," in s:
            s = s.split(",", 1)[1].strip()
        # Loại bỏ mọi whitespace để tránh lỗi "Incorrect padding"
        s = s.replace("\n", "").replace("\r", "").replace(" ", "")
        if s:
            s += "=" * (-len(s) % 4)
        loop = asyncio.get_event_loop()
        image_data = await loop.run_in_executor(None, base64.b64decode, s)
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
    """
    Giải mã chuỗi: SECRET_KEY + base64(float32 bytes).

    Lưu ý: KHÔNG dùng `replace(secret_key, '')` vì secret_key có thể xuất hiện
    trong chính phần base64 (đặc biệt khi secret_key ngắn) gây lỗi giải mã.
    """
    if embedding is None:
        raise ValueError("embedding rỗng")

    s = str(embedding).strip()
    if secret_key and s.startswith(secret_key):
        payload = s[len(secret_key):].strip()
    else:
        # Trường hợp không match prefix (fallback): cố gắng decode trực tiếp.
        payload = s

    # Base64 có thể thiếu padding '='
    if payload:
        payload += "=" * (-len(payload) % 4)
    raw = base64.b64decode(payload)

    # Một số dữ liệu trong DB có thể bị "thiếu" byte cuối lẻ (len%4 != 0),
    # làm `frombuffer(..., float32)` crash. Pad thêm byte 0 để hợp lệ.
    if len(raw) % 4 != 0:
        pad_needed = (-len(raw)) % 4
        raw = raw + (b"\x00" * pad_needed)

    arr = np.frombuffer(raw, dtype=np.float32)
    if arr.size == 512:
        return arr
    # fallback: nếu có dư do padding/format khác, chỉ lấy 512 phần đầu
    if arr.size > 512:
        return arr[:512]
    raise ValueError(f"embedding decoded vector size {arr.size}, need 512")


def decode_smf_face_feature_bytes(
    raw: bytes,
    secret_key: str = app_config.SECRET_KEY,
) -> np.ndarray:
    """
    Giải mã cột face_feature (BYTEA) trong smf_face_events:
    - 2048 byte = 512 float32 thô; hoặc
    - UTF-8 là chuỗi base64 của buffer float32; hoặc format SECRET_KEY + base64 (DB nội bộ).
    """
    if not raw:
        raise ValueError("face_feature rỗng")
    if len(raw) == 512 * 4:
        arr = np.frombuffer(raw, dtype=np.float32)
        if arr.shape[0] == 512:
            return arr.copy()
    try:
        s = raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        arr = np.frombuffer(raw, dtype=np.float32)
        if arr.size == 512:
            return arr.copy()
        raise ValueError(f"face_feature nhị phân sai kích thước: {len(raw)} byte") from None

    # base64 có thể có whitespace hoặc thiếu padding '='
    s = s.replace("\n", "").replace("\r", "").replace(" ", "")
    try:
        if s:
            s_padded = s + "=" * (-len(s) % 4)
        else:
            s_padded = s
        buf = base64.b64decode(s_padded)
        arr = np.frombuffer(buf, dtype=np.float32)
    except Exception:
        arr = np.array([], dtype=np.float32)
    if arr.size == 512:
        return arr
    if secret_key and s.startswith(secret_key):
        return decode_embedding(s, secret_key)
    if arr.size > 0:
        raise ValueError(f"face_feature sai kích thước vector: {arr.size}, cần 512")
    raise ValueError("Không decode được face_feature")


def batch_cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity từ một vector query tới từng hàng của matrix (Mx512)."""
    qn = np.linalg.norm(query)
    if qn == 0:
        return np.zeros(matrix.shape[0], dtype=np.float64)
    mn = np.linalg.norm(matrix, axis=1, keepdims=True)
    mn = np.where(mn == 0, 1.0, mn)
    normed = matrix / mn
    return normed @ (query / qn)

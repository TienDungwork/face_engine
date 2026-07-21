import re
import traceback
from typing import Optional

import asyncpg

from app.core.config import app_config, get_smart_face_dsn

_pool: Optional[asyncpg.Pool] = None
_last_pool_error: Optional[str] = None

# FQN bảng camera: schema.bảng hoặc schema.public.bảng (1–2 dấu chấm sau phần đầu).
_SAFE_FQN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*){1,2}$")
# Bảng event: smf_face_events hoặc schema.smf_face_events
_SAFE_EVENTS_TABLE = re.compile(
    r"^[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?$"
)


def validate_qualified_table_name(fqn: str) -> bool:
    return bool(fqn and _SAFE_FQN.fullmatch(fqn))


def validate_events_table_ref(name: str) -> bool:
    return bool(name and _SAFE_EVENTS_TABLE.fullmatch(name))


def _ssl_kwarg() -> dict:
    raw = (app_config.SMART_FACE_DB_SSL or "").strip().lower()
    if raw in ("disable", "false", "0", "no", "off"):
        return {"ssl": False}
    if raw in ("require", "true", "1", "yes", "on"):
        return {"ssl": True}
    return {}


async def init_smart_face_pool() -> None:
    global _pool, _last_pool_error
    url = (get_smart_face_dsn() or "").strip()
    if not url:
        _last_pool_error = None
        return
    if _pool is not None:
        return
    _last_pool_error = None
    try:
        _pool = await asyncpg.create_pool(
            url,
            min_size=1,
            max_size=5,
            command_timeout=120,
            # Truy vấn đổi số placeholder/kiểu theo request; cache statement gây lệch OID
            # (lỗi kiểu 'int' object has no attribute 'bytes' ở tham số $n).
            statement_cache_size=0,
            **_ssl_kwarg(),
        )
    except Exception as e:
        _pool = None
        _last_pool_error = f"{type(e).__name__}: {e}"
        print(f"[smart_face] Không tạo được pool PostgreSQL: {_last_pool_error}")
        traceback.print_exc()


def get_last_smart_face_pool_error() -> Optional[str]:
    return _last_pool_error


async def close_smart_face_pool() -> None:
    global _pool, _last_pool_error
    if _pool is not None:
        await _pool.close()
        _pool = None
    _last_pool_error = None


def get_smart_face_pool() -> Optional[asyncpg.Pool]:
    return _pool

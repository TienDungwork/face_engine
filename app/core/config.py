import yaml
import json
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / "resources" / ".env"
load_dotenv(_ENV_FILE)


class InsightfaceConfig(BaseModel):
    ctx_id: int = 0
    device_id: int = 0
    det_size: List[int] = [320, 320]
    allowed_modules: List[str] = ["detection", "recognition"]


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # APP
    APP_NAME: str
    APP_VERSION: str
    API_V1_STR: str
    ALLOWED_ORIGINS: List[str] = ["*"]

    # API
    API_HOST: str
    API_PORT: int
    API_WORKERS: int
    API_RELOAD: bool
    SECRET_KEY: str

    # Threshold
    SIMILARITY_THRESHOLD: float = 0.7
    CENTROID_THRESHOLD: float = 1.0

    # Paging
    PAGING_URL: str
    GET_PERSON_URL: str

    # Sync configuration
    SYNC_INTERVAL_MINUTES: int = 5

    # Database reload configuration
    RELOAD_DB: bool = True

    # Centroid reload configuration
    RELOAD_CENTROID: bool = True
    CENTROID_RELOAD_HOUR: int = 2
    CENTROID_RELOAD_MINUTE: int = 0

    # smart_face DB (smf_face_events) — bật khi có URL hoặc đủ SMART_FACE_DB_*
    SMART_FACE_DATABASE_URL: Optional[str] = None
    SMART_FACE_DB_HOST: Optional[str] = None
    SMART_FACE_DB_PORT: int = 5432
    SMART_FACE_DB_USER: Optional[str] = None
    SMART_FACE_DB_PASSWORD: Optional[str] = None
    SMART_FACE_DB_NAME: Optional[str] = None
    # asyncpg: disable / require / để trống (mặc định). Một số PG nội bộ cần disable
    SMART_FACE_DB_SSL: Optional[str] = None
    SMART_FACE_DEFAULT_SEARCH_DAYS: int = 7
    SMART_FACE_EVENTS_MAX_ROWS: int = 10000
    # Nếu access_time trong PG là TIMESTAMP naive theo múi này (vd Asia/Ho_Chi_Minh), set để bind đúng
    SMART_FACE_ACCESS_TIME_TZ: Optional[str] = None
    # Tên bảng event (mặc định smf_face_events); có thể schema.bảng
    SMART_FACE_EVENTS_TABLE: str = "smf_face_events"
    # Bảng camera để LEFT JOIN, ví dụ vms_db.camera (ưu tiên hơn SMART_FACE_CAMERA_TABLE_FQN)
    SMART_FACE_CAMERA_TABLE_QUALIFIED: Optional[str] = None
    SMART_FACE_CAMERA_TABLE_FQN: Optional[str] = None

    # Configs
    insightface: InsightfaceConfig


class Config:
    env_file = "resources/.env"
    env_file_encoding = "utf-8"


def load_yaml_config(file_path: str) -> dict:
    """Helper function to load YAML config."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def load_config() -> AppConfig:
    # Load configurations from YAML files
    yaml_path = _PROJECT_ROOT / "resources" / "config" / "recognize.yaml"
    insightface_config = load_yaml_config(str(yaml_path))
    return AppConfig(**insightface_config)


# Create an instance of the config
app_config = load_config()


def get_smart_face_dsn() -> Optional[str]:
    """
    Chuỗi kết nối asyncpg. Ưu tiên SMART_FACE_DATABASE_URL;
    nếu trống thì ghép từ SMART_FACE_DB_HOST / USER / PASSWORD / NAME / PORT.
    """
    url = (app_config.SMART_FACE_DATABASE_URL or "").strip()
    if url:
        return url
    host = (app_config.SMART_FACE_DB_HOST or "").strip()
    user = (app_config.SMART_FACE_DB_USER or "").strip()
    db = (app_config.SMART_FACE_DB_NAME or "").strip()
    if not host or not user or not db:
        return None
    password = app_config.SMART_FACE_DB_PASSWORD or ""
    port = int(app_config.SMART_FACE_DB_PORT)
    return (
        f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}"
        f"@{host}:{port}/{quote(db, safe='')}"
    )

if __name__ == "__main__":
    print(json.dumps(app_config.model_dump(), indent=4))

import yaml
import json
from typing import List, Optional
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv("resources/.env")


class InsightfaceConfig(BaseModel):
    ctx_id: int = 0
    device_id: int = 0
    det_size: List[int] = [320, 320]
    allowed_modules: List[str] = ["detection", "recognition"]


class AppConfig(BaseSettings):
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
    SYNC_INTERVAL_HOURS: int = 1

    # Database reload configuration
    RELOAD_DB: bool = True

    # Centroid reload configuration
    RELOAD_CENTROID: bool = True
    CENTROID_RELOAD_HOUR: int = 2
    CENTROID_RELOAD_MINUTE: int = 0

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
    insightface_config = load_yaml_config("resources/config/recognize.yaml")
    return AppConfig(**insightface_config)


# Create an instance of the config
app_config = load_config()

if __name__ == "__main__":
    print(json.dumps(app_config.model_dump(), indent=4))

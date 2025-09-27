from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import List, Any

# Point to project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    APP_NAME: str = "Interview AI Assistant"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    DATABASE_URL: str = "sqlite+aiosqlite:///./interview_ai.db"
    REDIS_URL: str = "redis://localhost:6379/0"
    SECRET_KEY: str = "super-secret-key-change-me"
    API_KEY: str = "my-test-api-key"

    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    PROMETHEUS_ENABLED: bool = True
    RATE_LIMIT: str = "100/minute"

    AUDIO_STORAGE_PATH: str = "uploads/audio"

    LLAVA_API_URL: str = "http://localhost:11434/api/generate"
    LLAVA_API_KEY: str = "your-llava-api-key"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "nous-hermes2"

    VOICE_SAMPLE_RATE: int = 16000
    VOICE_MAX_DURATION: int = 300
    VOICE_MODEL: str = "whisper-small"

    IMAGE_MAX_SIZE_MB: int = 5
    ALLOWED_IMAGE_TYPES: List[str] = ["jpg", "jpeg", "png"]
    IMAGE_MODEL: str = "clip-vit-base"

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        case_sensitive=True,
        extra="allow",
    )

    @field_validator(
        "ALLOWED_HOSTS",
        "BACKEND_CORS_ORIGINS",
        "CORS_ORIGINS",
        "ALLOWED_IMAGE_TYPES",
        mode="before",
    )
    @classmethod
    def parse_list(cls, v: Any) -> list[str]:
        """
        Accept either:
        - a proper JSON array (preferred), or
        - a simple comma-separated string (fallback).
        """
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):  # looks like JSON
                import json
                return json.loads(v)
            return [item.strip() for item in v.split(",") if item.strip()]
        return v


settings = Settings()

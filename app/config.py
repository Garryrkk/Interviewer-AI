from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import List, Any

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    # General
    APP_NAME: str = "Interview AI Assistant"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    # DBs
    DATABASE_URL: str
    MONGO_URL: str
    REDIS_URL: str
    UPSTASH_REDIS_REST_TOKEN: str

    # Security
    SECRET_KEY: str
    API_KEY: str

    # LLMs
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    NOUS_HERMES_BASE_URL: str = "http://localhost:11434"
    NOUS_HERMES_MODEL: str = r"C:/llama-gpu/models/Nous-Hermes-2-Mistral-7B-DPO.Q4_K_S.gguf"
    LLAVA_BASE_URL: str = "http://localhost:11435"
    LLAVA_MODEL: str = r"C:/llama-gpu/models/llava-hr-7b-sft-1024"

    # App Settings
    ALLOWED_HOSTS: List[str]
    BACKEND_CORS_ORIGINS: List[str]
    CORS_ORIGINS: List[str]
    PROMETHEUS_ENABLED: bool = True
    RATE_LIMIT: str = "100/minute"

    # Audio / Images
    AUDIO_STORAGE_PATH: str = "uploads/audio"
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
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                import json
                return json.loads(v)
            return [item.strip() for item in v.split(",") if item.strip()]
        return v


settings = Settings()
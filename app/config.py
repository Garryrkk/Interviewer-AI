from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    APP_NAME: str = "Interview AI Assistant"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    DATABASE_URL: str
    REDIS_URL: str

    SECRET_KEY: str
    API_KEY: str

    ALLOWED_HOSTS: list[str] = ["localhost"]
    BACKEND_CORS_ORIGINS: list[str] = ["*"]

    PROMETHEUS_ENABLED: bool = False
    RATE_LIMIT: str = "100/minute"

    # Optional LLM / voice / image settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "nous-hermes2"

    VOICE_SAMPLE_RATE: int = 16000
    VOICE_MAX_DURATION: int = 300
    VOICE_MODEL: str = "whisper-small"

    IMAGE_MAX_SIZE_MB: int = 5
    ALLOWED_IMAGE_TYPES: list[str] = ["jpg", "jpeg", "png"]
    IMAGE_MODEL: str = "clip-vit-base"

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        case_sensitive=True,
        extra="allow",
    )

    @field_validator("ALLOWED_HOSTS", "BACKEND_CORS_ORIGINS", mode="before")
    def parse_env_list(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v


settings = Settings()

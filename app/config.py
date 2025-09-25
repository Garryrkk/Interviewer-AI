
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ENVIRONMENT: str
    SECRET_KEY: str
    API_KEY: str
    REDIS_URL: str
    ALLOWED_HOSTS: str
    CORS_ORIGINS: str
    PROMETHEUS_ENABLED: bool
    RATE_LIMIT: str

    # Pydantic v2 config
    model_config = SettingsConfigDict(extra="allow")
# Load .env file from project root
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    # ---------------------------
    # Application metadata
    # ---------------------------
    APP_NAME: str = "Interviewer-AI"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "AI-powered interviewer assistant with summarization, quick response, voice & image recognition."

    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # ---------------------------
    # Database config
    # ---------------------------
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./interviewer.db")

    # ---------------------------
    # Ollama / LLM settings
    # ---------------------------
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "nous-hermes2")

    # ---------------------------
    # Security / Auth
    # ---------------------------
    JWT_SECRET: str = os.getenv("JWT_SECRET", "super-secret-key")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

    # ---------------------------
    # CORS / Allowed Origins
    # ---------------------------
    BACKEND_CORS_ORIGINS: list[str] = os.getenv("BACKEND_CORS_ORIGINS", "*").split(",")

    # ---------------------------
    # Voice recognition
    # ---------------------------
    VOICE_SAMPLE_RATE: int = int(os.getenv("VOICE_SAMPLE_RATE", "16000"))
    VOICE_MAX_DURATION: int = int(os.getenv("VOICE_MAX_DURATION", "300"))  # in seconds
    VOICE_MODEL: str = os.getenv("VOICE_MODEL", "whisper-small")


    # ---------------------------
    # Image recognition
    # ---------------------------
    IMAGE_MAX_SIZE_MB: int = int(os.getenv("IMAGE_MAX_SIZE_MB", "5"))
    ALLOWED_IMAGE_TYPES: list[str] = os.getenv("ALLOWED_IMAGE_TYPES", "jpg,jpeg,png").split(",")
    IMAGE_MODEL: str = os.getenv("IMAGE_MODEL", "clip-vit-base")


    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()
"""Backend settings; .env is resolved relative to the project root."""

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        extra="forbid",
        hide_input_in_errors=True,
    )

    APP_NAME: str = Field(default="Enterprise SalesOps Agent", min_length=1)
    APP_ENV: str = Field(default="development", min_length=1)
    DEBUG: bool = False
    BACKEND_HOST: str = Field(default="127.0.0.1", min_length=1)
    BACKEND_PORT: int = Field(default=8000, ge=1, le=65535)
    DATABASE_URL: str = "sqlite:///./backend/storage/salesops.db"
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]
    LLM_PROVIDER: str = ""
    LLM_MODEL: str = ""
    LLM_BASE_URL: str = ""
    LLM_API_KEY: SecretStr | None = None
    EMBEDDING_PROVIDER: str = "siliconflow"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_BASE_URL: str = "https://api.siliconflow.cn/v1"
    EMBEDDING_API_KEY: SecretStr | None = None
    APPROVAL_TTL_MINUTES: int = Field(default=30, gt=0, le=1440)

    @field_validator("CORS_ORIGINS")
    @classmethod
    def explicit_origins(cls, origins: list[str]) -> list[str]:
        for origin in origins:
            parsed = urlsplit(origin)
            if (
                "*" in origin
                or parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("CORS_ORIGINS must contain explicit http(s) origins without paths or wildcards")
            # Accessing port also validates malformed port numbers.
            _ = parsed.port
        return origins


@lru_cache
def get_settings() -> Settings:
    return Settings()

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Конфигурация приложения. Значения берутся из .env или переменных окружения."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        frozen=True,
    )

    # App
    DEBUG: bool = False

    # DB
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "jobagent"
    DB_PASS: str = "jobagent"
    DB_NAME: str = "jobagent"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # LLM
    LLM_PROVIDER: str = "openai"  # openai | ollama
    LLM_TEXT_MODEL: str = "gpt-4o-mini"
    LLM_VISION_MODEL: str = "gpt-4o"
    OPENAI_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Browser executor
    BROWSER_EXECUTOR_URL: str = "http://localhost:8010"

    # Telegram
    TG_BOT_TOKEN: str = ""

    # Security
    FERNET_KEY: str = ""  # base64-encoded 32-byte key


settings = Settings()

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
    OPENAI_BASE_URL: str = (
        ""  # OpenAI-совместимый base_url (напр. https://api.deepseek.com)
    )
    LLM_EXTRA_BODY: dict = (
        {}
    )  # доп. параметры запроса (напр. DeepSeek: {"thinking":{"type":"disabled"}})
    LLM_STRUCTURED_METHOD: str = (
        "function_calling"  # текст: function_calling | json_mode (DeepSeek не поддерживает json_schema)
    )
    LLM_VISION_STRUCTURED_METHOD: str = (
        "json_mode"  # vision: thinking-режим DeepSeek не поддерживает tools
    )
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_NUM_CTX: int = 8192  # контекст текстовой модели Ollama
    OLLAMA_VISION_NUM_CTX: int = 16384  # контекст vision-модели (скриншоты)
    EMBEDDING_PROVIDER: str = "openai"  # openai | ollama
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIM: int = (
        1536  # openai text-embedding-3-small = 1536; ollama nomic-embed-text = 768
    )

    # Browser executor
    BROWSER_EXECUTOR_URL: str = "http://localhost:8010"
    BROWSER_HEADLESS: bool = False  # False — видимое окно браузера на локальной машине
    BROWSER_MODE: str = (
        "auto"  # persistent | cdp | auto (cdp с фолбэком на свой браузер)
    )
    BROWSER_CDP_URL: str = "http://localhost:9222"
    BROWSER_MAX_STEPS: int = 15  # лимит шагов VLM-цикла браузера
    BROWSER_API_TOKEN: str = ""  # общий секрет browser-сервиса (пусто — без auth)

    # Search
    MAX_CANDIDATES: int = 20  # сколько найденных вакансий скорить (первые N)
    HH_SEARCH_URL: str = ""  # прямой URL результатов поиска hh.ru (с фильтрами)
    RESUMES: dict[str, str] = {}  # имя резюме -> описание (для выбора под вакансию)

    # Telegram
    TG_BOT_TOKEN: str = ""

    # Security
    FERNET_KEY: str = ""  # base64-encoded 32-byte key


settings = Settings()

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Moodly Watch API"
    debug: bool = True

    # LLM (Groq)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Embeddings
    embedding_model: str = "intfloat/multilingual-e5-small"

    # Datos externos
    tmdb_api_key: str = ""


settings = Settings()

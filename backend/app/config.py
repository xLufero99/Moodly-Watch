from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Moodly Watch API"
    debug: bool = True

    # Origen permitido en CORS. En local es el dev server de Vite; en Railway es el
    # dominio de Cloudflare Pages, que se pone como variable de entorno.
    frontend_url: str = "http://localhost:5173"

        # LLM (Groq). El modelo NO es arbitrario: structured outputs con `strict: true` solo
    # funcionan en gpt-oss-20b, gpt-oss-120b y qwen3.8-27b. Con cualquier otro Groq
    # devuelve 400 y el parser se degrada a buscar sin filtros. Comprobado en la doc.
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"

    # Segundos que se espera a Groq antes de rendirse. Un LLM colgado en un request HTTP
    # es peor que no tener LLM: el usuario espera y no obtiene nada.
    groq_timeout: float = 10.0

    # Embeddings
    embedding_model: str = "intfloat/multilingual-e5-small"

    # Datos externos
    tmdb_api_key: str = ""
    mal_client_id: str = ""


settings = Settings()

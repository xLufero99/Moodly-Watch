from typing import Literal

from pydantic import BaseModel, Field, field_validator

MediaType = Literal["movie", "tv", "anime"]


class RecommendRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=500,
        description="Texto de entrada para generar recomendaciones",
    )
    media_types: list[MediaType] = Field(
        default=["movie", "tv", "anime"],
        min_length=1,
        description="Tipos de medios a recomendar",
    )
    liked_ids: list[str] = Field(
        default=[],
        description="IDs de ítems ya vistos/gustados",
    )

    @field_validator("text", mode="before")
    @classmethod
    def _strip_and_validate_text(cls, v: str) -> str:
        if isinstance(v, str):
            v = v.strip()
        if not v:
            raise ValueError("text no puede estar vacío")
        if len(v) > 500:
            raise ValueError("text excede 500 caracteres")
        return v


class Recommendation(BaseModel):
    id: str
    title: str
    media_type: MediaType
    year: int | None = None
    genres: list[str] = Field(default=[])
    poster_url: str | None = None
    score: float = Field(ge=0, le=1)
    # Opcional mientras la explicación la genera Groq. El buscador devuelve None y
    # `ResultCard.jsx` renderiza `{explanation}` sin condición, así que un null pinta un
    # párrafo vacío en vez de romper. Cuando Groq esté, esto pasa a ser obligatorio otra
    # vez y el schema se cierra.
    explanation: str | None = None


class RecommendResponse(BaseModel):
    results: list[Recommendation]

from fastapi import APIRouter, HTTPException

from app.models.schemas import RecommendRequest, RecommendResponse
from app.services.recommendation import IndiceNoDisponibleError, recomendar

router = APIRouter()


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest) -> RecommendResponse:
    """De la frase a las tarjetas con explicación.

    El pipeline entero (parsear filtros, buscar, explicar) vive en
    `app.services.recommendation`, no aquí: este router solo traduce HTTP.

    `IndiceNoDisponibleError` se traduce a un 503 y no a un 500 porque no es un fallo del
    código sino del despliegue: falta `data/index/`, o no se ha construido. Un 500 haría
    pensar que hay que arreglar algo, y lo que hay que hacer es correr
    `scripts.build_index`.
    """
    try:
        cuerpo = recomendar(
            texto=request.text,
            media_types=list(request.media_types),
            liked_ids=list(request.liked_ids),
        )
    except IndiceNoDisponibleError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "El índice no está disponible. Constrúyelo con "
                "`uv run python -m scripts.build_index`."
            ),
        ) from error
    return RecommendResponse(results=cuerpo["results"])
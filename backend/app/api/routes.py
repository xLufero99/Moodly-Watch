from fastapi import APIRouter

from app.models.schemas import RecommendRequest, RecommendResponse
from app.services.mock_recommender import get_mock_recommendations

router = APIRouter()


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest) -> RecommendResponse:
    results = get_mock_recommendations(request)
    return RecommendResponse(results=results)

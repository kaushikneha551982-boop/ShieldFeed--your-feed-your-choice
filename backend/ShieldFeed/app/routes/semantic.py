from fastapi import APIRouter, HTTPException

from app.schemas.semantic_schema import SemanticFilterRequest, SemanticFilterResponse
from app.services import semantic_service

router = APIRouter()


@router.post("/semantic-filter", response_model=SemanticFilterResponse)
def semantic_filter(payload: SemanticFilterRequest):
    if not payload.posts:
        raise HTTPException(status_code=400, detail="posts list cannot be empty")

    try:
        results = semantic_service.score_posts(payload.preference, payload.posts)
    except Exception as exc:
        # Never leak raw model/library stack traces to the client.
        raise HTTPException(
            status_code=500, detail="Semantic filter failed"
        ) from exc

    return SemanticFilterResponse(results=results)

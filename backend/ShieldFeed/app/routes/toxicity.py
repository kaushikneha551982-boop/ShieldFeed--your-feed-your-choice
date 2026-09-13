from fastapi import APIRouter, HTTPException

from app.schemas.toxicity_schema import ToxicityRequest, ToxicityResponse
from app.services import toxicity_service

router = APIRouter()


@router.post("/toxicity", response_model=ToxicityResponse)
def check_toxicity(payload: ToxicityRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    try:
        score, action = toxicity_service.classify(payload.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Toxicity check failed") from exc

    return ToxicityResponse(toxicity_score=score, action=action)

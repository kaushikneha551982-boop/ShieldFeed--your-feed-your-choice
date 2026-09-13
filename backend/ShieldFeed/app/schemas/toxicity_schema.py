from typing import Literal

from pydantic import BaseModel, Field


class ToxicityRequest(BaseModel):
    text: str = Field(..., description="Comment text to check before posting")


class ToxicityResponse(BaseModel):
    toxicity_score: float
    action: Literal["SAFE", "WARN"]

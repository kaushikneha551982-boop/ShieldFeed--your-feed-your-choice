from typing import List, Literal

from pydantic import BaseModel, Field


class PostIn(BaseModel):
    id: int
    text: str


class SemanticFilterRequest(BaseModel):
    # Empty string is allowed on purpose: it means "no preference set yet",
    # and the service treats that as "show everything".
    preference: str = Field(
        default="",
        description="Natural language description of content the user wants to avoid",
    )
    posts: List[PostIn]


class SemanticResult(BaseModel):
    post_id: int
    similarity: float
    action: Literal["SHOW", "WARN", "FILTER"]


class SemanticFilterResponse(BaseModel):
    results: List[SemanticResult]

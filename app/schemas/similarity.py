"""
Pydantic schemas for similarity API
"""
from typing import List, Tuple
from pydantic import BaseModel, HttpUrl, Field


class ImageUrlRequest(BaseModel):
    """Request schema for image URL"""
    url: HttpUrl = Field(..., description="Image URL to analyze")


class SimilarityResponse(BaseModel):
    """Response schema for similarity results"""
    top_k: List[Tuple[str, float]] = Field(
        ...,
        description="List of (class_name, similarity_score) tuples sorted by score descending"
    )
    verdict: str = Field(
        ...,
        description="Final verdict: class name or 'unknown'"
    )
    s1: float = Field(
        ...,
        description="Top similarity score"
    )
    margin: float = Field(
        ...,
        description="Difference between top two scores (s1 - s2)"
    )
    is_unknown: bool = Field(
        ...,
        description="Whether the query should be classified as unknown"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "top_k": [
                    ["피카츄", 0.85],
                    ["라이츄", 0.72],
                    ["파이리", 0.65]
                ],
                "verdict": "피카츄",
                "s1": 0.85,
                "margin": 0.13,
                "is_unknown": False
            }
        }

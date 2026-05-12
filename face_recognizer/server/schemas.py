"""Pydantic response schemas for the face recognition API.

Provides typed models for automatic OpenAPI docs and response validation.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class PeopleResponse(BaseModel):
    names: list[str] = Field(default_factory=list)
    threshold: float = 0.0
    vector_count: int = 0


class DeleteResponse(BaseModel):
    removed: str


class EnrollResponse(BaseModel):
    name: str
    row_id: int
    liveness_score: Optional[float] = None
    quality: float = 0.0
    threshold: float = 0.0


class RecognizeResponse(BaseModel):
    matched: bool
    name: Optional[str] = None
    confidence: Optional[float] = None
    is_real: Optional[bool] = None
    face_bbox: Optional[list[float]] = None
    per_stage_ms: dict[str, float] = Field(default_factory=dict)
    total_ms: float = 0.0


class CalibrateResponse(BaseModel):
    threshold: float
    max_cross_id: float
    margin: float
    identities: int
    vectors: int


class ErrorResponse(BaseModel):
    detail: str

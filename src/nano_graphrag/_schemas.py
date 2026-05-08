from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, Field, field_validator


def _coerce_iso_date(v):
    if not v or not isinstance(v, str):
        return None
    v = v.strip()
    if not v:
        return None
    try:
        return datetime.fromisoformat(v).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        pass
    if re.match(r"^\d{4}$", v):
        return f"{v}-01-01"
    if re.match(r"^\d{4}-\d{2}$", v):
        return f"{v}-01"
    if re.match(r"^\d{4}/\d{2}/\d{2}$", v):
        return v.replace("/", "-")
    return None


class ExtractedEntity(BaseModel):
    entity_name: str = Field(..., alias="name", description="The name of the entity, capitalized")
    entity_type: str = Field(..., alias="type", description="The type of the entity")
    description: str = Field(..., description="Comprehensive description of the entity")
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative names, abbreviations, or nicknames for the entity",
    )
    event_date: str | None = Field(
        default=None,
        description="For EVENT type entities only: the date the event occurred in YYYY-MM-DD format. Must be null for non-event entities.",
    )

    @field_validator("event_date", mode="before")
    @classmethod
    def coerce_event_date(cls, v):
        return _coerce_iso_date(v)

    model_config = {"populate_by_name": True}


class ExtractedRelationship(BaseModel):
    source: str = Field(..., description="Name of the source entity, capitalized")
    target: str = Field(..., description="Name of the target entity, capitalized")
    description: str = Field(..., description="Explanation of the relationship")
    weight: float = Field(default=1.0, description="Strength of the relationship (0-10)")
    temporal_context: str | None = Field(
        default=None,
        description="Free-text description of when this relationship was/is true (e.g. 'in 2023', 'from 2019 to 2022')",
    )
    valid_from: str | None = Field(
        default=None,
        description="Start date in YYYY-MM-DD format when this relationship became true. Null if unknown.",
    )
    valid_to: str | None = Field(
        default=None,
        description="End date in YYYY-MM-DD format when this relationship ceased to be true. Null if still current or unknown.",
    )

    @field_validator("valid_from", "valid_to", mode="before")
    @classmethod
    def coerce_date_fields(cls, v):
        return _coerce_iso_date(v)


class EntityExtractionOutput(BaseModel):
    entities: list[ExtractedEntity] = Field(
        default_factory=list, description="List of extracted entities"
    )
    relationships: list[ExtractedRelationship] = Field(
        default_factory=list, description="List of extracted relationships"
    )


class ChunkExtractionResult(BaseModel):
    chunk_id: str = Field(..., description="Identifier of the source chunk")
    entities: list[ExtractedEntity] = Field(
        default_factory=list, description="Entities extracted from this chunk"
    )
    relationships: list[ExtractedRelationship] = Field(
        default_factory=list, description="Relationships extracted from this chunk"
    )


class BatchedEntityExtractionOutput(BaseModel):
    chunks: list[ChunkExtractionResult] = Field(
        default_factory=list,
        description="Extraction results for each chunk in the batch",
    )


class CommunityReportFinding(BaseModel):
    summary: str = Field(..., description="Short summary of the finding")
    explanation: str = Field(..., description="Detailed explanation grounded in the source data")


class CommunityReportOutput(BaseModel):
    title: str = Field(..., description="Report title representing key entities")
    summary: str = Field(..., description="Executive summary of the community")
    rating: float = Field(..., ge=0, le=10, description="Impact severity rating (0-10)")
    rating_explanation: str = Field(..., description="Explanation of the rating")
    findings: list[CommunityReportFinding] = Field(
        default_factory=list, description="Key insights about the community"
    )


class GlobalMapPoint(BaseModel):
    description: str = Field(..., description="Description of the key point")
    score: int = Field(..., ge=0, le=100, description="Importance score (0-100)")


class GlobalMapOutput(BaseModel):
    points: list[GlobalMapPoint] = Field(default_factory=list, description="List of key points")

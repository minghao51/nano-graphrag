from typing import List

from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    entity_name: str = Field(..., description="The name of the entity, capitalized")
    entity_type: str = Field(..., description="The type of the entity")
    description: str = Field(..., description="Comprehensive description of the entity")


class ExtractedRelationship(BaseModel):
    source: str = Field(..., description="Name of the source entity, capitalized")
    target: str = Field(..., description="Name of the target entity, capitalized")
    description: str = Field(..., description="Explanation of the relationship")
    weight: float = Field(default=1.0, description="Strength of the relationship (0-10)")


class EntityExtractionOutput(BaseModel):
    entities: List[ExtractedEntity] = Field(
        default_factory=list, description="List of extracted entities"
    )
    relationships: List[ExtractedRelationship] = Field(
        default_factory=list, description="List of extracted relationships"
    )


class CommunityReportFinding(BaseModel):
    summary: str = Field(..., description="Short summary of the finding")
    explanation: str = Field(..., description="Detailed explanation grounded in the source data")


class CommunityReportOutput(BaseModel):
    title: str = Field(..., description="Report title representing key entities")
    summary: str = Field(..., description="Executive summary of the community")
    rating: float = Field(..., ge=0, le=10, description="Impact severity rating (0-10)")
    rating_explanation: str = Field(..., description="Explanation of the rating")
    findings: List[CommunityReportFinding] = Field(
        default_factory=list, description="Key insights about the community"
    )


class GlobalMapPoint(BaseModel):
    description: str = Field(..., description="Description of the key point")
    score: int = Field(..., ge=0, le=100, description="Importance score (0-100)")


class GlobalMapOutput(BaseModel):
    points: List[GlobalMapPoint] = Field(default_factory=list, description="List of key points")

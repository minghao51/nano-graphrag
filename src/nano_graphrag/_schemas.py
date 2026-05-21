from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

RELATION_VOCABULARY = {
    "part_of",
    "contains",
    "parent_organization_of",
    "subsidiary_of",
    "member_of",
    "located_in",
    "headquartered_in",
    "operates_in",
    "originates_from",
    "created_by",
    "authored_by",
    "founded_by",
    "managed_by",
    "led_by",
    "developed_by",
    "influences",
    "cites",
    "builds_on",
    "extends",
    "contradicts",
    "supports",
    "references",
    "precedes",
    "causes",
    "enables",
    "prevents",
    "uses",
    "depends_on",
    "produces",
    "consumes",
    "implements",
    "provides",
    "employed_by",
    "collaborates_with",
    "works_on",
    "invests_in",
    "competes_with",
    "instance_of",
    "has_characteristic",
    "classified_as",
    "related_to",
}

RELATION_ALIASES: dict[str, str] = {
    "is_part_of": "part_of",
    "is_member_of": "member_of",
    "works_for": "employed_by",
    "headquartered_at": "headquartered_in",
    "located_at": "located_in",
    "authored": "authored_by",
    "written_by": "authored_by",
    "founded": "founded_by",
    "developed": "developed_by",
    "built_on": "builds_on",
    "is_instance_of": "instance_of",
    "has_property": "has_characteristic",
    "uses_tool": "uses",
    "uses_technology": "uses",
    "invests": "invests_in",
    "competes": "competes_with",
    "collaborates": "collaborates_with",
}

_BANNED_RELATION_TYPES = {
    "related",
    "associated_with",
    "connected_to",
    "associated",
    "linked_to",
    "linked",
    "similar_to",
}


def normalize_relation_type(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return "related_to"
    cleaned = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if cleaned in _BANNED_RELATION_TYPES:
        return "related_to"
    if cleaned in RELATION_ALIASES:
        return RELATION_ALIASES[cleaned]
    if cleaned in RELATION_VOCABULARY:
        return cleaned
    return "related_to"


RELATION_VOCABULARY_SORTED = sorted(RELATION_VOCABULARY - {"related_to"})


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
    relation_type: str = Field(
        default="related_to",
        description="Typed relationship from the curated vocabulary",
    )
    weight: float = Field(default=1.0, description="Strength of the relationship (0-10)")
    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Extraction confidence 0.0-1.0"
    )
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

    @field_validator("relation_type", mode="before")
    @classmethod
    def normalize_relation(cls, v):
        return normalize_relation_type(v)

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


# =============================================================================
# Structured result types for public API
# =============================================================================


@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float | None = None


@dataclass
class QuerySource:
    source_type: Literal["entity", "community", "chunk"]
    id: str
    name: str | None = None
    relevance_score: float | None = None
    text_snippet: str | None = None


@dataclass
class QueryResult:
    answer: str
    mode: str
    sources: list[QuerySource] = field(default_factory=list)
    tokens_used: TokenUsage | None = None
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return self.answer

    def __bool__(self) -> bool:
        return bool(self.answer)


@dataclass
class InsertResult:
    documents_processed: int
    documents_skipped: int = 0
    entities_created: int = 0
    relationships_created: int = 0
    communities_updated: int = 0
    tokens_used: TokenUsage | None = None
    latency_ms: float = 0.0


@dataclass
class QueryTrace:
    entities_matched: list[str] = field(default_factory=list)
    communities_used: list[str] = field(default_factory=list)
    chunks_used: list[str] = field(default_factory=list)
    retrieval_scores: dict[str, float] = field(default_factory=dict)
    mode_specific: dict = field(default_factory=dict)


@dataclass
class StreamTextChunk:
    text: str

    def __str__(self) -> str:
        return self.text


@dataclass
class StreamSourceRef:
    sources: list[QuerySource] = field(default_factory=list)


@dataclass
class StreamComplete:
    trace: QueryTrace | None = None
    tokens: TokenUsage | None = None
    latency_ms: float = 0.0

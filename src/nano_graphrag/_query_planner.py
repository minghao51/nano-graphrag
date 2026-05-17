from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ._utils import logger


class QueryEntity(BaseModel):
    name: str = Field(description="Entity name")
    aliases: list[str] = Field(default_factory=list, description="Alternative names/aliases")


class QueryAnalysisOutput(BaseModel):
    entities: list[QueryEntity] = Field(
        default_factory=list,
        description="List of entities mentioned in the query",
    )
    relation_clues: list[str] = Field(
        default_factory=list,
        description="Relation types or connection clues implied by the query",
    )
    constraints: dict[str, str] = Field(
        default_factory=dict,
        description="Temporal, spatial, or other constraints (e.g. {'temporal': '2023'})",
    )
    answer_type: str | None = Field(
        default=None,
        description="Expected answer type (person, organization, date, number, location, etc.)",
    )
    pseudo_queries: list[dict] = Field(
        default_factory=list,
        description="Alternative phrasings of the query with confidence scores",
    )


@dataclass
class QueryAnalysis:
    entities: list[tuple[str, list[str]]] = field(default_factory=list)
    relation_clues: list[str] = field(default_factory=list)
    constraints: dict[str, str] = field(default_factory=dict)
    answer_type: str | None = None
    pseudo_queries: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class SoftEntityMask:
    scores: dict[str, float] = field(default_factory=dict)
    signals: dict[str, list[tuple[str, float]]] = field(default_factory=dict)


QUERY_ANALYSIS_PROMPT = """Analyze the following question and extract structured information.

Question: {query}

Extract:
1. Named entities mentioned in the question (people, organizations, locations, etc.)
2. Relationship clues or connection types implied
3. Any temporal or spatial constraints
4. The expected answer type
5. Alternative phrasings (pseudo-queries) that would retrieve the same information

Return a JSON object with exact fields:
- entities: list of {{"name": str, "aliases": list[str]}}
- relation_clues: list of str
- constraints: dict of str to str
- answer_type: str or null
- pseudo_queries: list of {{"query": str, "confidence": float}}
"""


async def _analyze_query(
    query: str,
    global_config: dict,
) -> QueryAnalysis | None:
    use_model = global_config.get("cheap_model_func")
    if use_model is None:
        return None
    prompt = QUERY_ANALYSIS_PROMPT.format(query=query)
    try:
        result = await use_model(
            prompt,
            response_format=QueryAnalysisOutput,
        )
        if isinstance(result, str):
            import json

            result = QueryAnalysisOutput(**json.loads(result))
        if not isinstance(result, QueryAnalysisOutput):
            return None
        entities = [(e.name, e.aliases) for e in result.entities if e.name]
        pseudo_queries = [
            (pq["query"], pq.get("confidence", 0.5))
            for pq in result.pseudo_queries
            if pq.get("query")
        ]
        return QueryAnalysis(
            entities=entities,
            relation_clues=result.relation_clues,
            constraints=result.constraints,
            answer_type=result.answer_type,
            pseudo_queries=pseudo_queries,
        )
    except Exception as e:
        logger.warning("query_analysis_failed", error=str(e))
        return None


def build_soft_entity_mask(
    analysis: QueryAnalysis,
    all_entity_names: dict[str, str],
    entity_name_to_id: dict[str, str],
    entity_id_to_aliases: dict[str, list[str]],
) -> SoftEntityMask:
    mask = SoftEntityMask()
    if not analysis.entities:
        return mask

    reverse_alias_lookup: dict[str, str] = {}
    for eid, aliases in entity_id_to_aliases.items():
        for a in aliases:
            reverse_alias_lookup[a.lower()] = eid
        ename = all_entity_names.get(eid, "")
        if ename:
            reverse_alias_lookup[ename.lower()] = eid

    for name, aliases in analysis.entities:
        name_lower = name.lower().strip()
        best_id = entity_name_to_id.get(name_lower)
        if best_id:
            mask.scores[best_id] = max(mask.scores.get(best_id, 0.0), 1.0)
            mask.signals.setdefault(best_id, []).append(("exact", 1.0))
            continue
        for alias in aliases:
            alias_lower = alias.lower().strip()
            best_id = entity_name_to_id.get(alias_lower)
            if best_id:
                mask.scores[best_id] = max(mask.scores.get(best_id, 0.0), 0.8)
                mask.signals.setdefault(best_id, []).append(("alias", 0.8))
                break
        else:
            match_eid = reverse_alias_lookup.get(name_lower)
            if match_eid:
                mask.scores[match_eid] = max(mask.scores.get(match_eid, 0.0), 0.6)
                mask.signals.setdefault(match_eid, []).append(("ner", 0.6))
    for pq, confidence in analysis.pseudo_queries:
        for eid, ename in all_entity_names.items():
            if pq.lower() in ename.lower() or ename.lower() in pq.lower():
                score = 0.4 * confidence
                mask.scores[eid] = max(mask.scores.get(eid, 0.0), score)
                mask.signals.setdefault(eid, []).append(("pseudo_query", score))
    return mask


def compute_final_scores(
    vector_results: list[dict],
    soft_mask: SoftEntityMask,
    vector_weight: float = 0.6,
    mask_weight: float = 0.4,
) -> list[tuple[dict, float]]:
    scored = []
    for r in vector_results:
        eid = r["id"]
        vec_score = r.get("similarity", r.get("distance", 0.0))
        if not isinstance(vec_score, int | float):
            vec_score = 1.0 - r.get("distance", 0.0) if "distance" in r else 0.5
        mask_score = soft_mask.scores.get(eid, 0.0)
        combined = vector_weight * vec_score + mask_weight * mask_score
        scored.append((r, combined))
    for eid, mask_score in soft_mask.scores.items():
        if mask_score > 0 and not any(r["id"] == eid for r in vector_results):
            scored.append(
                ({"id": eid, "entity_name": "", "similarity": 0.0}, mask_weight * mask_score)
            )
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

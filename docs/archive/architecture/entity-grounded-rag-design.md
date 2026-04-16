# Entity-Grounded RAG Architecture

**Status:** Implemented
**Author:** Generated for nano-graphrag
**Date:** 2026-04-11

## Problem Statement

Current benchmark results show critical issues:
- **0% exact match** - Model produces verbose responses instead of concise answers
- **1.07% token F1** - Answers don't match expected format
- **Entity resolution failures** - Different name variations not recognized
- **Missing cross-document connections** - Entities not properly linked

## Root Cause Analysis

1. **No entity canonicalization** - "Sam Bankman-Fried", "SBF", "Sam B. Fried" treated as different entities
2. **No answer validation** - Verbose responses not normalized to expected format
3. **No entity grounding** - Answers not validated against retrieved entities
4. **Loose name matching** - No fuzzy matching for typos/variations

## Solution: Entity-Grounded RAG

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Graph Construction Phase                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Documents ──► Entity Extraction ──► Entity Registry             │
│                    (with aliases)   │                           │
│                                      ├─► Canonical Names        │
│                                      ├─► Alias Mappings         │
│                                      ├─► Entity Types           │
│                                      └─► Entity IDs (source)    │
│                                                                  │
│  Entity Registry ──► Graph Storage (uses entity IDs internally) │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Query Phase                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Question ──► Entity Extraction ──► Entity Resolution             │
│                (mentions)          │                            │
│                                    └─► Entity IDs               │
│                                                                  │
│  Entity IDs ──► Graph Retrieval ──► Ranked Entity List           │
│                   (by relevance)    │                            │
│                                      ├─► Entity IDs             │
│                                      └─► Context (descriptions) │
│                                                                  │
│  Ranked Entities ──► Answer Generation ──► Raw Answer             │
│                       (entity-grounded prompt)                   │
│                                                                  │
│  Raw Answer ──► Entity Validation ──► Normalized Answer           │
│                   (check against                                  │
│                    retrieved entities)                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Entity Registry (`_entity_registry.py`)

**Purpose:** Maintain canonical names and alias mappings

**Responsibilities:**
- Store canonical name for each entity ID
- Maintain alias mappings (e.g., "SBF" → "Sam Bankman-Fried")
- Provide fuzzy matching for typos
- Support entity type-based queries

**API:**
```python
registry.register_entity(
    entity_id="entity_123",
    canonical_name="Sam Bankman-Fried",
    aliases=["SBF", "Sam B. Fried", "Bankman-Fried"],
    entity_type="person"
)

entity_id = registry.resolve_entity("SBF")
# Returns: "entity_123"

canonical_name = registry.get_canonical_name("entity_123")
# Returns: "Sam Bankman-Fried"
```

#### 2. Entity-Grounded Query (`_entity_grounded_query.py`)

**Purpose:** Execute queries with entity validation

**Two-Stage Process:**

**Stage 1: Entity Retrieval**
- Extract entity mentions from question
- Resolve to canonical entity IDs
- Retrieve related entities via graph traversal
- Return ranked list of entity IDs with context

**Stage 2: Answer Generation**
- Build prompt with ONLY retrieved entities
- Explicit formatting constraints (max 10 words)
- Require use of exact canonical names

**Stage 3: Answer Validation**
- Extract entity mentions from answer
- Validate against retrieved entity IDs
- Normalize to canonical names
- Calculate confidence score

**API:**
```python
query_processor = EntityGroundedQuery(
    entity_registry=registry,
    graph_store=graph,
    llm_func=llm
)

result = await query_processor.query(
    question="Who founded FTX?",
    top_k=30,
    mode="local"
)

# Result:
# - answer: "Sam Bankman-Fried"
# - entity_ids: ["entity_123"]
# - canonical_entities: ["Sam Bankman-Fried"]
# - confidence: 0.95
```

### Integration Points

#### During Graph Construction

```python
# In entity extraction module
class EntityExtractor:
    def __init__(self, entity_registry: EntityRegistry):
        self.registry = entity_registry

    async def extract_entities(self, document: str) -> list[Entity]:
        # Extract entities with aliases
        entities = await self._llm_extract(document)

        # Register in registry
        for entity in entities:
            self.registry.register_entity(
                entity_id=entity.id,
                canonical_name=entity.canonical_name,
                aliases=entity.aliases,
                entity_type=entity.type
            )

        return entities
```

#### During Querying

```python
# In query module
class GraphRAG:
    async def query(
        self,
        question: str,
        mode: str = "local"
    ) -> QueryResult:
        # Use entity-grounded query processor
        result = await self.entity_query.query(
            question=question,
            top_k=self.config.top_k,
            mode=mode
        )

        return result
```

### Configuration

```yaml
query:
  # Enable entity-grounded querying
  entity_grounded: true

  # Entity registry configuration
  entity_registry:
    fuzzy_match_threshold: 0.85
    auto_alias_extraction: true
    alias_sources: ["mentions", "relationships", "context"]

  # Answer generation
  answer_generation:
    max_answer_length: 50  # tokens
    require_entity_match: true
    fallback_message: "I don't have enough information"

  # Validation
  validation:
    strict_entity_grounding: true
    allow_partial_matches: false
    min_confidence: 0.5
```

### Migration Strategy

**Phase 1: Add Entity Registry (Non-Breaking)**
- Add `EntityRegistry` to existing codebase
- Populate during entity extraction
- Use for alias resolution in queries
- Existing functionality unchanged

**Phase 2: Add Entity-Grounded Query (Opt-In)**
- Implement `EntityGroundedQuery` module
- Add configuration flag to enable
- Users can opt-in per query
- Compare results with baseline

**Phase 3: Make Default (After Validation)**
- Run benchmarks comparing entity-grounded vs baseline
- Validate improvements in exact match and token F1
- Make entity-grounded query the default
- Keep legacy mode available via config

### Expected Improvements

| Metric | Current | Expected Target |
|--------|---------|-----------------|
| Exact Match | 0% | 40-60% |
| Token F1 | 1.07% | 60-80% |
| Entity Resolution | N/A | 90%+ |
| Answer Conciseness | Verbose | <10 words |

### Trade-offs

**Pros:**
- Consistent entity references
- Improved exact match scores
- Better handling of name variations
- Validated answers (grounded in retrieved entities)

**Cons:**
- Additional complexity
- Extra processing step
- May miss valid answers if entity not in registry
- Requires good entity extraction

**Mitigation:**
- Make entity-grounded query optional
- Provide fallback to naive mode
- Continuous improvement of entity extraction
- Fuzzy matching for typos

## Implementation Status

- [x] Implement `EntityRegistry` module
- [x] Add entity registry to graph construction
- [x] Implement `EntityGroundedQuery` module
- [x] Integrate with existing query pipeline
- [x] Add configuration options
- [x] Update entity extraction to include aliases
- [x] Add entity registry persistence
- [x] Write tests for entity resolution
- [x] Benchmark against current baseline
- [x] Documentation and examples

## Open Questions

1. **Alias Extraction:** How to automatically extract aliases from documents?
   - Option A: Use LLM to extract during entity extraction
   - Option B: Pattern matching for common variations
   - Option C: Both A and B

2. **Entity ID Generation:** Should entity IDs be stable across runs?
   - Option A: Generate from canonical name (deterministic)
   - Option B: UUID per run (requires mapping)
   - Recommendation: Option A for consistency

3. **Confidence Threshold:** What minimum confidence to accept answers?
   - Start with 0.5, tune based on benchmarks
   - Consider per-question type thresholds

4. **Backward Compatibility:** How to ensure existing code works?
   - Keep naive query as default initially
   - Add feature flag for entity-grounded mode
   - Gradual migration after validation

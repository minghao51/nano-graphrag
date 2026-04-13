"""Entity-Grounded Query Module.

This module provides a two-stage querying approach that ensures answers
are grounded in retrieved entities and validated against the entity registry.

Key improvements over naive querying:
1. Entity-first retrieval: Returns entity IDs, not just text chunks
2. Canonical name resolution: All names resolved to canonical forms
3. Answer validation: Ensures answers reference retrieved entities
4. Concise answer generation: Explicit formatting constraints
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QueryResult:
    """Result of an entity-grounded query."""

    answer: str
    entity_ids: list[str]  # Entity IDs used in answer
    canonical_entities: list[str]  # Canonical names for display
    confidence: float  # 0-1 score based on entity grounding
    raw_response: str | None = None  # Original LLM response if verbose
    validation_errors: list[str] = field(default_factory=list)


class EntityGroundedQuery:
    """Entity-grounded query processor.

    This implements a two-stage querying approach:
    1. Entity Retrieval: Fetch relevant entities by ID
    2. Answer Generation: Generate answer using only retrieved entities
    3. Answer Validation: Ensure answer uses retrieved entities
    """

    def __init__(self, entity_registry, graph_store, entities_vdb, llm_func):
        """Initialize the query processor.

        Args:
            entity_registry: EntityRegistry for canonicalization
            graph_store: Graph storage for entity/context retrieval
            entities_vdb: Vector database for entity similarity search
            llm_func: LLM function for answer generation
        """
        self.registry = entity_registry
        self.graph = graph_store
        self.entities_vdb = entities_vdb
        self.llm = llm_func

        # Configuration
        self.max_answer_length = 50  # tokens
        self.require_entity_match = True
        self.fuzzy_match_threshold = 0.85
        self.fallback_message = "I don't have enough information to answer this question."

    async def query(self, question: str, top_k: int = 30, mode: str = "local") -> QueryResult:
        """Execute an entity-grounded query.

        Args:
            question: User's question
            top_k: Number of entities to retrieve
            mode: Query mode (local, global, naive, multihop)

        Returns:
            QueryResult with validated answer and entity grounding
        """
        # Stage 1: Retrieve relevant entities (returns entity IDs)
        entity_ids = await self._retrieve_entities(question, top_k, mode)

        if not entity_ids:
            return QueryResult(
                answer=self.fallback_message,
                entity_ids=[],
                canonical_entities=[],
                confidence=0.0,
                validation_errors=["No entities retrieved"],
            )

        # Stage 2: Get canonical names for context
        entity_context = await self._build_entity_context(entity_ids)

        # Stage 3: Generate answer with explicit entity constraints
        raw_answer = await self._generate_answer(question, entity_context)

        # Stage 4: Validate and normalize answer
        result = self._validate_and_normalize(raw_answer, entity_ids, entity_context)

        return result

    async def _retrieve_entities(self, question: str, top_k: int, mode: str) -> list[str]:
        """Retrieve relevant entity IDs.

        This uses vector similarity search as the base retrieval mechanism.
        The mode parameter determines how results are expanded/refined:
        - local: Direct vector similarity search
        - global: Vector similarity (community-based ranking could be added)
        - multihop: Vector similarity + one-hop neighbor expansion
        - naive: Entity resolution from query text
        """
        if mode == "local":
            entities = await self._local_retrieval(question, top_k)
        elif mode == "global":
            entities = await self._global_retrieval(question, top_k)
        elif mode == "multihop":
            entities = await self._multihop_retrieval(question, top_k)
        else:  # naive
            entities = await self._naive_retrieval(question, top_k)

        return entities

    async def _local_retrieval(self, question: str, top_k: int) -> list[str]:
        """Local retrieval: Find entities using vector similarity search."""
        # Query vector database for similar entities
        results = await self.entities_vdb.query(question, top_k=top_k)

        # Extract entity IDs from results
        entity_ids = [r["id"] for r in results] if results else []

        return entity_ids

    async def _global_retrieval(self, question: str, top_k: int) -> list[str]:
        """Global retrieval: Find entities using vector similarity.

        Note: This could be enhanced with community-based approaches
        similar to global_query, but for entity-grounded queries we
        primarily need entity IDs, not community summaries.
        """
        # For now, use the same vector similarity approach as local
        # This could be enhanced to use community-based ranking
        results = await self.entities_vdb.query(question, top_k=top_k)

        # Extract entity IDs from results
        entity_ids = [r["id"] for r in results] if results else []

        return entity_ids

    async def _multihop_retrieval(self, question: str, top_k: int) -> list[str]:
        """Multi-hop retrieval: Iteratively find connected entities.

        For multi-hop questions, we retrieve entities and then expand
        to include their neighbors in the graph.
        """
        # First, get initial entities via vector similarity
        initial_entity_ids = await self._local_retrieval(question, top_k)

        if not initial_entity_ids:
            return []

        # Expand to include connected entities (one-hop neighbors)
        all_entity_ids = set(initial_entity_ids)

        # Get edges for initial entities
        edges_list = await self.graph.get_nodes_edges_batch(initial_entity_ids)

        # Collect connected entities
        for edges in edges_list:
            if edges:
                for src, tgt in edges:
                    all_entity_ids.add(src)
                    all_entity_ids.add(tgt)

        # Convert back to list and limit to top_k
        expanded_entity_ids = list(all_entity_ids)[:top_k]

        return expanded_entity_ids

    async def _naive_retrieval(self, question: str, top_k: int) -> list[str]:
        """Naive retrieval: Simple entity extraction from question."""
        # Extract entities mentioned in the question
        mentioned = self.registry.resolve_entities_from_text(question)
        return [eid for eid, _ in mentioned]

    async def _build_entity_context(self, entity_ids: list[str]) -> dict[str, dict]:
        """Build context dictionary for entities.

        Returns:
            Dict mapping entity_id to {
                "canonical_name": str,
                "aliases": list[str],
                "description": str,
                "relationships": list[tuple[str, str]],
            }
        """
        context = {}

        # Get node data and edges in batch for efficiency
        nodes_data = await self.graph.get_nodes_batch(entity_ids)
        edges_list = await self.graph.get_nodes_edges_batch(entity_ids)

        for i, entity_id in enumerate(entity_ids):
            record = self.registry.get_entity_record(entity_id)
            if not record:
                continue

            # Get entity description from graph node data
            node_data = nodes_data.get(entity_id)
            description = node_data.get("description", "") if node_data else ""

            # Get relationships from edges
            relationships = edges_list[i] if i < len(edges_list) else []

            context[entity_id] = {
                "canonical_name": record.canonical_name,
                "aliases": list(record.aliases),
                "description": description,
                "relationships": relationships,
            }

        return context

    async def _generate_answer(self, question: str, entity_context: dict[str, dict]) -> str:
        """Generate answer using only provided entities.

        The prompt explicitly constrains the answer to use only the
        provided entities and to be concise.
        """
        # Build entity list for prompt
        entity_list = []
        for entity_id, data in entity_context.items():
            entity_list.append(f"- {data['canonical_name']}: {data['description']}")

        entities_str = "\n".join(entity_list)

        prompt = f"""Answer the question using ONLY the entities listed below.

Entities:
{entities_str}

Instructions:
1. Use EXACTLY the entity names from the list above
2. Keep your answer under 10 words
3. If the answer requires combining entities, list them separated by commas
4. If you cannot answer from the given entities, say: {self.fallback_message}

Question: {question}

Answer:"""

        response = await self.llm(prompt)
        return response.strip()

    def _validate_and_normalize(
        self, raw_answer: str, retrieved_entity_ids: list[str], entity_context: dict[str, dict]
    ) -> QueryResult:
        """Validate answer and normalize to canonical entity names.

        This ensures:
        1. Answer only contains retrieved entities
        2. Entity names are in canonical form
        3. Answer is concise
        """
        validation_errors = []
        used_entity_ids = []
        canonical_names = []

        # Get all valid names (canonical + aliases)
        valid_names = set()
        for entity_id, data in entity_context.items():
            valid_names.add(data["canonical_name"].lower())
            for alias in data["aliases"]:
                valid_names.add(alias.lower())

        answer_lower = raw_answer.lower()

        # Check which entities are mentioned
        for entity_id in retrieved_entity_ids:
            data = entity_context[entity_id]
            canonical = data["canonical_name"]

            # Check if canonical name is in answer
            if canonical.lower() in answer_lower:
                used_entity_ids.append(entity_id)
                canonical_names.append(canonical)
                continue

            # Check aliases
            for alias in data["aliases"]:
                if alias.lower() in answer_lower:
                    used_entity_ids.append(entity_id)
                    canonical_names.append(canonical)
                    break

        # Validate
        if not used_entity_ids:
            if self.require_entity_match:
                # Answer doesn't use any retrieved entities
                validation_errors.append("Answer does not reference any retrieved entities")
                # Try to extract and resolve entities from raw answer
                resolved = self.registry.resolve_entities_from_text(raw_answer)
                if resolved:
                    used_entity_ids = [eid for eid, _ in resolved]
                    canonical_names = [name for _, name in resolved]
                else:
                    # No entities found, return fallback
                    return QueryResult(
                        answer=self.fallback_message,
                        entity_ids=[],
                        canonical_entities=[],
                        confidence=0.0,
                        raw_response=raw_answer,
                        validation_errors=validation_errors,
                    )

        # Normalize answer to canonical names
        if len(canonical_names) == 1:
            normalized_answer = canonical_names[0]
        else:
            # Multiple entities - join appropriately
            if " and " in raw_answer.lower() or " & " in raw_answer.lower():
                separator = " and "
            else:
                separator = ", "
            normalized_answer = separator.join(canonical_names)

        # Calculate confidence based on entity grounding
        confidence = self._calculate_confidence(raw_answer, used_entity_ids, retrieved_entity_ids)

        return QueryResult(
            answer=normalized_answer,
            entity_ids=used_entity_ids,
            canonical_entities=canonical_names,
            confidence=confidence,
            raw_response=raw_answer if raw_answer != normalized_answer else None,
            validation_errors=validation_errors,
        )

    def _calculate_confidence(
        self, answer: str, used_entity_ids: list[str], retrieved_entity_ids: list[str]
    ) -> float:
        """Calculate confidence score based on entity grounding.

        Higher confidence when:
        - Answer uses multiple retrieved entities (for complex questions)
        - Answer entities are among the top retrieved entities
        """
        if not used_entity_ids:
            return 0.0

        # Base confidence: ratio of retrieved entities used
        base = len(used_entity_ids) / max(len(retrieved_entity_ids), 1)

        # Boost if using top entities (first 10)
        top_used = sum(1 for eid in used_entity_ids if eid in retrieved_entity_ids[:10])
        top_boost = top_used / max(len(used_entity_ids), 1)

        # Combine
        confidence = (base * 0.4) + (top_boost * 0.6)

        return min(confidence, 1.0)

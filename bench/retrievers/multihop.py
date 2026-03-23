"""Multi-hop retrieval implementation for complex queries."""

from dataclasses import dataclass, field
from nano_graphrag import GraphRAG
from nano_graphrag.base import QueryParam
from .base import HopState


@dataclass
class MultiHopRetriever:
    """Iterative graph retriever for multi-hop questions.

    Algorithm:
      1. Decompose the original question into sub-questions via LLM.
      2. For each sub-question, run local GraphRAG retrieval seeded with
         entities discovered in previous hops.
      3. Accumulate entity state across hops (bridging chain).
      4. Merge all retrieved context, deduplicate, and rank.
    """

    max_hops: int = 4
    entities_per_hop: int = 10
    context_token_budget: int = 8000
    decompose_model: str = "gpt-4o-mini"

    async def retrieve(self, question: str, graph_rag: GraphRAG) -> str:
        """Retrieve context for a multi-hop question.

        Args:
            question: User's multi-hop question
            graph_rag: GraphRAG instance

        Returns:
            Merged context from all hops
        """
        # Step 1: Decompose
        sub_questions = await self._decompose(question, graph_rag)

        # Step 2 & 3: Iterative hop retrieval
        hop_states: list[HopState] = []
        carry_entities: list[str] = []

        for sub_q in sub_questions[:self.max_hops]:
            state = HopState(sub_question=sub_q)
            context = await self._retrieve_hop(
                sub_q, graph_rag, seed_entities=carry_entities
            )
            state.context_chunks = context["chunks"]
            state.retrieved_entities = context["entities"]
            carry_entities = state.retrieved_entities
            hop_states.append(state)

        # Step 4: Merge contexts
        return self._merge_contexts(hop_states, self.context_token_budget)

    async def _decompose(self, question: str, graph_rag: GraphRAG) -> list[str]:
        """Decompose question into sub-questions."""
        prompt = self._build_decompose_prompt(question)
        # Use best_model_func if available, otherwise cheap_model_func
        if graph_rag.best_model_func is not None:
            response = await graph_rag.best_model_func(prompt)
        elif graph_rag.cheap_model_func is not None:
            response = await graph_rag.cheap_model_func(prompt)
        else:
            raise ValueError("No model function available for query decomposition")
        return self._parse_sub_questions(response)

    async def _retrieve_hop(
        self, sub_q: str, graph_rag: GraphRAG, seed_entities: list[str]
    ) -> dict:
        """Retrieve context for a single hop."""
        param = QueryParam(
            mode="local",
            only_need_context=True,
            top_k=self.entities_per_hop,
        )
        context_str = await graph_rag.aquery(sub_q, param=param)
        return self._parse_context(context_str)

    def _merge_contexts(self, hop_states: list[HopState], budget: int) -> str:
        """Merge and deduplicate contexts from all hops."""
        seen, merged = set(), []
        for state in reversed(hop_states):
            for chunk in state.context_chunks:
                h = hash(chunk[:80])
                if h not in seen:
                    seen.add(h)
                    merged.append(chunk)

        # Trim to token budget
        out, total = [], 0
        for chunk in merged:
            tokens = len(chunk) // 4
            if total + tokens > budget:
                break
            out.append(chunk)
            total += tokens

        return "\n\n".join(out)

    def _build_decompose_prompt(self, question: str) -> str:
        """Build prompt for query decomposition."""
        return f"""You are decomposing a complex question into simpler sub-questions that,
when answered in sequence, build toward answering the original question.

Original question: {question}

Generate at most {self.max_hops} sub-questions. Each sub-question should be
independently answerable and together they should bridge to the final answer.
Output only a JSON array of strings.
"""

    def _parse_sub_questions(self, response: str) -> list[str]:
        """Parse LLM response into list of sub-questions."""
        import json

        try:
            questions = json.loads(response)
            if isinstance(questions, list):
                return [str(q) for q in questions]
            return [response]
        except json.JSONDecodeError:
            # Fallback: split by newlines
            return [line.strip() for line in response.split("\n") if line.strip()]

    def _parse_context(self, context_str: str) -> dict:
        """Parse context string into chunks and entities."""
        # Simple implementation - can be enhanced
        return {
            "chunks": [context_str],
            "entities": [],
        }

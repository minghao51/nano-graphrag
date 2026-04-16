"""Adaptive mode router for automatic retrieval mode selection.

Uses heuristic keyword matching and optional LLM fallback to route queries
to the most appropriate retrieval mode: local, global, or multihop.

Routing Rules:
- Multi-hop signals: "who.*also", "both.*and", "connection", "relationship",
  "compared to", "in common"
- Global signals: "themes", "overall", "summarize", "main ideas"
- Default: local

Example:
    router = AdaptiveRouter(use_llm_fallback=False)
    mode = router.route("Who is also connected to both X and Y?")
    # Returns: "multihop"
"""

from __future__ import annotations

import re
from typing import Any, List

from nano_graphrag import GraphRAG
from nano_graphrag.base import QueryParam


class AdaptiveRouter:
    """Automatically select the best retrieval mode for a query.

    Uses a two-stage routing approach:
    1. Fast heuristic keyword matching (default)
    2. Optional LLM classifier for ambiguous queries

    Args:
        use_llm_fallback: If True, use LLM to classify ambiguous queries.
            If False, always use heuristic routing (faster).
        llm_fallback_threshold: Minimum heuristic score to skip LLM.
            Defaults to 1.0 (only use LLM if no heuristic matches).

    Attributes:
        use_llm_fallback: Whether to use LLM for ambiguous queries.
        multihop_patterns: Regex patterns for multi-hop queries.
        global_patterns: Regex patterns for global queries.
    """

    # Multi-hop query patterns
    MULTIHOP_PATTERNS: List[str] = [
        r"\bwho.*also\b",
        r"\bboth.*and\b",
        r"\bconnect(?:ion|ed|s)?\b",
        r"\brelationship\b",
        r"\bcompared?\b",
        r"\bin common\b",
        r"\bbetween\b",
        r"\brelated to\b",
        r"\bassociate with\b",
    ]

    # Global query patterns
    GLOBAL_PATTERNS: List[str] = [
        r"\bthemes?\b",
        r"\boverall\b",
        r"\bin general\b",
        r"\bsummariz(?:e|ing)\b",
        r"\bacross\b",
        r"\bmain ideas?\b",
        r"\bhigh-level\b",
        r"\bbroad view\b",
    ]

    def __init__(self, use_llm_fallback: bool = False, llm_fallback_threshold: float = 1.0) -> None:
        self._use_llm_fallback = use_llm_fallback
        self._llm_fallback_threshold = llm_fallback_threshold

        # Compile patterns for performance
        self._multihop_regex = [re.compile(p, re.IGNORECASE) for p in self.MULTIHOP_PATTERNS]
        self._global_regex = [re.compile(p, re.IGNORECASE) for p in self.GLOBAL_PATTERNS]

    def route(self, question: str) -> str:
        """Route a question to the appropriate retrieval mode.

        Args:
            question: User question to route.

        Returns:
            One of "local", "global", or "multihop".
        """
        # Count heuristic matches
        multihop_score = sum(1 for pattern in self._multihop_regex if pattern.search(question))
        global_score = sum(1 for pattern in self._global_regex if pattern.search(question))

        # Determine routing based on scores
        if multihop_score >= self._llm_fallback_threshold:
            return "multihop"
        if global_score >= self._llm_fallback_threshold:
            return "global"

        # If LLM fallback is enabled and no clear heuristic match, use LLM
        if self._use_llm_fallback and multihop_score < 1 and global_score < 1:
            return self._llm_route(question)

        # Default to local
        return "local"

    async def __call__(
        self,
        query: str,
        graph_rag: GraphRAG,
        param: QueryParam,
        **kwargs,
    ) -> str:
        """Route query and retrieve using the selected mode.

        Args:
            query: User question.
            graph_rag: GraphRAG instance.
            param: Query parameters.
            **kwargs: Additional parameters.

        Returns:
            Retrieved context as string.
        """
        # Determine the best mode
        mode = self.route(query)

        # Get the appropriate retriever from registry
        from bench.registry import resolve

        retriever_class = resolve("retriever", mode)
        retriever = retriever_class()

        # Retrieve using the selected mode
        return await retriever(query, graph_rag, param, **kwargs)

    def _llm_route(self, question: str) -> str:
        """Route using LLM classification.

        This is only called if use_llm_fallback is True and no clear
        heuristic match was found.

        Args:
            question: User question.

        Returns:
            One of "local", "global", or "multihop".
        """
        # For now, just return "local" as the default
        # This can be enhanced later to actually call an LLM
        return "local"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AdaptiveRouter":
        """Create router from configuration dict.

        Args:
            config: Configuration dict with keys:
                - use_llm_fallback (bool): Whether to use LLM for ambiguous queries
                - llm_fallback_threshold (float): Minimum score to skip LLM

        Returns:
            Configured AdaptiveRouter instance.
        """
        return cls(
            use_llm_fallback=config.get("use_llm_fallback", False),
            llm_fallback_threshold=config.get("llm_fallback_threshold", 1.0),
        )

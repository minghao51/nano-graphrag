"""Tests for AdaptiveRouter."""

import pytest

from bench.techniques.adaptive_router import AdaptiveRouter


class TestAdaptiveRouter:
    """Test AdaptiveRouter functionality."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        router = AdaptiveRouter()

        assert router._use_llm_fallback is False
        assert router._llm_fallback_threshold == 1.0
        assert len(router._multihop_regex) == len(router.MULTIHOP_PATTERNS)
        assert len(router._global_regex) == len(router.GLOBAL_PATTERNS)

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        router = AdaptiveRouter(use_llm_fallback=True, llm_fallback_threshold=2.0)

        assert router._use_llm_fallback is True
        assert router._llm_fallback_threshold == 2.0

    def test_route_multihop_query(self):
        """Test routing of multi-hop queries."""
        router = AdaptiveRouter()

        # Multi-hop queries
        assert router.route("Who is also connected to both X and Y?") == "multihop"
        assert router.route("What is the relationship between X and Y?") == "multihop"
        assert router.route("How are X and Y connected?") == "multihop"
        assert router.route("What do X and Y have in common?") == "multihop"

    def test_route_global_query(self):
        """Test routing of global queries."""
        router = AdaptiveRouter()

        # Global queries
        assert router.route("What are the main themes?") == "global"
        assert router.route("Summarize the overall content") == "global"
        assert router.route("What is the high-level view?") == "global"
        assert router.route("What are the main ideas?") == "global"

    def test_route_local_query(self):
        """Test routing of local queries (default)."""
        router = AdaptiveRouter()

        # Local queries (default)
        assert router.route("What is the capital of France?") == "local"
        assert router.route("Who wrote this book?") == "local"
        assert router.route("When was the treaty signed?") == "local"

    def test_route_case_insensitive(self):
        """Test that routing is case-insensitive."""
        router = AdaptiveRouter()

        assert router.route("WHO IS ALSO CONNECTED") == "multihop"
        assert router.route("What are the THEMES") == "global"
        assert router.route("summarize THE CONTENT") == "global"

    def test_route_with_threshold(self):
        """Test routing with different threshold values."""
        router = AdaptiveRouter(llm_fallback_threshold=2.0)

        # Single match should route to local with threshold of 2.0
        assert router.route("Who is connected to X?") == "local"

        # Two matches should route
        assert router.route("Who is also connected to X and Y?") == "multihop"

    def test_from_config(self):
        """Test creating router from configuration dict."""
        config = {
            "use_llm_fallback": True,
            "llm_fallback_threshold": 2.0,
        }

        router = AdaptiveRouter.from_config(config)

        assert router._use_llm_fallback is True
        assert router._llm_fallback_threshold == 2.0

    def test_from_config_with_defaults(self):
        """Test creating router from configuration dict with defaults."""
        config = {"use_llm_fallback": True}

        router = AdaptiveRouter.from_config(config)

        assert router._use_llm_fallback is True
        assert router._llm_fallback_threshold == 1.0

    def test_llm_fallback_disabled(self):
        """Test that LLM fallback is not used when disabled."""
        router = AdaptiveRouter(use_llm_fallback=False)

        # Ambiguous query should route to local
        assert router.route("Tell me about the document") == "local"

    def test_multihop_patterns_coverage(self):
        """Test that all multi-hop patterns are compiled."""
        router = AdaptiveRouter()

        assert len(router._multihop_regex) == len(router.MULTIHOP_PATTERNS)
        for pattern in router._multihop_regex:
            assert hasattr(pattern, "search")

    def test_global_patterns_coverage(self):
        """Test that all global patterns are compiled."""
        router = AdaptiveRouter()

        assert len(router._global_regex) == len(router.GLOBAL_PATTERNS)
        for pattern in router._global_regex:
            assert hasattr(pattern, "search")

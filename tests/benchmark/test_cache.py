"""Tests for BenchmarkLLMCache."""

import tempfile
import pytest
from nano_graphrag._benchmark.cache import create_benchmark_cache


@pytest.mark.asyncio
async def test_cache_tracks_hits_and_misses():
    """Cache should track hits and misses correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = create_benchmark_cache(tmpdir, enabled=True)

        # First call should miss
        result1 = await cache.get("test prompt", "gpt-4o-mini")
        assert result1 is None
        assert cache.misses == 1
        assert cache.hits == 0

        # Set cache
        await cache.set("test prompt", "gpt-4o-mini", "cached response")

        # Second call should hit
        result2 = await cache.get("test prompt", "gpt-4o-mini")
        assert result2 == "cached response"
        assert cache.hits == 1
        assert cache.misses == 1

        # Stats should include hit rate
        stats = await cache.stats()
        assert stats["hit_rate"] == 0.5  # 1 hit out of 2 calls

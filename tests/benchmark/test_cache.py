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


@pytest.mark.asyncio
async def test_cache_disabled_no_tracking():
    """Cache should not track hits/misses when disabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = create_benchmark_cache(tmpdir, enabled=False)

        # get() should return None and not track
        result1 = await cache.get("test prompt", "gpt-4o-mini")
        assert result1 is None
        assert cache.misses == 0  # Not incremented when disabled
        assert cache.hits == 0

        # Set a value (should still work even if disabled)
        await cache.set("test prompt", "gpt-4o-mini", "cached response")

        # get() should still return None (not using cache)
        result2 = await cache.get("test prompt", "gpt-4o-mini")
        assert result2 is None
        assert cache.hits == 0  # No hit tracking
        assert cache.misses == 0  # No miss tracking

        # get_batch() should return list of None values
        results = await cache.get_batch(["prompt1", "prompt2"], "gpt-4o-mini")
        assert results == [None, None]
        assert cache.hits == 0  # Still no tracking
        assert cache.misses == 0


@pytest.mark.asyncio
async def test_cache_hit_rate_with_no_calls():
    """Cache hit rate should be 0.0 when no calls have been made."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = create_benchmark_cache(tmpdir, enabled=True)

        # No calls made yet
        stats = await cache.stats()
        assert stats["hit_rate"] == 0.0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

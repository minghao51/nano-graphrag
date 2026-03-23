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


@pytest.mark.asyncio
async def test_cache_wrapper_decorates_llm_function():
    """Cache.wrap() should add caching to any LLM function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = create_benchmark_cache(tmpdir, enabled=True)
        call_count = {"value": 0}

        # Mock LLM function
        async def mock_llm(prompt, model="gpt-4o-mini", system_prompt=None):
            call_count["value"] += 1
            return f"Response to: {prompt}"

        # Wrap the function
        wrapped_llm = cache.wrap(mock_llm)

        # First call should invoke the function
        result1 = await wrapped_llm("test prompt", model="gpt-4o-mini")
        assert result1 == "Response to: test prompt"
        assert call_count["value"] == 1

        # Second call should hit cache
        result2 = await wrapped_llm("test prompt", model="gpt-4o-mini")
        assert result2 == "Response to: test prompt"
        assert call_count["value"] == 1  # No additional call

        # Different prompt should miss
        result3 = await wrapped_llm("different prompt", model="gpt-4o-mini")
        assert result3 == "Response to: different prompt"
        assert call_count["value"] == 2

        # Verify cache stats
        assert cache.hits == 1
        assert cache.misses == 2


@pytest.mark.asyncio
async def test_cache_wrapper_with_model_in_kwargs():
    """Cache.wrap() should handle model parameter in kwargs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = create_benchmark_cache(tmpdir, enabled=True)
        call_count = {"value": 0}

        # Mock LLM function with default model
        async def mock_llm(prompt, model="gpt-4o", system_prompt=None):
            call_count["value"] += 1
            return f"Response from {model} to: {prompt}"

        # Wrap the function
        wrapped_llm = cache.wrap(mock_llm)

        # Call with model in kwargs
        result1 = await wrapped_llm("test prompt", model="gpt-4o-mini")
        assert result1 == "Response from gpt-4o-mini to: test prompt"
        assert call_count["value"] == 1

        # Same call should hit cache
        result2 = await wrapped_llm("test prompt", model="gpt-4o-mini")
        assert result2 == "Response from gpt-4o-mini to: test prompt"
        assert call_count["value"] == 1

        # Different model should miss
        result3 = await wrapped_llm("test prompt", model="gpt-4o")
        assert result3 == "Response from gpt-4o to: test prompt"
        assert call_count["value"] == 2


@pytest.mark.asyncio
async def test_cache_wrapper_with_system_prompt():
    """Cache.wrap() should handle system_prompt parameter correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = create_benchmark_cache(tmpdir, enabled=True)
        call_count = {"value": 0}

        # Mock LLM function
        async def mock_llm(prompt, model="gpt-4o-mini", system_prompt=None):
            call_count["value"] += 1
            sys_prefix = f"[{system_prompt}] " if system_prompt else ""
            return f"{sys_prefix}Response to: {prompt}"

        # Wrap the function
        wrapped_llm = cache.wrap(mock_llm)

        # Call with system_prompt
        result1 = await wrapped_llm("test prompt", model="gpt-4o-mini", system_prompt="You are helpful")
        assert result1 == "[You are helpful] Response to: test prompt"
        assert call_count["value"] == 1

        # Same call should hit cache
        result2 = await wrapped_llm("test prompt", model="gpt-4o-mini", system_prompt="You are helpful")
        assert result2 == "[You are helpful] Response to: test prompt"
        assert call_count["value"] == 1

        # Different system_prompt should miss
        result3 = await wrapped_llm("test prompt", model="gpt-4o-mini", system_prompt="You are terse")
        assert result3 == "[You are terse] Response to: test prompt"
        assert call_count["value"] == 2

        # No system_prompt should also miss
        result4 = await wrapped_llm("test prompt", model="gpt-4o-mini")
        assert result4 == "Response to: test prompt"
        assert call_count["value"] == 3


@pytest.mark.asyncio
async def test_cache_wrapper_preserves_function_metadata():
    """Cache.wrap() should preserve original function's metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = create_benchmark_cache(tmpdir, enabled=True)

        # Mock LLM function with metadata
        async def mock_llm(prompt, model="gpt-4o-mini", system_prompt=None):
            """This is the original LLM function."""
            return f"Response to: {prompt}"

        # Wrap the function
        wrapped_llm = cache.wrap(mock_llm)

        # Verify metadata is preserved
        assert wrapped_llm.__name__ == "mock_llm"
        assert wrapped_llm.__doc__ == "This is the original LLM function."

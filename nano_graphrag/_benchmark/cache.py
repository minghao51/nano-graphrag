"""LLM cache formalization for benchmark experiments."""

import functools
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .._storage import JsonKVStorage
from .._utils import compute_args_hash
from ..base import BaseKVStorage


@dataclass
class BenchmarkLLMCache:
    """Persistent LLM cache using BaseKVStorage pattern.

    Wraps the existing hashing_kv pattern used in GraphRAG for benchmark experiments.

    Note: Hit/miss counters are not thread-safe. This is acceptable for the current
    use case (single-threaded benchmark runs).
    """

    storage: BaseKVStorage
    cache_name: str = "benchmark_llm_cache"
    enabled: bool = True
    hits: int = 0
    misses: int = 0

    def _make_cache_key(self, prompt: str, model: str, system_prompt: Optional[str] = None) -> str:
        """Create a cache key from prompt, model, and system prompt."""
        args_hash = compute_args_hash(prompt, model, system_prompt)
        return f"{self.cache_name}:{args_hash}"

    async def get(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
    ) -> Optional[str]:
        """Get cached response if available.

        Args:
            prompt: The LLM prompt
            model: Model name
            system_prompt: Optional system prompt

        Returns:
            Cached response string or None if not found
        """
        # Return None immediately if disabled (don't track)
        if not self.enabled:
            return None

        cache_key = self._make_cache_key(prompt, model, system_prompt)
        result = await self.storage.get_by_id(cache_key)

        if result is not None:
            self.hits += 1
            return result.get("response")

        self.misses += 1
        return None

    async def set(
        self,
        prompt: str,
        model: str,
        response: str,
        system_prompt: Optional[str] = None,
    ) -> None:
        """Cache a response.

        Args:
            prompt: The LLM prompt
            model: Model name
            response: The response to cache
            system_prompt: Optional system prompt
        """
        cache_key = self._make_cache_key(prompt, model, system_prompt)
        await self.storage.upsert(
            {
                cache_key: {
                    "response": response,
                    "prompt": prompt,
                    "model": model,
                    "system_prompt": system_prompt,
                }
            }
        )

    async def get_batch(
        self,
        prompts: List[str],
        model: str,
        system_prompts: Optional[List[Optional[str]]] = None,
    ) -> List[Optional[str]]:
        """Get multiple cached responses.

        Args:
            prompts: List of LLM prompts
            model: Model name
            system_prompts: Optional list of system prompts (one per prompt)

        Returns:
            List of cached responses (None if not found or cache disabled)
        """
        # Return list of None values if disabled (don't track)
        if not self.enabled:
            return [None] * len(prompts)

        if system_prompts is None:
            system_prompts = [None] * len(prompts)

        if len(prompts) != len(system_prompts):
            raise ValueError(
                f"Length mismatch: {len(prompts)} prompts vs {len(system_prompts)} system_prompts"
            )

        cache_keys = [
            self._make_cache_key(prompt, model, sys_prompt)
            for prompt, sys_prompt in zip(prompts, system_prompts)
        ]

        results = await self.storage.get_by_ids(cache_keys)
        return [r.get("response") if r else None for r in results]

    async def set_batch(
        self,
        prompts: List[str],
        model: str,
        responses: List[str],
        system_prompts: Optional[List[Optional[str]]] = None,
    ) -> None:
        """Cache multiple responses.

        Args:
            prompts: List of LLM prompts
            model: Model name
            responses: List of responses to cache
            system_prompts: Optional list of system prompts (one per prompt)
        """
        if len(prompts) != len(responses):
            raise ValueError(
                f"Length mismatch: {len(prompts)} prompts vs {len(responses)} responses"
            )

        if system_prompts is None:
            system_prompts = [None] * len(prompts)

        if len(prompts) != len(system_prompts):
            raise ValueError(
                f"Length mismatch: {len(prompts)} prompts vs {len(system_prompts)} system_prompts"
            )

        cache_data = {}
        for prompt, response, sys_prompt in zip(prompts, responses, system_prompts):
            cache_key = self._make_cache_key(prompt, model, sys_prompt)
            cache_data[cache_key] = {
                "response": response,
                "prompt": prompt,
                "model": model,
                "system_prompt": sys_prompt,
            }

        await self.storage.upsert(cache_data)

    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics including:
                - total_entries: Total number of cached entries
                - models: Set of models in cache
                - hits: Number of cache hits
                - misses: Number of cache misses
                - hit_rate: Cache hit rate (hits / (hits + misses))
        """
        all_keys = await self.storage.all_keys()

        # Filter keys for this cache
        cache_keys = [k for k in all_keys if k.startswith(f"{self.cache_name}:")]

        # Get all entries to analyze
        entries = await self.storage.get_by_ids(cache_keys)
        models = set()

        for entry in entries:
            if entry:
                models.add(entry.get("model", "unknown"))

        # Calculate hit rate
        total_calls = self.hits + self.misses
        hit_rate = self.hits / total_calls if total_calls > 0 else 0.0

        return {
            "total_entries": len(cache_keys),
            "models": sorted(list(models)),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }

    async def clear(self) -> None:
        """Clear all cached entries for this cache."""
        all_keys = await self.storage.all_keys()
        cache_keys = [k for k in all_keys if k.startswith(f"{self.cache_name}:")]

        if cache_keys:
            await self.storage.delete(cache_keys)

    async def flush(self) -> None:
        """Flush cache to disk (if applicable)."""
        await self.storage.index_done_callback()

    def wrap(self, llm_func: Callable[..., Awaitable[str]]) -> Callable[..., Awaitable[str]]:
        """Wrap an LLM function with transparent caching.

        The wrapped function will check the cache before calling the original
        LLM function, and store results in the cache after calls.

        Args:
            llm_func: An async LLM function with signature:
                (prompt, model=None, system_prompt=None, **kwargs) -> str

        Returns:
            A wrapped async function with the same signature that uses caching.
        """

        @functools.wraps(llm_func)
        async def wrapped(
            prompt: str,
            model: Optional[str] = None,
            system_prompt: Optional[str] = None,
            **kwargs,
        ) -> str:
            # Extract model and system_prompt from kwargs to avoid duplicates
            # Use explicit params if provided, otherwise fall back to kwargs
            final_model = model if model is not None else kwargs.pop("model", None)
            final_system_prompt = system_prompt if system_prompt is not None else kwargs.pop("system_prompt", None)

            # Check cache (skip if model is None - let original function handle defaults)
            if final_model is not None:
                cached_response = await self.get(prompt, final_model, final_system_prompt)
                if cached_response is not None:
                    return cached_response

            # Cache miss - call original function with cleaned kwargs
            # Pass model and system_prompt explicitly if we have them, otherwise let None
            call_kwargs = {}
            if final_model is not None:
                call_kwargs["model"] = final_model
            if final_system_prompt is not None:
                call_kwargs["system_prompt"] = final_system_prompt
            call_kwargs.update(kwargs)

            response = await llm_func(prompt, **call_kwargs)

            # Store in cache (only if we have a model)
            if final_model is not None:
                await self.set(prompt, final_model, response, final_system_prompt)

            return response

        return wrapped


def create_benchmark_cache(
    working_dir: str,
    cache_name: str = "benchmark_llm_cache",
    enabled: bool = True,
) -> BenchmarkLLMCache:
    """Create a benchmark LLM cache with JSON storage.

    Args:
        working_dir: Directory for cache storage
        cache_name: Name for this cache instance
        enabled: Whether cache is enabled (default: True)

    Returns:
        BenchmarkLLMCache instance
    """
    # Create working directory if it doesn't exist
    os.makedirs(working_dir, exist_ok=True)

    # Create storage instance
    storage = JsonKVStorage(
        namespace=f"{cache_name}_storage",
        global_config={"working_dir": working_dir},
    )

    return BenchmarkLLMCache(storage=storage, cache_name=cache_name, enabled=enabled)

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import litellm
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ._utils import compute_args_hash, logger, wrap_embedding_func_with_attrs
from .base import BaseKVStorage

if TYPE_CHECKING:
    import numpy as np

PROVIDERS_SUPPORTING_STRUCTURED_OUTPUT = {
    "openai",
    "openrouter",
    "azure",
    "anthropic",
    "google_genai",
    "google_vertex_ai",
    "cohere",
}

# Model name prefixes for provider detection
PROVIDER_MODEL_PREFIXES = {
    "openai": ["gpt-", "o1", "o3"],
    "anthropic": ["claude"],
    "google_genai": ["gemini"],
    "cohere": ["command"],
    "mistral": ["mistral-", "mixtral-"],
    "ollama": ["llama", "mistral", "gemma", "phi", "qwen", "yi"],
}

UNSUPPORTED_STRUCTURED_OUTPUT_ERRORS = tuple(
    exc
    for exc in (
        getattr(litellm, "UnsupportedAPIError", None),
        getattr(litellm, "BadRequestError", None),
    )
    if isinstance(exc, type)
)

# Template for instructing models to respond with structured JSON
SCHEMA_INSTRUCTION_TEMPLATE = """
Respond with valid JSON matching this schema:
{schema_json}
"""


def _extract_usage(response, model: str, elapsed_ms: float, event_name: str, **extra) -> dict:
    """Extract usage data from a LiteLLM response and log it.

    Args:
        response: LiteLLM completion or embedding response object.
        model: Model name used for the call.
        elapsed_ms: Call latency in milliseconds.
        event_name: Structlog event name (e.g., "llm_call_complete").
        **extra: Additional key-value pairs to include in the log.

    Returns:
        Dict with prompt_tokens, completion_tokens, total_tokens, cost_usd.
    """
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    total_tokens = getattr(usage, "total_tokens", 0) if usage else 0
    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception as e:
        logger.debug("llm_cost_calculation_failed", model=model, error=str(e))
        cost = 0.0

    logger.info(
        event_name,
        model=model,
        latency_ms=round(elapsed_ms, 1),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=round(cost, 6),
        **extra,
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": round(cost, 6),
    }


def detect_provider(model: str) -> str:
    """Detect LiteLLM provider from model name.

    Args:
        model: Model name (e.g., "gpt-4o", "ollama/llama3.2", "claude-3-sonnet")

    Returns:
        Provider name (e.g., "openai", "ollama", "anthropic")
    """
    # Explicit provider prefix
    if "/" in model:
        provider, model_only = model.split("/", 1)
        if provider == "openrouter":
            return provider
        if provider in PROVIDERS_SUPPORTING_STRUCTURED_OUTPUT:
            return provider
        # For unknown providers, try to detect from model name
    else:
        model_only = model

    # Detect from model name prefix
    for provider, prefixes in PROVIDER_MODEL_PREFIXES.items():
        if any(model_only.startswith(p) for p in prefixes):
            return provider

    logger.error("provider_detection_failed", model=model)
    raise ValueError(
        f"Unable to detect provider for model {model!r}. "
        "Use an explicit provider prefix (e.g., 'openai/...', 'openrouter/...')."
    )


def supports_structured_output(model: str) -> bool:
    provider = detect_provider(model)
    return provider in PROVIDERS_SUPPORTING_STRUCTURED_OUTPUT


def _is_transient_llm_exception(exc: BaseException) -> bool:
    transient_types = tuple(
        err
        for err in (
            getattr(litellm, "RateLimitError", None),
            getattr(litellm, "APIConnectionError", None),
            getattr(litellm, "ServiceUnavailableError", None),
            getattr(litellm, "InternalServerError", None),
            getattr(litellm, "Timeout", None),
        )
        if isinstance(err, type)
    )
    if transient_types and isinstance(exc, transient_types):
        return True
    if isinstance(exc, asyncio.TimeoutError):
        return True

    message = str(exc).lower()
    transient_patterns = (
        "rate limit",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "service unavailable",
        "connection reset",
        "connection refused",
        "internal server error",
        "bad gateway",
        "gateway timeout",
    )
    return any(pattern in message for pattern in transient_patterns)


def should_fallback_without_structured_output(exc: Exception) -> bool:
    """Return True when the provider rejected structured output parameters."""
    message = str(exc).lower()
    return any(
        pattern in message
        for pattern in (
            "response_format",
            "json_schema",
            "structured output",
            "json_object",
        )
    )


def build_json_schema_response_format(response_format: type[BaseModel]) -> dict[str, Any]:
    """Build a provider-native json_schema response_format payload."""
    schema = response_format.model_json_schema()
    schema_name = schema.get("title") or response_format.__name__
    # Azure/OpenRouter requires additionalProperties: false for strict mode
    if "additionalProperties" not in schema:
        schema["additionalProperties"] = False
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": schema,
        },
    }


def build_provider_requirements(model: str) -> dict[str, Any] | None:
    """Build provider-specific routing requirements for structured output calls."""
    if detect_provider(model) == "openrouter":
        return {"require_parameters": True}
    return None


def _prepare_structured_output(
    model: str,
    response_format: type[BaseModel] | None,
    messages: list,
    has_system_prompt: bool,
    use_native: bool,
    litellm_kwargs: dict,
) -> None:
    """Configure structured output — modifies messages and litellm_kwargs in-place."""
    if response_format is None or not supports_structured_output(model):
        return

    if is_gemma_model(model) or is_qwen_model(model):
        _add_schema_instruction_to_messages(response_format, messages, has_system_prompt)
        if is_qwen_model(model):
            messages[:] = ensure_json_keyword_in_prompt(messages)
            litellm_kwargs["response_format"] = {"type": "json_object"}
        return

    if use_native:
        if isinstance(response_format, dict):
            litellm_kwargs["response_format"] = response_format
        else:
            litellm_kwargs["response_format"] = build_json_schema_response_format(response_format)
            provider_requirements = build_provider_requirements(model)
            if provider_requirements is not None:
                litellm_kwargs["provider"] = provider_requirements
    else:
        _add_schema_instruction_to_messages(response_format, messages, has_system_prompt)


def is_qwen_model(model: str) -> bool:
    """Detect if model is Qwen-based (needs special JSON handling).

    Qwen models require 'json' keyword in message and json_object response format.
    """
    model_lower = model.lower()
    return "qwen" in model_lower or "@preset/cheap-fast" in model_lower


def is_gemma_model(model: str) -> bool:
    """Detect if model is Gemma-based and needs prompt-based schema.

    Gemma 4+ supports native json_schema structured output via OpenRouter.
    Only Gemma 1/2/3 need the legacy prompt-based approach.
    """
    model_lower = model.lower()
    if "gemma" not in model_lower:
        return False
    if re.search(r"gemma[\s_-]?4", model_lower):
        return False
    return True


def ensure_json_keyword_in_prompt(messages: list) -> list:
    """Ensure 'JSON' keyword is in prompt for Qwen models.

    Qwen requires the word 'json' (case-insensitive) in messages to use
    response_format with json_object.
    """
    json_keywords = ["json", "JSON", "Json"]
    has_json = any(any(kw in msg.get("content", "") for kw in json_keywords) for msg in messages)

    if not has_json:
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += "\n\nPlease respond with JSON."
        else:
            messages.insert(0, {"role": "system", "content": "Please respond with JSON."})

    return messages


def _add_schema_instruction_to_messages(
    response_format: type[BaseModel],
    messages: list,
    has_system_prompt: bool,
) -> None:
    """Add JSON schema instruction to messages.

    Modifies messages in-place by prepending or appending schema instruction.

    Args:
        response_format: Pydantic model class defining the expected schema
        messages: Message list to modify in-place
        has_system_prompt: Whether a system prompt already exists at messages[0]
    """
    if hasattr(response_format, "model_json_schema"):
        schema_json = json.dumps(response_format.model_json_schema(), indent=2)
    elif isinstance(response_format, dict):
        schema_json = json.dumps(response_format, indent=2)
    else:
        schema_json = str(response_format)

    schema_instruction = SCHEMA_INSTRUCTION_TEMPLATE.format(schema_json=schema_json)

    if has_system_prompt:
        messages[0]["content"] += "\n\n" + schema_instruction
    else:
        messages.insert(0, {"role": "system", "content": schema_instruction})


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(_is_transient_llm_exception),
    reraise=True,
)
async def litellm_completion(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[Any] | None = None,
    response_format: type[BaseModel] | None = None,
    use_native_structured_output: bool = True,
    hashing_kv: BaseKVStorage | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    timeout: int = 120,
    **kwargs,
) -> str | BaseModel:
    history_messages = history_messages or []
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None:
        if response_format is not None:
            response_format_name = getattr(response_format, "__name__", str(response_format))
        else:
            response_format_name = None
        args_hash = compute_args_hash(
            model,
            messages,
            api_base,
            response_format_name,
            use_native_structured_output,
        )
        cached_result = await hashing_kv.get_by_id(args_hash)
        if cached_result is not None:
            logger.info("llm_cache_hit", model=model, args_hash=args_hash)
            if cached_result.get("is_structured") and response_format is not None:
                return response_format.model_validate_json(cached_result["return"])
            return cached_result["return"]

    litellm_kwargs = {
        "model": model,
        "messages": messages,
        "timeout": timeout,
        **kwargs,
    }

    # Add custom API base and key for self-hosted endpoints
    if api_base:
        litellm_kwargs["api_base"] = api_base
    if api_key:
        litellm_kwargs["api_key"] = api_key

    _prepare_structured_output(
        model,
        response_format,
        messages,
        bool(system_prompt),
        use_native_structured_output,
        litellm_kwargs,
    )

    async def _call_llm():
        try:
            response = await litellm.acompletion(**litellm_kwargs)
            return response
        except UNSUPPORTED_STRUCTURED_OUTPUT_ERRORS as e:
            if "response_format" not in litellm_kwargs:
                raise
            if not should_fallback_without_structured_output(e):
                raise
            logger.warning(
                "structured_output_fallback",
                model=model,
                error=str(e),
            )
            litellm_kwargs.pop("response_format", None)
            response = await litellm.acompletion(**litellm_kwargs)
            return response

    start_time = time.monotonic()
    try:
        response = await asyncio.wait_for(_call_llm(), timeout=timeout)
    except TimeoutError:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.error(
            "llm_call_timeout", model=model, timeout_s=timeout, latency_ms=round(elapsed_ms, 1)
        )
        raise
    except Exception as e:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.error(
            "llm_call_failed",
            model=model,
            error=str(e),
            error_type=type(e).__name__,
            latency_ms=round(elapsed_ms, 1),
        )
        raise

    elapsed_ms = (time.monotonic() - start_time) * 1000
    result = response.choices[0].message.content

    usage_data = _extract_usage(response, model, elapsed_ms, "llm_call_complete")

    if (
        response_format is not None
        and isinstance(response_format, type)
        and issubclass(response_format, BaseModel)
    ):
        if isinstance(result, str):
            try:
                result = response_format.model_validate_json(result)
            except (ValidationError, json.JSONDecodeError, AttributeError) as e:
                logger.warning("structured_output_parse_failed", model=model, error=str(e))
                if use_native_structured_output:
                    logger.info("structured_output_fallback_to_text", model=model)
                    return await litellm_completion(
                        model,
                        prompt,
                        system_prompt,
                        history_messages,
                        response_format=response_format,
                        use_native_structured_output=False,
                        hashing_kv=hashing_kv,
                        api_base=api_base,
                        api_key=api_key,
                        timeout=timeout,
                        **kwargs,
                    )

    if hashing_kv is not None:
        cached_payload = result.model_dump_json() if isinstance(result, BaseModel) else result
        await hashing_kv.upsert(
            {
                args_hash: {
                    "return": cached_payload,
                    "model": model,
                    "is_structured": isinstance(result, BaseModel),
                    **usage_data,
                }
            }
        )
        await hashing_kv.index_done_callback()

    return result


async def litellm_completion_stream(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[Any] | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    timeout: int = 120,
    **kwargs,
) -> AsyncIterator[str]:
    history_messages = history_messages or []
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    litellm_kwargs = {
        "model": model,
        "messages": messages,
        "timeout": timeout,
        "stream": True,
        **kwargs,
    }
    if api_base:
        litellm_kwargs["api_base"] = api_base
    if api_key:
        litellm_kwargs["api_key"] = api_key

    try:
        response = await asyncio.wait_for(litellm.acompletion(**litellm_kwargs), timeout=timeout)
    except Exception as e:
        logger.warning(
            "streaming_fallback_to_buffered",
            model=model,
            error=str(e),
        )
        result = await litellm_completion(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_base=api_base,
            api_key=api_key,
            timeout=timeout,
            **kwargs,
        )
        if isinstance(result, BaseModel):
            yield result.model_dump_json()
        elif result:
            yield result
        return

    async for chunk in response:
        if not getattr(chunk, "choices", None):
            continue
        delta = getattr(chunk.choices[0], "delta", None)
        if delta is None:
            continue
        content = getattr(delta, "content", None)
        if not content:
            continue
        if isinstance(content, list):
            for item in content:
                text = getattr(item, "text", None) or item.get("text")
                if text:
                    yield text
            continue
        yield content


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
async def litellm_embedding(
    texts: list[str],
    model: str = "text-embedding-3-small",
    api_base: str | None = None,
    api_key: str | None = None,
) -> np.ndarray:  # type: ignore[name-defined]
    import numpy as np

    kwargs = {"model": model, "input": texts}
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    start_time = time.monotonic()
    response = await litellm.aembedding(**kwargs)
    elapsed_ms = (time.monotonic() - start_time) * 1000

    _extract_usage(response, model, elapsed_ms, "embedding_call_complete", num_texts=len(texts))
    return np.array([dp["embedding"] for dp in response.data])


class LiteLLMWrapper:
    def __init__(
        self,
        model: str = "openrouter/google/gemma-4-31b-it",
        structured_output: bool = True,
        use_native_structured_output: bool = True,
        hashing_kv: BaseKVStorage | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: int = 120,
    ):
        self.model = model
        self.structured_output = structured_output
        self.use_native_structured_output = use_native_structured_output
        self.hashing_kv = hashing_kv
        self.api_base = api_base
        self.api_key = api_key
        self.timeout = timeout

    async def __call__(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[Any] | None = None,
        response_format: type[BaseModel] | None = None,
        **kwargs,
    ) -> str | BaseModel:
        format_to_use = response_format if self.structured_output else None
        return await litellm_completion(
            model=self.model,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            response_format=format_to_use,
            use_native_structured_output=self.use_native_structured_output,
            hashing_kv=self.hashing_kv,
            api_base=self.api_base,
            api_key=self.api_key,
            timeout=self.timeout,
            **kwargs,
        )

    async def astream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[Any] | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        async for chunk in litellm_completion_stream(
            model=self.model,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_base=self.api_base,
            api_key=self.api_key,
            timeout=self.timeout,
            **kwargs,
        ):
            yield chunk

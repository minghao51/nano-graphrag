import asyncio
from typing import TYPE_CHECKING, Any, List, Optional, Type, Union

import litellm
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

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

    # Unknown provider, log warning and default to openai
    logger.warning(
        f"Cannot detect provider for model '{model}', defaulting to 'openai'. "
        f"This may cause issues if the model is not an OpenAI model."
    )
    return "openai"


def supports_structured_output(model: str) -> bool:
    provider = detect_provider(model)
    return provider in PROVIDERS_SUPPORTING_STRUCTURED_OUTPUT


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


def build_json_schema_response_format(response_format: Type[BaseModel]) -> dict[str, Any]:
    """Build a provider-native json_schema response_format payload."""
    schema = response_format.model_json_schema()
    schema_name = schema.get("title") or response_format.__name__
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": schema,
        },
    }


def build_provider_requirements(model: str) -> Optional[dict[str, Any]]:
    """Build provider-specific routing requirements for structured output calls."""
    if detect_provider(model) == "openrouter":
        return {"require_parameters": True}
    return None


def is_qwen_model(model: str) -> bool:
    """Detect if model is Qwen-based (needs special JSON handling).

    Qwen models require 'json' keyword in message and json_object response format.
    """
    model_lower = model.lower()
    return "qwen" in model_lower or "@preset/cheap-fast" in model_lower


def build_qwen_response_format(response_format: Type[BaseModel]) -> dict[str, Any]:
    """Build Qwen-compatible json_object response format.

    Qwen requires {"type": "json_object"} instead of json_schema.
    """
    return {"type": "json_object"}


def ensure_json_keyword_in_prompt(messages: list, prompt: str) -> list:
    """Ensure 'JSON' keyword is in prompt for Qwen models.

    Qwen requires the word 'json' (case-insensitive) in messages to use
    response_format with json_object.
    """
    json_keywords = ["json", "JSON", "Json"]
    has_json = any(any(kw in msg.get("content", "") for kw in json_keywords) for msg in messages)

    if not has_json:
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = messages[0]["content"] + "\n\nPlease respond with JSON."
        else:
            messages.insert(0, {"role": "system", "content": "Please respond with JSON."})

    return messages


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((asyncio.TimeoutError, Exception)),
    reraise=True,
)
async def litellm_completion(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Any]] = None,
    response_format: Optional[Type[BaseModel]] = None,
    use_native_structured_output: bool = True,
    hashing_kv: Optional[BaseKVStorage] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 120,
    **kwargs,
) -> Union[str, BaseModel]:
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

    # Choose between native structured output or prompt-based schema guidance.
    if response_format is not None and supports_structured_output(model):
        if use_native_structured_output:
            # Use provider-native strict schema mode when available.
            litellm_kwargs["response_format"] = build_json_schema_response_format(response_format)
            provider_requirements = build_provider_requirements(model)
            if provider_requirements is not None:
                litellm_kwargs["provider"] = provider_requirements
        else:
            # Legacy route: Add schema to system prompt
            import json

            if hasattr(response_format, "model_json_schema"):
                schema_json = json.dumps(response_format.model_json_schema(), indent=2)
            elif isinstance(response_format, dict):
                schema_json = json.dumps(response_format, indent=2)
            else:
                schema_json = str(response_format)

            schema_instruction = f"""
Respond with valid JSON matching this schema:
{schema_json}
"""
            if system_prompt:
                messages[0]["content"] += "\n\n" + schema_instruction
            else:
                messages.insert(0, {"role": "system", "content": schema_instruction})
            # For Qwen models, ensure JSON keyword is present and set response_format
            if is_qwen_model(model):
                messages = ensure_json_keyword_in_prompt(messages, prompt)
                litellm_kwargs["response_format"] = {"type": "json_object"}

    async def _call_llm():
        try:
            response = await litellm.acompletion(**litellm_kwargs)
            return response.choices[0].message.content
        except UNSUPPORTED_STRUCTURED_OUTPUT_ERRORS as e:
            if (
                "response_format" not in litellm_kwargs
                or not should_fallback_without_structured_output(e)
            ):
                raise
            logger.warning(f"Structured output not supported, retrying without it: {e}")
            litellm_kwargs.pop("response_format", None)
            response = await litellm.acompletion(**litellm_kwargs)
            return response.choices[0].message.content

    try:
        result = await asyncio.wait_for(_call_llm(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.error(f"LiteLLM call timed out after {timeout}s")
        raise
    except Exception as e:
        logger.error(f"LiteLLM call failed: {e}")
        raise

    # Parse if using structured output
    if response_format is not None and isinstance(result, str):
        try:
            import json

            parsed = json.loads(result)
            if hasattr(response_format, "model_validate_json"):
                result = response_format.model_validate_json(result)
            elif hasattr(response_format, "__call__"):
                result = response_format(**parsed)
            else:
                result = parsed
        except Exception as e:
            logger.warning(f"Failed to parse structured output: {e}")
            if use_native_structured_output:
                # Fallback to text mode
                logger.info("Falling back to text parsing mode")
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
            # Return string as-is if parsing fails
            logger.warning("Could not parse as structured output, returning raw string")

    if hashing_kv is not None:
        cached_payload = result.model_dump_json() if isinstance(result, BaseModel) else result
        await hashing_kv.upsert(
            {
                args_hash: {
                    "return": cached_payload,
                    "model": model,
                    "is_structured": isinstance(result, BaseModel),
                }
            }
        )
        await hashing_kv.index_done_callback()

    return result


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
async def litellm_embedding(
    texts: list[str],
    model: str = "text-embedding-3-small",
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> "np.ndarray":  # type: ignore[name-defined]
    import numpy as np

    kwargs = {"model": model, "input": texts}
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    response = await litellm.aembedding(**kwargs)
    return np.array([dp["embedding"] for dp in response.data])


class LiteLLMWrapper:
    def __init__(
        self,
        model: str = "gpt-4o-mini",  # Aligned with DEFAULT_LLM_MODEL (no provider prefix)
        structured_output: bool = True,
        use_native_structured_output: bool = True,
        hashing_kv: Optional[BaseKVStorage] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
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
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Any]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> Union[str, BaseModel]:
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

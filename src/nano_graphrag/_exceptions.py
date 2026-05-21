from __future__ import annotations

from typing import Any


class GraphRAGError(Exception):
    """Base exception for all nano-graphrag errors."""

    def __init__(
        self, message: str, details: dict[str, Any] | None = None, cause: Exception | None = None
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        parts = [self.message]
        if self.details:
            parts.append(f" details={self.details}")
        return "".join(parts)


class ConfigError(GraphRAGError):
    """Invalid configuration (model names, batch sizes, etc.)."""


class StorageConfigError(ConfigError):
    """Backend-specific storage configuration issues."""


class ExtractionError(GraphRAGError):
    """Entity extraction failures."""


class LLMExtractionError(ExtractionError):
    """LLM call failures during extraction."""


class ParsingError(ExtractionError):
    """Structured output parsing failures."""


class QueryError(GraphRAGError):
    """Query-time failures."""


class ModeNotEnabledError(QueryError):
    """Requested query mode is not enabled."""


class NoContextError(QueryError):
    """No relevant context found for query."""


class StorageError(GraphRAGError):
    """Storage backend failures."""


class GraphIntegrityError(StorageError):
    """Graph/manifest mismatch detected."""


class VectorDBError(StorageError):
    """Vector DB failures."""


class LLMError(GraphRAGError):
    """LiteLLM wrapper errors."""


class RateLimitError(LLMError):
    """Rate limit exceeded (retries exhausted)."""


class AuthError(LLMError):
    """API key / authentication issues."""


_LITELLM_EXCEPTION_MAP: dict[type, type] = {}


def _build_litellm_exception_map() -> dict[type, type]:
    try:
        import litellm

        mapping = {}
        for attr, target in [
            ("RateLimitError", RateLimitError),
            ("AuthenticationError", AuthError),
            ("APIConnectionError", LLMError),
            ("ServiceUnavailableError", LLMError),
            ("InternalServerError", LLMError),
            ("Timeout", LLMError),
        ]:
            cls = getattr(litellm, attr, None)
            if cls is not None:
                mapping[cls] = target
        return mapping
    except ImportError:
        return {}


def translate_litellm_errors():
    """Context manager that maps litellm exceptions to nano-graphrag exceptions.

    Uses the httpx pattern: lookup dict + ``raise MappedExc(...) from exc``.
    Already-wrapped GraphRAGError instances pass through untouched.
    Unmapped exceptions propagate as-is.
    """
    from contextlib import contextmanager

    global _LITELLM_EXCEPTION_MAP
    if not _LITELLM_EXCEPTION_MAP:
        _LITELLM_EXCEPTION_MAP = _build_litellm_exception_map()

    @contextmanager
    def _ctx():
        try:
            yield
        except GraphRAGError:
            raise
        except Exception as exc:
            for from_exc, to_exc in _LITELLM_EXCEPTION_MAP.items():
                if isinstance(exc, from_exc):
                    raise to_exc(str(exc), cause=exc) from exc
            raise

    return _ctx()

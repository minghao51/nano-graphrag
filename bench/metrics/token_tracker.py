"""Token usage tracking for benchmark cost measurement."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict


@dataclass
class TokenUsage:
    """Accumulated token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0

    def add(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.llm_calls += 1

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenUsage":
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            llm_calls=data.get("llm_calls", 0),
        )

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            llm_calls=self.llm_calls + other.llm_calls,
        )


class TokenTracker:
    """Wraps LLM functions to track token usage via character-based estimation."""

    def __init__(self) -> None:
        self._usage = TokenUsage()

    @property
    def usage(self) -> TokenUsage:
        return self._usage

    def wrap(self, llm_func: Callable[..., Awaitable[str]]) -> Callable[..., Awaitable[str]]:
        """Wrap an LLM function to estimate and track token usage."""

        @functools.wraps(llm_func)
        async def wrapped(prompt: str, **kwargs) -> str:
            estimated_prompt_tokens = self._usage.estimate_tokens(prompt)
            response = await llm_func(prompt, **kwargs)
            estimated_completion_tokens = self._usage.estimate_tokens(
                response if isinstance(response, str) else ""
            )
            self._usage.add(
                prompt_tokens=estimated_prompt_tokens,
                completion_tokens=estimated_completion_tokens,
            )
            return response

        return wrapped

    def reset(self) -> None:
        self._usage = TokenUsage()

"""Plugin registry for swappable pipeline components."""

from typing import Any, Optional, Protocol, TypeVar

from nano_graphrag import GraphRAG
from nano_graphrag._ops import chunking_by_seperators, chunking_by_token_size
from nano_graphrag.base import QueryParam

T = TypeVar("T")


class Chunker(Protocol):
    """Splits text into chunks. Matches nano-graphrag's chunk_func signature."""

    def __call__(
        self,
        tokens_list: list[list[int]],
        doc_keys,
        tokenizer_wrapper: Any,
        overlap_token_size: int = 128,
        max_token_size: int = 1024,
        **kwargs,
    ) -> list[dict[str, Any]]: ...


class EntityExtractor(Protocol):
    """Extracts entities and relations from a chunk. Wraps best_model_func."""

    async def __call__(
        self,
        chunk: str,
        source_sub_graph: Any = None,
        language: Optional[str] = None,
        model: Optional[str] = None,
        model_max_token_size: Optional[int] = None,
        **kwargs,
    ) -> dict[str, Any]: ...


class Retriever(Protocol):
    """Retrieves context from the graph given a query."""

    async def __call__(
        self,
        query: str,
        graph_rag: GraphRAG,
        param: QueryParam,
        **kwargs,
    ) -> str: ...


class Reranker(Protocol):
    """Re-scores a list of retrieved passages."""

    def __call__(
        self,
        query: str,
        passages: list[str],
        **kwargs,
    ) -> list[tuple[str, float]]: ...


class Generator(Protocol):
    """Generates a final answer given query + context."""

    async def __call__(
        self,
        query: str,
        context: str,
        **kwargs,
    ) -> str: ...


_REGISTRY: dict[str, dict[str, Any]] = {
    "chunker": {},
    "entity_extractor": {},
    "retriever": {},
    "reranker": {},
    "generator": {},
}


def register(stage: str, name: str):
    """Decorator: @register('chunker', 'separator') registers a chunker.

    Args:
        stage: One of 'chunker', 'entity_extractor', 'retriever', 'reranker', 'generator'
        name: Unique name for this plugin variant

    Returns:
        Decorator function that registers the class

    Example:
        @register("chunker", "separator")
        class SeparatorChunker:
            ...
    """

    def decorator(cls: type[T]) -> type[T]:
        if stage not in _REGISTRY:
            raise ValueError(f"Unknown stage '{stage}'. Valid stages: {list(_REGISTRY.keys())}")
        _REGISTRY[stage][name] = cls
        return cls

    return decorator


def resolve(stage: str, name: str) -> Any:
    """Resolve a registered plugin by stage and name.

    Args:
        stage: Plugin stage (e.g., 'chunker', 'retriever')
        name: Plugin name (e.g., 'token_size', 'local')

    Returns:
        The registered plugin class

    Raises:
        KeyError: If no plugin is registered for the given stage/name
    """
    if stage not in _REGISTRY:
        raise KeyError(f"Unknown stage '{stage}'. Valid stages: {list(_REGISTRY.keys())}")
    if name not in _REGISTRY[stage]:
        available = list(_REGISTRY[stage].keys())
        raise KeyError(f"No {stage} registered as '{name}'. Available: {available}")
    return _REGISTRY[stage][name]


def list_registered(stage: str) -> list[str]:
    """List all registered plugins for a given stage.

    Args:
        stage: Plugin stage to query

    Returns:
        List of registered plugin names
    """
    if stage not in _REGISTRY:
        return []
    return list(_REGISTRY[stage].keys())


def clear_registry(stage: Optional[str] = None) -> None:
    """Clear registry for a specific stage or all stages.

    Args:
        stage: Optional stage to clear. If None, clears all stages.
    """
    if stage is None:
        for s in _REGISTRY:
            _REGISTRY[s].clear()
    elif stage in _REGISTRY:
        _REGISTRY[stage].clear()


@register("chunker", "token_size")
class TokenSizeChunker:
    """Wrapper around chunking_by_token_size."""

    def __init__(
        self,
        overlap_token_size: int = 128,
        max_token_size: int = 1024,
    ):
        self.overlap_token_size = overlap_token_size
        self.max_token_size = max_token_size

    def __call__(
        self,
        tokens_list: list[list[int]],
        doc_keys,
        tokenizer_wrapper: Any,
        **kwargs,
    ) -> list[dict[str, Any]]:
        return chunking_by_token_size(
            tokens_list,
            doc_keys,
            tokenizer_wrapper,
            overlap_token_size=self.overlap_token_size,
            max_token_size=self.max_token_size,
        )


@register("chunker", "separator")
class SeparatorChunker:
    """Wrapper around chunking_by_seperators."""

    def __init__(
        self,
        overlap_token_size: int = 128,
        max_token_size: int = 1024,
    ):
        self.overlap_token_size = overlap_token_size
        self.max_token_size = max_token_size

    def __call__(
        self,
        tokens_list: list[list[int]],
        doc_keys,
        tokenizer_wrapper: Any,
        **kwargs,
    ) -> list[dict[str, Any]]:
        return chunking_by_seperators(
            tokens_list,
            doc_keys,
            tokenizer_wrapper,
            overlap_token_size=self.overlap_token_size,
            max_token_size=self.max_token_size,
        )


@register("retriever", "local")
class LocalRetriever:
    """Retriever using GraphRAG local mode."""

    def __init__(self, top_k: int = 20):
        self.top_k = top_k

    async def __call__(
        self,
        query: str,
        graph_rag: GraphRAG,
        param: QueryParam,
        **kwargs,
    ) -> str:
        local_param = QueryParam(
            mode="local",
            only_need_context=param.only_need_context,
            top_k=kwargs.get("top_k", self.top_k),
            **{
                k: v
                for k, v in param.__dict__.items()
                if k not in ("mode", "only_need_context", "top_k")
            },
        )
        return await graph_rag.aquery(query, param=local_param)


@register("retriever", "global")
class GlobalRetriever:
    """Retriever using GraphRAG global mode."""

    def __init__(self, top_k: int = 20):
        self.top_k = top_k

    async def __call__(
        self,
        query: str,
        graph_rag: GraphRAG,
        param: QueryParam,
        **kwargs,
    ) -> str:
        global_param = QueryParam(
            mode="global",
            only_need_context=param.only_need_context,
            top_k=kwargs.get("top_k", self.top_k),
            **{
                k: v
                for k, v in param.__dict__.items()
                if k not in ("mode", "only_need_context", "top_k")
            },
        )
        return await graph_rag.aquery(query, param=global_param)


@register("retriever", "naive")
class NaiveRetriever:
    """Retriever using GraphRAG naive mode."""

    def __init__(self, top_k: int = 20):
        self.top_k = top_k

    async def __call__(
        self,
        query: str,
        graph_rag: GraphRAG,
        param: QueryParam,
        **kwargs,
    ) -> str:
        naive_param = QueryParam(
            mode="naive",
            only_need_context=param.only_need_context,
            top_k=kwargs.get("top_k", self.top_k),
            **{
                k: v
                for k, v in param.__dict__.items()
                if k not in ("mode", "only_need_context", "top_k")
            },
        )
        return await graph_rag.aquery(query, param=naive_param)


@register("retriever", "multihop")
class MultiHopRetrieverWrapper:
    """Wrapper around MultiHopRetriever for registry compatibility."""

    def __init__(
        self,
        max_hops: int = 4,
        entities_per_hop: int = 10,
        context_token_budget: int = 8000,
        decompose_model: str = "gpt-4o-mini",
    ):
        from .retrievers.multihop import MultiHopRetriever

        self._retriever = MultiHopRetriever(
            max_hops=max_hops,
            entities_per_hop=entities_per_hop,
            context_token_budget=context_token_budget,
            decompose_model=decompose_model,
        )

    async def __call__(
        self,
        query: str,
        graph_rag: GraphRAG,
        param: QueryParam,
        **kwargs,
    ) -> str:
        return await self._retriever.retrieve(query, graph_rag)


@register("reranker", "cross_encoder")
class CrossEncoderRerankerWrapper:
    """Wrapper around CrossEncoderReranker for registry compatibility."""

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 20,
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        from .techniques.reranker import CrossEncoderReranker

        self._reranker = CrossEncoderReranker(
            model=model,
            top_k=top_k,
            device=device,
            batch_size=batch_size,
        )

    def __call__(
        self,
        query: str,
        passages: list[str],
        **kwargs,
    ) -> list[tuple[str, float]]:
        return self._reranker(query, passages)


@register("retriever", "adaptive")
class AdaptiveRouterWrapper:
    """Wrapper around AdaptiveRouter for registry compatibility."""

    def __init__(
        self,
        use_llm_fallback: bool = False,
        llm_fallback_threshold: float = 1.0,
    ):
        from .techniques.adaptive_router import AdaptiveRouter

        self._router = AdaptiveRouter(
            use_llm_fallback=use_llm_fallback,
            llm_fallback_threshold=llm_fallback_threshold,
        )

    async def __call__(
        self,
        query: str,
        graph_rag: GraphRAG,
        param: QueryParam,
        **kwargs,
    ) -> str:
        return await self._router(query, graph_rag, param, **kwargs)


@register("retriever", "hipporag")
class HippoRAGRetrieverWrapper:
    """Wrapper around HippoRAGRetriever for registry compatibility."""

    def __init__(
        self,
        alpha: float = 0.85,
        top_k_seed: int = 5,
        top_k_result: int = 20,
    ):
        from .retrievers.hipporag_ppr import HippoRAGRetriever

        self._retriever = HippoRAGRetriever(
            alpha=alpha,
            top_k_seed=top_k_seed,
            top_k_result=top_k_result,
        )

    async def __call__(
        self,
        query: str,
        graph_rag: GraphRAG,
        param: QueryParam,
        **kwargs,
    ) -> str:
        return await self._retriever(query, graph_rag, param, **kwargs)


@register("retriever", "hybrid")
class HybridRetrieverWrapper:
    """Wrapper around HybridRetriever for registry compatibility."""

    def __init__(
        self,
        retrievers: Optional[list[str]] = None,
        weights: Optional[list[float]] = None,
        fusion: str = "weighted_avg",
        top_k: int = 20,
    ):
        from .retrievers.hybrid import HybridRetriever

        if retrievers is None:
            retrievers = ["local", "global"]

        self._retriever = HybridRetriever(
            retrievers=retrievers,
            weights=weights,
            fusion=fusion,
            top_k=top_k,
        )

    async def __call__(
        self,
        query: str,
        graph_rag: GraphRAG,
        param: QueryParam,
        **kwargs,
    ) -> str:
        return await self._retriever(query, graph_rag, param, **kwargs)


@register("retriever", "raptor")
class RaptorRetrieverWrapper:
    """Wrapper around RaptorRetriever for registry compatibility."""

    def __init__(
        self,
        max_levels: int = 3,
        cluster_model: str = "gmm",
        summary_model: str = "cheap",
        chunk_size: int = 200,
        top_k: int = 10,
    ):
        from .techniques.raptor import RaptorRetriever

        self._retriever = RaptorRetriever(
            max_levels=max_levels,
            cluster_model=cluster_model,
            summary_model=summary_model,
            chunk_size=chunk_size,
            top_k=top_k,
        )

    async def __call__(
        self,
        query: str,
        graph_rag: GraphRAG,
        param: QueryParam,
        **kwargs,
    ) -> str:
        return await self._retriever(query, graph_rag, param, **kwargs)

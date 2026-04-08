import os
import warnings
from dataclasses import asdict
from functools import partial

from ._utils import EmbeddingFunc, TokenizerWrapper, limit_async_func_call
from .base import (
    SUPPORTED_GRAPH_CLUSTERING,
)


def _normalize_settings(self):
    if self.graph_cluster_algorithm not in SUPPORTED_GRAPH_CLUSTERING:
        raise ValueError(
            f"Unsupported graph_cluster_algorithm={self.graph_cluster_algorithm!r}. "
            f"Supported: {', '.join(SUPPORTED_GRAPH_CLUSTERING)}"
        )

    if self.embedding_batch_size is not None:
        self.embedding_batch_num = self.embedding_batch_size
    elif self.embedding_batch_num != 32:
        warnings.warn(
            "`embedding_batch_num` is deprecated; use `embedding_batch_size`.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.embedding_batch_size = self.embedding_batch_num
    else:
        self.embedding_batch_size = self.embedding_batch_num

    if self.llm_max_async is not None:
        self.best_model_max_async = self.llm_max_async
        self.cheap_model_max_async = self.llm_max_async
    if self.embedding_max_async is not None:
        self.embedding_func_max_async = self.embedding_max_async


def _configure_logging(self):
    import logging

    from ._utils import logger

    numeric_level = getattr(logging, self.log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if not any(getattr(h, "_nano_graphrag_console", False) for h in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        console_handler._nano_graphrag_console = True
        logger.addHandler(console_handler)

    if self.log_file:
        if not any(
            getattr(h, "_nano_graphrag_file", None) == self.log_file for h in logger.handlers
        ):
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            file_handler._nano_graphrag_file = self.log_file
            logger.addHandler(file_handler)

    params_str = ",\n  ".join(f"{k} = {v}" for k, v in asdict(self).items())
    logger.debug(f"GraphRAG init with param:\n  {params_str}")


def _build_tokenizer(self):
    model_name = (
        self.tiktoken_model_name
        if self.tokenizer_type == "tiktoken"
        else self.huggingface_model_name
    )
    self.tokenizer_wrapper = TokenizerWrapper(
        tokenizer_type=self.tokenizer_type,
        model_name=model_name,
    )


def _configure_runtime(self):
    from ._llm_litellm import LiteLLMWrapper, litellm_embedding, supports_structured_output
    from ._utils import logger

    if not os.path.exists(self.working_dir) and self.always_create_working_dir:
        logger.info(f"Creating working directory {self.working_dir}")
        os.makedirs(self.working_dir)

    self.llm_response_cache = (
        self.key_string_value_json_storage_cls(
            namespace="llm_response_cache",
            global_config=asdict(self),
        )
        if self.enable_llm_cache
        else None
    )

    use_custom_models = self.best_model_func is not None or self.cheap_model_func is not None
    use_custom_embedding = self.embedding_func is not None

    if use_custom_models or use_custom_embedding:
        if self.best_model_func is not None:
            self.best_model_func = limit_async_func_call(self.best_model_max_async)(
                self.best_model_func
            )
        if self.cheap_model_func is not None:
            self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
                self.cheap_model_func
            )
        if self.embedding_func is not None:
            if not isinstance(self.embedding_func, EmbeddingFunc):
                self.embedding_func = EmbeddingFunc(
                    embedding_dim=self.embedding_dim,
                    max_token_size=8192,
                    func=limit_async_func_call(self.embedding_func_max_async)(
                        self.embedding_func
                    ),
                )
        self._use_structured_extraction = False
        return

    structured = self.structured_output and supports_structured_output(self.llm_model)
    if not structured:
        logger.info(
            f"Model {self.llm_model} does not support structured output, "
            "will use text mode with fallback parsing"
        )

    self.best_model_func = limit_async_func_call(self.best_model_max_async)(
        LiteLLMWrapper(
            model=self.llm_model,
            structured_output=self.structured_output,
            use_native_structured_output=self.use_pydantic_structured_output,
            hashing_kv=self.llm_response_cache,
            api_base=self.llm_api_base,
            api_key=self.llm_api_key,
            timeout=self.llm_timeout,
        )
    )
    self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
        LiteLLMWrapper(
            model=self.llm_cheap_model,
            structured_output=self.structured_output,
            use_native_structured_output=self.use_pydantic_structured_output,
            hashing_kv=self.llm_response_cache,
            api_base=self.llm_api_base,
            api_key=self.llm_api_key,
            timeout=self.llm_timeout,
        )
    )

    limited_embedding = limit_async_func_call(self.embedding_func_max_async)(
        partial(
            litellm_embedding,
            model=self.embedding_model,
            api_base=self.embedding_api_base,
            api_key=self.embedding_api_key,
        )
    )
    self.embedding_func = EmbeddingFunc(
        embedding_dim=self.embedding_dim,
        max_token_size=8192,
        func=limited_embedding,
    )

    logger.info(
        f"Using LiteLLM: model={self.llm_model}, "
        f"api_base={self.llm_api_base}, "
        f"structured_output={self.structured_output}"
    )

    self._use_structured_extraction = True


def _build_storages(self):
    self.full_docs = self.key_string_value_json_storage_cls(
        namespace="full_docs", global_config=asdict(self)
    )
    self.text_chunks = self.key_string_value_json_storage_cls(
        namespace="text_chunks", global_config=asdict(self)
    )
    self.community_reports = self.key_string_value_json_storage_cls(
        namespace="community_reports", global_config=asdict(self)
    )
    self.document_index = self.key_string_value_json_storage_cls(
        namespace="document_index", global_config=asdict(self)
    )
    self.chunk_entity_relation_graph = self.graph_storage_cls(
        namespace="chunk_entity_relation", global_config=asdict(self)
    )

    self.entities_vdb = (
        self.vector_db_storage_cls(
            namespace="entities",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )
        if self.enable_local
        else None
    )
    self.chunks_vdb = (
        self.vector_db_storage_cls(
            namespace="chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        if self.enable_naive_rag
        else None
    )


def _runtime_config(self) -> dict:
    runtime_config = asdict(self)
    runtime_config["_use_structured_extraction"] = self._use_structured_extraction
    return runtime_config

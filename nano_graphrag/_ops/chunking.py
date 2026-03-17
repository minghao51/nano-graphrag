from typing import Dict, Union

from .._splitter import SeparatorSplitter
from .._utils import TokenizerWrapper, compute_mdhash_id
from ..prompt import PROMPTS


def chunking_by_token_size(
    tokens_list: list[list[int]],
    doc_keys,
    tokenizer_wrapper: TokenizerWrapper,
    overlap_token_size=128,
    max_token_size=1024,
):
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = []
        lengths = []
        for start in range(0, len(tokens), max_token_size - overlap_token_size):
            chunk_token.append(tokens[start : start + max_token_size])
            lengths.append(min(max_token_size, len(tokens) - start))

        chunk_texts = tokenizer_wrapper.decode_batch(chunk_token)

        for i, chunk in enumerate(chunk_texts):
            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )
    return results


def chunking_by_seperators(
    tokens_list: list[list[int]],
    doc_keys,
    tokenizer_wrapper: TokenizerWrapper,
    overlap_token_size=128,
    max_token_size=1024,
):
    separators = [tokenizer_wrapper.encode(s) for s in PROMPTS["default_text_separator"]]
    splitter = SeparatorSplitter(
        separators=separators,
        chunk_size=max_token_size,
        chunk_overlap=overlap_token_size,
    )
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_tokens = splitter.split_tokens(tokens)
        lengths = [len(c) for c in chunk_tokens]

        decoded_chunks = tokenizer_wrapper.decode_batch(chunk_tokens)
        for i, chunk in enumerate(decoded_chunks):
            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )
    return results


def get_chunks(
    new_docs,
    chunk_func=chunking_by_token_size,
    tokenizer_wrapper: TokenizerWrapper = None,
    **chunk_func_params,
):
    inserting_chunks: Dict[str, Dict[str, Union[str, int]]] = {}
    new_docs_list = list(new_docs.items())
    docs = [new_doc[1]["content"] for new_doc in new_docs_list]
    doc_keys = [new_doc[0] for new_doc in new_docs_list]

    tokens = [tokenizer_wrapper.encode(doc) for doc in docs]
    chunks = chunk_func(
        tokens,
        doc_keys=doc_keys,
        tokenizer_wrapper=tokenizer_wrapper,
        overlap_token_size=chunk_func_params.get("overlap_token_size", 128),
        max_token_size=chunk_func_params.get("max_token_size", 1024),
    )
    for chunk in chunks:
        inserting_chunks[compute_mdhash_id(chunk["content"], prefix="chunk-")] = chunk
    return inserting_chunks

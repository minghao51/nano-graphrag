import asyncio

import numpy as np
import pytest

from nano_graphrag._utils import (
    AsyncRWLock,
    compute_args_hash,
    enclose_string_with_quotes,
    is_float_regex,
    limit_async_func_call,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    truncate_list_by_token_size,
    wrap_embedding_func_with_attrs,
)

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_limit_async_func_call_limits_concurrency():
    concurrent = 0
    max_concurrent = 0

    @limit_async_func_call(2)
    async def tracked_task():
        nonlocal concurrent, max_concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0.01)
        concurrent -= 1

    await asyncio.gather(*[tracked_task() for _ in range(10)])
    assert max_concurrent <= 2


def test_wrap_embedding_func_attrs():
    @wrap_embedding_func_with_attrs(embedding_dim=128, max_token_size=512)
    async def embed(texts):
        return np.zeros((len(texts), 128))

    assert embed.embedding_dim == 128
    assert embed.max_token_size == 512


def test_truncate_list_by_token_size_basic():
    class MockTokenizer:
        def encode(self, text):
            return list(range(len(text)))

    list_data = [{"t": "a"}, {"t": "bb"}, {"t": "ccc"}]
    mock = MockTokenizer()
    result = truncate_list_by_token_size(
        list_data, key=lambda x: x["t"], max_token_size=5, tokenizer_wrapper=mock
    )
    assert result == [{"t": "a"}, {"t": "bb"}]


def test_truncate_list_by_token_size_zero_returns_empty():
    class MockTokenizer:
        def encode(self, text):
            return list(range(len(text)))

    list_data = [{"t": "a"}]
    mock = MockTokenizer()
    result = truncate_list_by_token_size(
        list_data, key=lambda x: x["t"], max_token_size=0, tokenizer_wrapper=mock
    )
    assert result == []


def test_compute_args_hash_deterministic():
    assert compute_args_hash(1, "a") == compute_args_hash(1, "a")
    assert compute_args_hash(1, "a") != compute_args_hash(1, "b")


def test_is_float_regex():
    assert is_float_regex("3.14") is True
    assert is_float_regex("-1.0") is True
    assert is_float_regex("abc") is False
    assert is_float_regex("1e5") is False


def test_enclose_string_with_quotes():
    assert enclose_string_with_quotes("hello") == '"hello"'
    assert enclose_string_with_quotes(42) == "42"


@pytest.mark.asyncio
async def test_async_rw_lock_read_write():
    lock = AsyncRWLock()
    results = []

    async def reader(label):
        async with lock.read_lock():
            results.append(f"{label}_start")
            await asyncio.sleep(0.01)
            results.append(f"{label}_end")

    async def writer(label):
        async with lock.write_lock():
            results.append(f"{label}_start")
            await asyncio.sleep(0.01)
            results.append(f"{label}_end")

    await asyncio.gather(reader("r1"), reader("r2"))
    assert "r1_start" in results
    assert "r2_start" in results

    results.clear()
    await asyncio.gather(writer("w1"), reader("r3"))
    w1_idx = results.index("w1_start")
    w1_end_idx = results.index("w1_end")
    r3_idx = results.index("r3_start")
    assert w1_end_idx < r3_idx


def test_pack_user_ass_to_openai_messages():
    standard = pack_user_ass_to_openai_messages("p", "g", using_amazon_bedrock=False)
    assert standard == [
        {"role": "user", "content": "p"},
        {"role": "assistant", "content": "g"},
    ]

    bedrock = pack_user_ass_to_openai_messages("p", "g", using_amazon_bedrock=True)
    assert bedrock == [
        {"role": "user", "content": [{"text": "p"}]},
        {"role": "assistant", "content": [{"text": "g"}]},
    ]


def test_list_of_list_to_csv():
    result = list_of_list_to_csv([["a", "b"], ["c", "d"]])
    assert '"a"' in result
    assert '"c"' in result

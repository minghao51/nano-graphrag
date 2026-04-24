import pytest

from nano_graphrag._ops.chunking import (
    chunking_by_seperators,
    chunking_by_token_size,
    get_chunks,
)
from nano_graphrag._splitter import SeparatorSplitter

pytestmark = pytest.mark.unit


class _MockTokenizer:
    """Simple mock tokenizer for testing."""

    def __init__(self):
        self.words = [
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
            "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        ]

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 26 for c in text.lower() if c.isalpha()]

    def decode_batch(self, tokens_list: list[list[int]]) -> list[str]:
        return [" ".join([self.words[t % 26] for t in tokens]) for tokens in tokens_list]


class TestChunkingByTokenSize:
    """Tests for chunking_by_token_size function."""

    @pytest.fixture
    def tokenizer(self):
        return _MockTokenizer()

    def test_empty_input(self, tokenizer):
        result = chunking_by_token_size([], [], tokenizer)
        assert result == []

    def test_single_document_no_splitting(self, tokenizer):
        tokens = [[1, 2, 3, 4, 5]]
        result = chunking_by_token_size(
            tokens,
            ["doc1"],
            tokenizer,
            max_token_size=10,
            overlap_token_size=0,
        )
        assert len(result) == 1
        assert result[0]["full_doc_id"] == "doc1"

    def test_document_exceeds_max_token_size(self, tokenizer):
        tokens = list(range(15))
        result = chunking_by_token_size(
            [tokens], ["doc1"], tokenizer, max_token_size=5, overlap_token_size=1
        )
        assert len(result) == 4

    def test_multiple_documents(self, tokenizer):
        tokens = [[1, 2, 3], [4, 5, 6, 7, 8]]
        result = chunking_by_token_size(tokens, ["doc1", "doc2"], tokenizer)
        doc_ids = [r["full_doc_id"] for r in result]
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids

    def test_chunk_order_index_increments(self, tokenizer):
        tokens = list(range(20))
        result = chunking_by_token_size([tokens], ["doc1"], tokenizer, max_token_size=5, overlap_token_size=0)
        indices = [r["chunk_order_index"] for r in result]
        assert indices == [0, 1, 2, 3]


class TestChunkingBySeparators:
    """Tests for chunking_by_seperators function."""

    @pytest.fixture
    def tokenizer(self):
        return _MockTokenizer()

    def test_empty_input(self, tokenizer):
        result = chunking_by_seperators([], [], tokenizer)
        assert result == []

    def test_single_document(self, tokenizer):
        tokens = [[1, 2, 3, 4, 5]]
        result = chunking_by_seperators(
            tokens, ["doc1"], tokenizer, max_token_size=10, overlap_token_size=0
        )
        assert len(result) >= 1

    def test_result_has_required_fields(self, tokenizer):
        tokens = [[1, 2, 3]]
        result = chunking_by_seperators(
            tokens, ["doc1"], tokenizer, max_token_size=10, overlap_token_size=0
        )
        for chunk in result:
            assert "tokens" in chunk
            assert "content" in chunk
            assert "chunk_order_index" in chunk
            assert "full_doc_id" in chunk


class TestSeparatorSplitter:
    """Tests for SeparatorSplitter class."""

    def test_init_defaults(self):
        splitter = SeparatorSplitter(separators=[[1, 2]], chunk_size=100)
        assert splitter._chunk_size == 100

    def test_split_tokens_empty(self):
        splitter = SeparatorSplitter(separators=[[1]], chunk_size=10, chunk_overlap=0)
        result = splitter.split_tokens([])
        assert result == []

    def test_split_tokens_single_chunk(self):
        splitter = SeparatorSplitter(separators=[[99]], chunk_size=100, chunk_overlap=0)
        tokens = list(range(10))
        result = splitter.split_tokens(tokens)
        assert len(result) == 1
        assert result[0] == tokens

    def test_split_tokens_multiple_chunks(self):
        splitter = SeparatorSplitter(separators=[[99]], chunk_size=5, chunk_overlap=1)
        tokens = list(range(20))
        result = splitter.split_tokens(tokens)
        assert len(result) > 1

    def test_ignores_empty_separators(self):
        splitter = SeparatorSplitter(separators=[[], [99]], chunk_size=100, chunk_overlap=0)
        tokens = [1, 2, 3]
        result = splitter.split_tokens(tokens)
        assert result == [tokens]


class TestGetChunks:
    """Tests for get_chunks function."""

    @pytest.fixture
    def tokenizer_wrapper(self):
        return _MockTokenizer()

    def test_empty_documents(self, tokenizer_wrapper):
        result = get_chunks({}, tokenizer_wrapper=tokenizer_wrapper)
        assert result == {}

    def test_single_document(self, tokenizer_wrapper):
        docs = {"doc1": {"content": "Hello world this is a test"}}
        result = get_chunks(docs, tokenizer_wrapper=tokenizer_wrapper)
        assert len(result) >= 1
        for chunk_key, chunk in result.items():
            assert "content" in chunk
            assert "tokens" in chunk

    def test_chunk_keys_are_unique(self, tokenizer_wrapper):
        docs = {"doc1": {"content": "Test content"}}
        result = get_chunks(docs, tokenizer_wrapper=tokenizer_wrapper)
        keys = list(result.keys())
        assert len(keys) == len(set(keys))

    def test_custom_chunk_params(self, tokenizer_wrapper):
        docs = {"doc1": {"content": "x" * 1000}}
        result = get_chunks(
            docs,
            tokenizer_wrapper=tokenizer_wrapper,
            max_token_size=50,
            overlap_token_size=10,
        )
        assert len(result) >= 1


class TestGetChunksEdgeCases:
    """Edge case tests for get_chunks."""

    def test_very_long_document(self):
        tokenizer = _MockTokenizer()
        docs = {"long_doc": {"content": "a" * 10000}}
        result = get_chunks(docs, tokenizer_wrapper=tokenizer)
        assert len(result) > 1

    def test_document_with_special_characters(self):
        tokenizer = _MockTokenizer()
        docs = {"doc1": {"content": "Hello! @#$%^&*() world. Test\t\n123"}}
        result = get_chunks(docs, tokenizer_wrapper=tokenizer)
        assert len(result) >= 1

    def test_multiple_documents_in_order(self):
        tokenizer = _MockTokenizer()
        docs = {
            "doc1": {"content": "First document content"},
            "doc2": {"content": "Second document content"},
            "doc3": {"content": "Third document content"},
        }
        result = get_chunks(docs, tokenizer_wrapper=tokenizer)
        # Verify all docs are represented
        doc_ids = [chunk["full_doc_id"] for chunk in result.values()]
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids
        assert "doc3" in doc_ids

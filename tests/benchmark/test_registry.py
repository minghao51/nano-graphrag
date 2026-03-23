"""Tests for plugin registry."""

import pytest


def test_multihop_retriever_registered():
    """Verify multihop retriever is registered."""
    from bench.registry import resolve, list_registered

    # Check it's in the list
    assert "multihop" in list_registered("retriever")

    # Check we can resolve it
    retriever_class = resolve("retriever", "multihop")
    assert retriever_class is not None

"""Tests for benchmark dataset loaders."""

import json
import tempfile
from pathlib import Path

import pytest

from bench.datasets import (
    MultiHopRAGDataset,
    Passage,
    QAPair,
)


def test_dataset_returns_typed_qa_pairs():
    """Dataset questions() should return typed QAPair objects."""
    # Create test JSON with id, question, answer, supporting_facts
    test_questions = [
        {
            "id": "q1",
            "question": "What is the capital of France?",
            "answer": "Paris",
            "supporting_facts": ["France is a country in Europe.", "Paris is the capital."],
            "metadata": {"difficulty": "easy"},
        },
        {
            "id": "q2",
            "question": "What is 2+2?",
            "answer": "4",
            "supporting_facts": ["Basic arithmetic."],
        },
    ]

    test_corpus = [
        {"content": "France is a country in Europe."},
        {"content": "Paris is the capital of France."},
        {"content": "Basic arithmetic operations include addition."},
    ]

    # Create temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        questions_path = Path(tmpdir) / "questions.json"
        corpus_path = Path(tmpdir) / "corpus.json"

        with open(questions_path, "w") as f:
            json.dump(test_questions, f)

        with open(corpus_path, "w") as f:
            json.dump(test_corpus, f)

        # Load with MultiHopRAGDataset
        dataset = MultiHopRAGDataset(
            questions_path=str(questions_path),
            corpus_path=str(corpus_path),
        )

        # Verify questions() returns QAPair objects
        questions = list(dataset.questions())
        assert len(questions) == 2

        qa1 = questions[0]
        assert isinstance(qa1, QAPair)
        assert qa1.id == "q1"
        assert qa1.question == "What is the capital of France?"
        assert qa1.answer == "Paris"
        assert qa1.supporting_facts == [
            "France is a country in Europe.",
            "Paris is the capital.",
        ]
        assert qa1.metadata == {"difficulty": "easy"}

        qa2 = questions[1]
        assert isinstance(qa2, QAPair)
        assert qa2.id == "q2"
        assert qa2.question == "What is 2+2?"
        assert qa2.answer == "4"
        assert qa2.supporting_facts == ["Basic arithmetic."]
        assert qa2.metadata == {}  # Default empty dict

        # Verify corpus() returns Passage objects
        corpus = list(dataset.corpus())
        assert len(corpus) == 3

        doc1 = corpus[0]
        assert isinstance(doc1, Passage)
        assert doc1.text == "France is a country in Europe."
        assert doc1.title == ""  # Default empty string

        doc2 = corpus[1]
        assert isinstance(doc2, Passage)
        assert doc2.text == "Paris is the capital of France."

        doc3 = corpus[2]
        assert isinstance(doc3, Passage)
        assert doc3.text == "Basic arithmetic operations include addition."


def test_dataset_handles_missing_fields_gracefully():
    """Dataset should handle missing optional fields with defaults."""
    # Create test JSON with minimal fields
    test_questions = [
        {
            "question": "What is the capital of France?",
            "answer": "Paris",
            # Missing id, supporting_facts, metadata
        },
    ]

    test_corpus = [
        {"content": "France is a country in Europe."},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        questions_path = Path(tmpdir) / "questions.json"
        corpus_path = Path(tmpdir) / "corpus.json"

        with open(questions_path, "w") as f:
            json.dump(test_questions, f)

        with open(corpus_path, "w") as f:
            json.dump(test_corpus, f)

        dataset = MultiHopRAGDataset(
            questions_path=str(questions_path),
            corpus_path=str(corpus_path),
        )

        questions = list(dataset.questions())
        assert len(questions) == 1

        qa = questions[0]
        assert isinstance(qa, QAPair)
        assert qa.id != ""  # Should generate an ID
        assert qa.question == "What is the capital of France?"
        assert qa.answer == "Paris"
        assert qa.supporting_facts == []  # Default empty list
        assert qa.metadata == {}  # Default empty dict

        corpus = list(dataset.corpus())
        assert len(corpus) == 1
        doc = corpus[0]
        assert isinstance(doc, Passage)
        assert doc.text == "France is a country in Europe."
        assert doc.id != ""  # Should generate an ID

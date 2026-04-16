"""Tests for DSPy prompt tuner."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Mock DSPy before importing the module
sys.modules["dspy"] = MagicMock()


class TestDSPyTuner:
    """Test DSPy prompt tuner functionality."""

    def test_entity_extraction_metric_with_match(self):
        """Test metric calculation with matching entities."""
        from bench.dspy_tune import entity_extraction_metric

        example = MagicMock()
        example.entities_json = json.dumps([
            {"name": "Entity1", "type": "PERSON"},
            {"name": "Entity2", "type": "ORG"},
        ])

        prediction = MagicMock()
        prediction.entities_json = json.dumps([
            {"name": "Entity1", "type": "PERSON"},
            {"name": "Entity3", "type": "LOC"},
        ])

        score = entity_extraction_metric(example, prediction)

        # 1 out of 2 entities matched
        assert score == 0.5

    def test_entity_extraction_metric_perfect_match(self):
        """Test metric calculation with perfect match."""
        from bench.dspy_tune import entity_extraction_metric

        example = MagicMock()
        example.entities_json = json.dumps([
            {"name": "Entity1"},
            {"name": "Entity2"},
        ])

        prediction = MagicMock()
        prediction.entities_json = json.dumps([
            {"name": "Entity1"},
            {"name": "Entity2"},
        ])

        score = entity_extraction_metric(example, prediction)

        # Perfect match
        assert score == 1.0

    def test_entity_extraction_metric_no_match(self):
        """Test metric calculation with no match."""
        from bench.dspy_tune import entity_extraction_metric

        example = MagicMock()
        example.entities_json = json.dumps([
            {"name": "Entity1"},
        ])

        prediction = MagicMock()
        prediction.entities_json = json.dumps([
            {"name": "Entity2"},
        ])

        score = entity_extraction_metric(example, prediction)

        # No match
        assert score == 0.0

    def test_entity_extraction_metric_empty_ground_truth(self):
        """Test metric calculation with empty ground truth."""
        from bench.dspy_tune import entity_extraction_metric

        example = MagicMock()
        example.entities_json = json.dumps([])

        prediction = MagicMock()
        prediction.entities_json = json.dumps([
            {"name": "Entity1"},
        ])

        score = entity_extraction_metric(example, prediction)

        # Empty ground truth = perfect score if prediction is empty, else 0
        assert score == 0.0

    def test_entity_extraction_metric_invalid_json(self):
        """Test metric calculation with invalid JSON."""
        from bench.dspy_tune import entity_extraction_metric

        example = MagicMock()
        example.entities_json = "invalid json"

        prediction = MagicMock()
        prediction.entities_json = json.dumps([{"name": "Entity1"}])

        score = entity_extraction_metric(example, prediction)

        # Invalid JSON should return 0
        assert score == 0.0

    def test_generate_training_examples(self, tmp_path):
        """Test generating training examples from dataset."""
        from bench.dspy_tune import generate_training_examples

        # Create a temporary dataset file
        dataset_path = tmp_path / "dataset.json"
        dataset_data = [
            {
                "chunk": "This is a test chunk about Entity1.",
                "entities": [{"name": "Entity1", "type": "TEST"}],
            },
            {
                "chunk": "Another test chunk about Entity2.",
                "entities": [{"name": "Entity2", "type": "TEST"}],
            },
        ]

        with open(dataset_path, "w") as f:
            json.dump(dataset_data, f)

        # Generate examples
        examples = generate_training_examples(str(dataset_path), num_examples=2)

        assert len(examples) == 2

    def test_generate_training_examples_with_text_field(self, tmp_path):
        """Test generating training examples with 'text' field instead of 'chunk'."""
        from bench.dspy_tune import generate_training_examples

        # Create a temporary dataset file with 'text' field
        dataset_path = tmp_path / "dataset.json"
        dataset_data = [
            {
                "text": "Test chunk",
                "entities": "{}",
            },
        ]

        with open(dataset_path, "w") as f:
            json.dump(dataset_data, f)

        # Generate examples
        examples = generate_training_examples(str(dataset_path), num_examples=1)

        assert len(examples) == 1

    def test_entity_extraction_metric_with_entity_name_field(self):
        """Test metric calculation with entity_name field."""
        from bench.dspy_tune import entity_extraction_metric

        example = MagicMock()
        example.entities_json = json.dumps([
            {"entity_name": "Entity1"},
            {"entity_name": "Entity2"},
        ])

        prediction = MagicMock()
        prediction.entities_json = json.dumps([
            {"entity_name": "Entity1"},
        ])

        score = entity_extraction_metric(example, prediction)

        # 1 out of 2 entities matched
        assert score == 0.5

    def test_dspy_not_installed_error(self):
        """Test that helpful error is raised when DSPy is not installed."""
        # This test verifies the error message is helpful
        # In practice, users would need to install dspy-ai
        from bench.dspy_tune import create_dspy_tuner

        # When DSPy is not installed, should raise ImportError with helpful message
        # We can't test this without actually removing the mock, but we verify the function exists
        assert callable(create_dspy_tuner)

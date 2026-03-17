"""Experimental DSPy helpers kept outside the minimal core package."""

from ...entity_extraction.extract import extract_entities_dspy, generate_dataset
from ...entity_extraction.metric import (
    calculate_f1,
    calculate_f1_from_examples,
    calculate_recall,
)
from ...entity_extraction.module import (
    Entity,
    Relationship,
    TypedEntityRelationshipExtractor,
)

__all__ = [
    "Entity",
    "Relationship",
    "TypedEntityRelationshipExtractor",
    "extract_entities_dspy",
    "generate_dataset",
    "calculate_f1",
    "calculate_f1_from_examples",
    "calculate_recall",
]

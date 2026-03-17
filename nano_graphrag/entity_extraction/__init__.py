import warnings

from .extract import extract_entities_dspy, generate_dataset  # noqa: E402
from .module import Entity, Relationship, TypedEntityRelationshipExtractor  # noqa: E402

warnings.warn(
    "`nano_graphrag.entity_extraction` is deprecated; use "
    "`nano_graphrag.contrib.dspy` for DSPy integrations.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "Entity",
    "Relationship",
    "TypedEntityRelationshipExtractor",
    "extract_entities_dspy",
    "generate_dataset",
]

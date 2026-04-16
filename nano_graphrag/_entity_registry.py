"""Entity Registry for canonical name and alias management.

This module provides a centralized entity registry that maintains:
- Canonical names for all entities
- Alias mappings for name variations
- Entity ID to canonical name mapping
- Fuzzy matching for entity resolution
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any


@dataclass
class EntityRecord:
    """Record for a single entity with all its variations."""

    entity_id: str
    canonical_name: str
    aliases: set[str] = field(default_factory=set)
    entity_type: str = "unknown"  # person, organization, location, etc.
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_alias(self, alias: str) -> None:
        """Add an alias for this entity."""
        normalized = self._normalize_name(alias)
        if normalized and normalized != self._normalize_name(self.canonical_name):
            self.aliases.add(alias)

    def matches(self, name: str, fuzzy_threshold: float = 0.85) -> bool:
        """Check if a name matches this entity (exact or fuzzy)."""
        normalized = self._normalize_name(name)

        # Exact match with canonical name
        if normalized == self._normalize_name(self.canonical_name):
            return True

        # Exact match with alias
        for alias in self.aliases:
            if normalized == self._normalize_name(alias):
                return True

        # Fuzzy match with canonical name
        if (
            self._fuzzy_match(normalized, self._normalize_name(self.canonical_name))
            >= fuzzy_threshold
        ):
            return True

        # Fuzzy match with aliases
        for alias in self.aliases:
            if self._fuzzy_match(normalized, self._normalize_name(alias)) >= fuzzy_threshold:
                return True

        return False

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize a name for comparison."""
        # Lowercase, strip whitespace, remove special chars
        normalized = name.lower().strip()
        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", normalized)
        # Remove common title prefixes/suffixes for better matching
        for prefix in ["mr ", "mrs ", "ms ", "dr ", "prof ", "president ", "ceo "]:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
        return normalized

    @staticmethod
    def _fuzzy_match(s1: str, s2: str) -> float:
        """Calculate fuzzy match ratio between two strings."""
        return SequenceMatcher(None, s1, s2).ratio()


class EntityRegistry:
    """Central registry for entity canonicalization and alias resolution.

    This registry is built during graph construction and used throughout
    the querying pipeline to ensure consistent entity references.
    """

    def __init__(self) -> None:
        # Entity ID -> EntityRecord
        self._entities: dict[str, EntityRecord] = {}

        # Normalized name -> entity_id (for fast lookup)
        self._name_index: dict[str, str] = {}

        # Entity type -> set of entity IDs
        self._type_index: dict[str, set[str]] = defaultdict(set)

    def register_entity(
        self,
        entity_id: str,
        canonical_name: str,
        aliases: list[str] | None = None,
        entity_type: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a new entity with its canonical name and aliases."""
        existing = self._entities.get(entity_id)
        if existing is not None:
            if existing.entity_type != entity_type:
                self._type_index.get(existing.entity_type, set()).discard(entity_id)
            existing.canonical_name = canonical_name
            existing.entity_type = entity_type
            existing.metadata.update(metadata or {})
            for alias in aliases or []:
                existing.add_alias(alias)
            self._entities[entity_id] = existing
            self._type_index[entity_type].add(entity_id)
            self._rebuild_name_index_for_entity(existing)
            return

        record = EntityRecord(
            entity_id=entity_id,
            canonical_name=canonical_name,
            aliases=set(),
            entity_type=entity_type,
            metadata=metadata or {},
        )
        for alias in aliases or []:
            record.add_alias(alias)

        self._entities[entity_id] = record
        self._type_index[entity_type].add(entity_id)

        # Index canonical name
        canonical_normalized = record._normalize_name(canonical_name)
        self._name_index[canonical_normalized] = entity_id

        # Index aliases
        for alias in record.aliases:
            alias_normalized = record._normalize_name(alias)
            self._name_index[alias_normalized] = entity_id

    def _rebuild_name_index_for_entity(self, record: EntityRecord) -> None:
        normalized_names = {
            record._normalize_name(record.canonical_name),
            *(record._normalize_name(alias) for alias in record.aliases),
        }
        for normalized_name in list(self._name_index.keys()):
            if self._name_index[normalized_name] == record.entity_id:
                self._name_index.pop(normalized_name, None)
        for normalized_name in normalized_names:
            self._name_index[normalized_name] = record.entity_id

    def remove_entity(self, entity_id: str) -> None:
        """Remove an entity and its indexes from the registry."""
        record = self._entities.pop(entity_id, None)
        if record is None:
            return

        entity_ids = self._type_index.get(record.entity_type)
        if entity_ids is not None:
            entity_ids.discard(entity_id)
            if not entity_ids:
                self._type_index.pop(record.entity_type, None)

        normalized_names = {
            record._normalize_name(record.canonical_name),
            *(record._normalize_name(alias) for alias in record.aliases),
        }
        for normalized_name in normalized_names:
            if self._name_index.get(normalized_name) == entity_id:
                self._name_index.pop(normalized_name, None)

    def resolve_entity(self, name: str, fuzzy_threshold: float = 0.85) -> str | None:
        """Resolve a name to an entity ID.

        Args:
            name: The entity name to resolve
            fuzzy_threshold: Minimum similarity ratio for fuzzy matching (0-1)

        Returns:
            Entity ID if found, None otherwise
        """
        normalized = EntityRecord._normalize_name(name)

        # Try exact match first
        entity_id = self._name_index.get(normalized)
        if entity_id:
            return entity_id

        # Try fuzzy match
        for entity_id, record in self._entities.items():
            if record.matches(name, fuzzy_threshold):
                return entity_id

        return None

    def find_candidates(
        self,
        name: str,
        entity_type: str | None = None,
        fuzzy_threshold: float = 0.85,
        limit: int = 3,
    ) -> list[tuple[str, float]]:
        normalized = EntityRecord._normalize_name(name)
        scored_candidates: list[tuple[str, float]] = []
        for entity_id, record in self._entities.items():
            if entity_type and record.entity_type not in {entity_type, "unknown", '"UNKNOWN"'}:
                continue
            best_score = EntityRecord._fuzzy_match(
                normalized, record._normalize_name(record.canonical_name)
            )
            for alias in record.aliases:
                best_score = max(
                    best_score,
                    EntityRecord._fuzzy_match(normalized, record._normalize_name(alias)),
                )
            if best_score >= fuzzy_threshold:
                scored_candidates.append((entity_id, best_score))
        scored_candidates.sort(key=lambda item: item[1], reverse=True)
        return scored_candidates[:limit]

    def add_aliases(
        self, entity_id: str, aliases: list[str], metadata: dict[str, Any] | None = None
    ) -> None:
        record = self._entities.get(entity_id)
        if record is None:
            return
        for alias in aliases:
            record.add_alias(alias)
        if metadata:
            record.metadata.update(metadata)
        self._rebuild_name_index_for_entity(record)

    def get_canonical_name(self, entity_id: str) -> str | None:
        """Get the canonical name for an entity ID."""
        record = self._entities.get(entity_id)
        return record.canonical_name if record else None

    def get_entity_record(self, entity_id: str) -> EntityRecord | None:
        """Get the full record for an entity."""
        return self._entities.get(entity_id)

    def get_entities_by_type(self, entity_type: str) -> list[str]:
        """Get all entity IDs of a given type."""
        return list(self._type_index.get(entity_type, set()))

    def resolve_entities_from_text(
        self, text: str, fuzzy_threshold: float = 0.85
    ) -> list[tuple[str, str]]:
        """Extract and resolve entity mentions from text.

        Args:
            text: Text to extract entities from
            fuzzy_threshold: Minimum similarity for fuzzy matching

        Returns:
            List of (entity_id, canonical_name) tuples
        """
        # This is a simple implementation - could be enhanced with NER
        found = []

        for entity_id, record in self._entities.items():
            # Check if canonical name is in text
            if record.canonical_name.lower() in text.lower():
                found.append((entity_id, record.canonical_name))
                continue

            # Check aliases
            for alias in record.aliases:
                if alias.lower() in text.lower():
                    found.append((entity_id, record.canonical_name))
                    break

        return found

    def export_state(self) -> dict[str, Any]:
        """Export registry state for persistence."""
        return {
            "entities": {
                entity_id: {
                    "canonical_name": record.canonical_name,
                    "aliases": list(record.aliases),
                    "entity_type": record.entity_type,
                    "metadata": record.metadata,
                }
                for entity_id, record in self._entities.items()
            },
            "name_index": self._name_index,
            "type_index": {k: list(v) for k, v in self._type_index.items()},
        }

    def import_state(self, state: dict[str, Any]) -> None:
        """Import registry state from persistence."""
        self._entities = {}
        self._name_index = {}
        self._type_index = defaultdict(set)

        for entity_id, data in state["entities"].items():
            self.register_entity(
                entity_id=entity_id,
                canonical_name=data["canonical_name"],
                aliases=data.get("aliases", []),
                entity_type=data.get("entity_type", "unknown"),
                metadata=data.get("metadata", {}),
            )

    def __len__(self) -> int:
        """Return the number of registered entities."""
        return len(self._entities)

    def __contains__(self, entity_id: str) -> bool:
        """Check if an entity ID is registered."""
        return entity_id in self._entities

    def save_to_file(self, filepath: str) -> None:
        """Save registry state to a JSON file.

        Args:
            filepath: Path to the JSON file to write
        """
        state = self.export_state()
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filepath: str) -> "EntityRegistry":
        """Load registry state from a JSON file.

        Args:
            filepath: Path to the JSON file to read

        Returns:
            EntityRegistry instance with loaded state
        """
        registry = cls()

        if not os.path.exists(filepath):
            return registry

        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)

        registry.import_state(state)
        return registry

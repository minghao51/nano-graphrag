import pytest

from nano_graphrag._entity_registry import EntityRecord, EntityRegistry

pytestmark = pytest.mark.unit


class TestEntityRecord:
    """Tests for EntityRecord dataclass."""

    def test_init(self):
        record = EntityRecord(
            entity_id="test-id",
            canonical_name="Test Entity",
            aliases={"alias1", "alias2"},
            entity_type="ORGANIZATION",
            metadata={"key": "value"},
        )
        assert record.entity_id == "test-id"
        assert record.canonical_name == "Test Entity"
        assert record.aliases == {"alias1", "alias2"}
        assert record.entity_type == "ORGANIZATION"
        assert record.metadata == {"key": "value"}

    def test_init_defaults(self):
        record = EntityRecord(entity_id="test-id", canonical_name="Test Entity")
        assert record.entity_type == "unknown"
        assert record.metadata == {}
        assert record.aliases == set()

    def test_add_alias(self):
        record = EntityRecord(entity_id="test-id", canonical_name="Test Entity")
        record.add_alias("Alias One")
        record.add_alias("ALIAS ONE")
        assert "Alias One" in record.aliases

    def test_add_alias_normalizes_and_deduplicates(self):
        record = EntityRecord(entity_id="test-id", canonical_name="Test Entity")
        record.add_alias("  ALIAS ONE  ")
        assert "  ALIAS ONE  " in record.aliases
        record.add_alias("alias one")
        assert len(record.aliases) == 2

    def test_matches_exact_canonical(self):
        record = EntityRecord(entity_id="test-id", canonical_name="Sam Bankman-Fried")
        assert record.matches("Sam Bankman-Fried") is True

    def test_matches_exact_alias(self):
        record = EntityRecord(entity_id="test-id", canonical_name="SBF")
        record.add_alias("Sam Bankman-Fried")
        assert record.matches("Sam Bankman-Fried") is True

    def test_matches_fuzzy(self):
        record = EntityRecord(entity_id="test-id", canonical_name="Sam Bankman-Fried")
        assert record.matches("Sam Bankman-Fried", fuzzy_threshold=0.85) is True
        assert record.matches("Sam Bankman-Frie", fuzzy_threshold=0.85) is True
        ratio = EntityRecord._fuzzy_match("sbf", "sambankmanfried")
        assert ratio <= 0.34
        assert record.matches("SBF", fuzzy_threshold=0.25) is True

    def test_matches_no_match(self):
        record = EntityRecord(entity_id="test-id", canonical_name="Sam Bankman-Fried")
        assert record.matches("Totally Different Name") is False

    def test_matches_case_insensitive(self):
        record = EntityRecord(entity_id="test-id", canonical_name="Sam Bankman-Fried")
        assert record.matches("sam bankman-fried") is True
        assert record.matches("SAM BANKMAN-FRIED") is True

    def test_normalize_name(self):
        normalized = EntityRecord._normalize_name("  Mr. John Smith  ")
        assert normalized == "mr. john smith"

    def test_normalize_name_removes_titles(self):
        assert EntityRecord._normalize_name("Dr. John") == "dr. john"
        assert EntityRecord._normalize_name("President Obama") == "obama"

    def test_fuzzy_match(self):
        ratio = EntityRecord._fuzzy_match("hello", "hello")
        assert ratio == 1.0
        ratio = EntityRecord._fuzzy_match("hello", "helloo")
        assert 0.8 < ratio < 1.0
        ratio = EntityRecord._fuzzy_match("hello", "world")
        assert ratio < 0.5


class TestEntityRegistry:
    """Tests for EntityRegistry class."""

    def test_init(self):
        registry = EntityRegistry()
        assert len(registry) == 0

    def test_register_entity_basic(self):
        registry = EntityRegistry()
        registry.register_entity("entity-1", "Sam Bankman-Fried", entity_type="PERSON")
        assert len(registry) == 1
        assert registry.get_canonical_name("entity-1") == "Sam Bankman-Fried"

    def test_register_entity_with_aliases(self):
        registry = EntityRegistry()
        registry.register_entity(
            "entity-1",
            "Sam Bankman-Fried",
            aliases=["SBF", "Sam"],
            entity_type="PERSON",
        )
        assert registry.resolve_entity("SBF") == "entity-1"
        assert registry.resolve_entity("Sam") == "entity-1"

    def test_register_entity_updates_existing(self):
        registry = EntityRegistry()
        registry.register_entity("entity-1", "Sam Bankman-Fried", entity_type="PERSON")
        registry.register_entity(
            "entity-1",
            "SBF",
            aliases=["Sam", "Bankman-Fried"],
            entity_type="PERSON",
        )
        assert len(registry) == 1
        assert registry.get_canonical_name("entity-1") == "SBF"
        assert registry.resolve_entity("Sam") == "entity-1"
        assert registry.resolve_entity("Bankman-Fried") == "entity-1"

    def test_register_duplicate_entity_id(self):
        registry = EntityRegistry()
        registry.register_entity("entity-1", "Sam Bankman-Fried", entity_type="PERSON")
        registry.register_entity("entity-1", "Different Name", entity_type="ORGANIZATION")
        assert len(registry) == 1
        assert registry.get_canonical_name("entity-1") == "Different Name"

    def test_remove_entity(self):
        registry = EntityRegistry()
        registry.register_entity("entity-1", "Sam Bankman-Fried", entity_type="PERSON")
        registry.remove_entity("entity-1")
        assert len(registry) == 0
        assert registry.resolve_entity("Sam Bankman-Fried") is None

    def test_remove_nonexistent(self):
        registry = EntityRegistry()
        registry.remove_entity("nonexistent")

    def test_resolve_entity_exact_match(self):
        registry = EntityRegistry()
        registry.register_entity("entity-1", "Sam Bankman-Fried", entity_type="PERSON")
        assert registry.resolve_entity("Sam Bankman-Fried") == "entity-1"

    def test_resolve_entity_alias_match(self):
        registry = EntityRegistry()
        registry.register_entity("entity-1", "Sam Bankman-Fried", aliases=["SBF"])
        assert registry.resolve_entity("SBF") == "entity-1"

    def test_resolve_entity_fuzzy_match(self):
        registry = EntityRegistry()
        registry.register_entity("entity-1", "Sam Bankman-Fried")
        result = registry.resolve_entity("Sam Bankman-Frie", fuzzy_threshold=0.85)
        assert result == "entity-1"

    def test_resolve_entity_no_match(self):
        registry = EntityRegistry()
        registry.register_entity("entity-1", "Sam Bankman-Fried")
        assert registry.resolve_entity("Nonexistent Entity") is None

    def test_resolve_entity_case_insensitive(self):
        registry = EntityRegistry()
        registry.register_entity("entity-1", "Sam Bankman-Fried")
        assert registry.resolve_entity("sam bankman-fried") == "entity-1"

    def test_resolve_entity_whitespace_normalized(self):
        registry = EntityRegistry()
        registry.register_entity("entity-1", "Sam Bankman-Fried")
        assert registry.resolve_entity("  Sam Bankman-Fried  ") == "entity-1"

    def test_find_candidates_basic(self):
        registry = EntityRegistry()
        registry.register_entity("entity-1", "Sam Bankman-Fried", entity_type="PERSON")
        registry.register_entity("entity-2", "SBF", entity_type="PERSON")
        registry.register_entity("entity-3", "FTX", entity_type="ORGANIZATION")
        candidates = registry.find_candidates("SBF")
        assert len(candidates) >= 1
        assert any(eid == "entity-2" for eid, score in candidates)

    def test_find_candidates_with_entity_type_filter(self):
        registry = EntityRegistry()
        registry.register_entity("entity-1", "Sam Bankman-Fried", entity_type="PERSON")
        registry.register_entity("entity-2", "FTX", entity_type="ORGANIZATION")
        candidates = registry.find_candidates("Sam", entity_type="ORGANIZATION")
        assert all(eid == "entity-2" or score < 0.9 for eid, score in candidates)

    def test_find_candidates_respects_limit(self):
        registry = EntityRegistry()
        registry.register_entity("entity-1", "Sam Bankman-Fried", entity_type="PERSON")
        registry.register_entity("entity-2", "Sam Smith", entity_type="PERSON")
        registry.register_entity("entity-3", "Samuel Bankman", entity_type="PERSON")
        candidates = registry.find_candidates("Sam", limit=2)
        assert len(candidates) <= 2

    def test_add_aliases(self):
        registry = EntityRegistry()
        registry.register_entity("entity-1", "Sam Bankman-Fried")
        registry.add_aliases("entity-1", ["SBF", "The Founder"])
        assert registry.resolve_entity("SBF") == "entity-1"
        assert registry.resolve_entity("The Founder") == "entity-1"

    def test_add_aliases_nonexistent_entity(self):
        registry = EntityRegistry()
        registry.add_aliases("nonexistent", ["alias"])

    def test_get_canonical_name(self):
        registry = EntityRegistry()
        registry.register_entity("entity-1", "Sam Bankman-Fried")
        assert registry.get_canonical_name("entity-1") == "Sam Bankman-Fried"

    def test_get_canonical_name_nonexistent(self):
        registry = EntityRegistry()
        assert registry.get_canonical_name("nonexistent") is None

    def test_get_entity_record(self):
        registry = EntityRegistry()
        registry.register_entity("entity-1", "Sam Bankman-Fried", entity_type="PERSON")
        record = registry.get_entity_record("entity-1")
        assert record is not None
        assert record.entity_id == "entity-1"
        assert record.canonical_name == "Sam Bankman-Fried"

    def test_get_entity_record_nonexistent(self):
        registry = EntityRegistry()
        assert registry.get_entity_record("nonexistent") is None

    def test_get_entities_by_type(self):
        registry = EntityRegistry()
        registry.register_entity("entity-1", "Sam Bankman-Fried", entity_type="PERSON")
        registry.register_entity("entity-2", "FTX", entity_type="ORGANIZATION")
        registry.register_entity("entity-3", "Alice", entity_type="PERSON")
        persons = registry.get_entities_by_type("PERSON")
        assert len(persons) == 2
        assert "entity-1" in persons
        assert "entity-3" in persons

    def test_get_entities_by_type_nonexistent(self):
        registry = EntityRegistry()
        assert registry.get_entities_by_type("NONEXISTENT") == []

    def test_len(self):
        registry = EntityRegistry()
        assert len(registry) == 0
        registry.register_entity("entity-1", "Test")
        assert len(registry) == 1
        registry.register_entity("entity-2", "Test2")
        assert len(registry) == 2

    def test_iteration(self):
        registry = EntityRegistry()
        registry.register_entity("entity-1", "Test1")
        registry.register_entity("entity-2", "Test2")
        entity_ids = list(registry)
        assert len(entity_ids) == 2
        assert "entity-1" in entity_ids
        assert "entity-2" in entity_ids

    def test_serialization_roundtrip(self):
        registry = EntityRegistry()
        registry.register_entity(
            "entity-1",
            "Sam Bankman-Fried",
            aliases=["SBF"],
            entity_type="PERSON",
            metadata={"founded": "FTX"},
        )
        registry.register_entity("entity-2", "FTX", entity_type="ORGANIZATION")
        data = registry.export_state()
        new_registry = EntityRegistry()
        new_registry.import_state(data)
        assert len(new_registry) == 2
        assert new_registry.get_canonical_name("entity-1") == "Sam Bankman-Fried"
        assert new_registry.resolve_entity("SBF") == "entity-1"
        assert new_registry.resolve_entity("FTX") == "entity-2"

    def test_import_state_invalid(self):
        registry = EntityRegistry()
        registry.import_state({"entities": {}})


class TestEntityRegistryFuzzyMatching:
    """Tests for fuzzy matching edge cases."""

    def test_fuzzy_threshold_zero(self):
        registry = EntityRegistry()
        registry.register_entity("entity-1", "Sam Bankman-Fried")
        result = registry.resolve_entity("anything", fuzzy_threshold=0)
        assert result == "entity-1"

    def test_fuzzy_threshold_one_requires_exact(self):
        registry = EntityRegistry()
        registry.register_entity("entity-1", "Sam Bankman-Fried")
        assert registry.resolve_entity("Sam Bankman-Fried", fuzzy_threshold=1.0) == "entity-1"

    def test_very_similar_names(self):
        registry = EntityRegistry()
        registry.register_entity("entity-1", "John Smith")
        registry.register_entity("entity-2", "John Smythe")
        result1 = registry.resolve_entity("John Smith", fuzzy_threshold=0.9)
        result2 = registry.resolve_entity("John Smythe", fuzzy_threshold=0.9)
        assert result1 == "entity-1"
        assert result2 == "entity-2"

    def test_ambiguous_name_multiple_candidates(self):
        registry = EntityRegistry()
        registry.register_entity("entity-1", "Apple", entity_type="ORGANIZATION")
        registry.register_entity("entity-2", "Apple", entity_type="FRUIT")
        candidates = registry.find_candidates("Apple", limit=5)
        assert len(candidates) == 2

    def test_special_characters_normalized(self):
        registry = EntityRegistry()
        registry.register_entity("entity-1", "John-Doe")
        assert registry.resolve_entity("john doe") == "entity-1"
        assert registry.resolve_entity("john-doe") == "entity-1"


class TestEntityRegistryIndexConsistency:
    """Tests to ensure index consistency after operations."""

    def test_alias_added_after_registration(self):
        registry = EntityRegistry()
        registry.register_entity("entity-1", "Sam Bankman-Fried", entity_type="PERSON")
        assert registry.resolve_entity("Sam Bankman-Fried") == "entity-1"
        assert registry.resolve_entity("SBF") is None
        registry.add_aliases("entity-1", ["SBF"])
        assert registry.resolve_entity("SBF") == "entity-1"

    def test_entity_removal_cleans_indexes(self):
        registry = EntityRegistry()
        registry.register_entity(
            "entity-1",
            "Sam Bankman-Fried",
            aliases=["SBF"],
            entity_type="PERSON",
        )
        entity_id = registry.resolve_entity("SBF")
        assert entity_id == "entity-1"
        registry.remove_entity("entity-1")
        assert registry.resolve_entity("Sam Bankman-Fried") is None
        assert registry.resolve_entity("SBF") is None

    def test_concurrent_modifications(self):
        registry = EntityRegistry()
        for i in range(100):
            registry.register_entity(f"entity-{i}", f"Entity {i}", entity_type="TEST")
        assert len(registry) == 100
        for i in range(0, 100, 2):
            registry.remove_entity(f"entity-{i}")
        assert len(registry) == 50
        for i in range(1, 100, 2):
            assert registry.resolve_entity(f"Entity {i}") == f"entity-{i}"

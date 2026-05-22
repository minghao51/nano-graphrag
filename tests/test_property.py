from __future__ import annotations

import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nano_graphrag._schemas import (
    RELATION_ALIASES,
    RELATION_VOCABULARY,
    BatchedEntityExtractionOutput,
    ChunkExtractionResult,
    ExtractedEntity,
    ExtractedRelationship,
    normalize_relation_type,
)
from nano_graphrag._splitter import SeparatorSplitter
from nano_graphrag._utils import (
    clean_str,
    compute_mdhash_id,
    compute_sha256_id,
    deserialize_source_ids,
    generate_stable_entity_id,
    serialize_source_ids,
    split_string_by_multi_markers,
)
from tests.strategies import (
    st_content_string,
    st_entity_name,
    st_extracted_entity,
    st_extracted_relationship,
    st_namespace,
    st_token_list,
)

pytestmark = [pytest.mark.unit, pytest.mark.property]


@given(content=st_content_string)
@settings(max_examples=20)
def test_mdhash_id_deterministic(content):
    assert compute_mdhash_id(content) == compute_mdhash_id(content)


@given(content=st_content_string)
@settings(max_examples=20)
def test_mdhash_id_idempotent(content):
    first = compute_mdhash_id(content)
    compute_mdhash_id(compute_mdhash_id(content))
    assert first == compute_mdhash_id(content)


@given(a=st_content_string, b=st_content_string)
@settings(max_examples=20)
def test_mdhash_id_different_inputs(a, b):
    if a != b:
        assert compute_mdhash_id(a) != compute_mdhash_id(b)


@given(content=st_content_string)
@settings(max_examples=20)
def test_sha256_id_deterministic(content):
    assert compute_sha256_id(content) == compute_sha256_id(content)


@given(content=st_content_string)
@settings(max_examples=20)
def test_sha256_id_idempotent(content):
    first = compute_sha256_id(content)
    compute_sha256_id(compute_sha256_id(content))
    assert first == compute_sha256_id(content)


@given(name=st_entity_name, ns=st_namespace)
@settings(max_examples=20)
def test_stable_entity_id_namespace_aware(name, ns):
    assert generate_stable_entity_id(name, namespace=ns) == generate_stable_entity_id(
        name, namespace=ns
    )


@given(name=st_entity_name, ns_a=st_namespace, ns_b=st_namespace)
@settings(max_examples=20)
def test_stable_entity_id_different_namespace(name, ns_a, ns_b):
    if ns_a != ns_b:
        assert generate_stable_entity_id(name, namespace=ns_a) != generate_stable_entity_id(
            name, namespace=ns_b
        )


@given(raw=st_content_string)
@settings(max_examples=20)
def test_normalize_relation_idempotent(raw):
    once = normalize_relation_type(raw)
    twice = normalize_relation_type(once)
    assert once == twice


@given(dummy=st.none())
@settings(max_examples=1)
def test_normalize_vocabulary_members(dummy):
    for rel in RELATION_VOCABULARY:
        assert normalize_relation_type(rel) == rel


@given(dummy=st.none())
@settings(max_examples=1)
def test_normalize_aliases_resolve(dummy):
    for alias, target in RELATION_ALIASES.items():
        assert normalize_relation_type(alias) == target


@given(dummy=st.none())
@settings(max_examples=1)
def test_normalize_empty_returns_related_to(dummy):
    assert normalize_relation_type("") == "related_to"
    assert normalize_relation_type(None) == "related_to"
    assert normalize_relation_type("   ") == "related_to"


@given(entity=st_extracted_entity)
@settings(max_examples=20)
def test_extracted_entity_roundtrip(entity):
    data = entity.model_dump()
    restored = ExtractedEntity.model_validate(data)
    assert restored.entity_name == entity.entity_name
    assert restored.entity_type == entity.entity_type
    assert restored.description == entity.description


@given(rel=st_extracted_relationship)
@settings(max_examples=20)
def test_extracted_relationship_roundtrip(rel):
    data = rel.model_dump()
    restored = ExtractedRelationship.model_validate(data)
    assert restored.source == rel.source
    assert restored.target == rel.target
    assert restored.relation_type == rel.relation_type


@given(
    entities=st.lists(st_extracted_entity, min_size=0, max_size=3),
    rels=st.lists(st_extracted_relationship, min_size=0, max_size=3),
)
@settings(max_examples=20)
def test_batched_output_roundtrip(entities, rels):
    chunk = ChunkExtractionResult(chunk_id="test_chunk", entities=entities, relationships=rels)
    batched = BatchedEntityExtractionOutput(chunks=[chunk])
    data = batched.model_dump()
    restored = BatchedEntityExtractionOutput.model_validate(data)
    assert len(restored.chunks) == 1
    assert restored.chunks[0].chunk_id == "test_chunk"
    assert len(restored.chunks[0].entities) == len(entities)
    assert len(restored.chunks[0].relationships) == len(rels)


@given(rel=st_extracted_relationship)
@settings(max_examples=20)
def test_confidence_bounds(rel):
    assert 0.0 <= rel.confidence <= 1.0


@given(tokens=st_token_list)
@settings(max_examples=20)
def test_split_tokens_concatenation(tokens):
    splitter = SeparatorSplitter(separators=[[10]], chunk_size=10000, chunk_overlap=0)
    chunks = splitter.split_tokens(tokens)
    all_tokens = [t for chunk in chunks for t in chunk]
    non_sep = [t for t in tokens if t != 10]
    non_sep_result = [t for t in all_tokens if t != 10]
    assert non_sep == non_sep_result


@given(tokens=st_token_list)
@settings(max_examples=20)
def test_split_tokens_no_empty_chunks(tokens):
    splitter = SeparatorSplitter(separators=[[10]], chunk_size=10000, chunk_overlap=0)
    chunks = splitter.split_tokens(tokens)
    for chunk in chunks:
        assert len(chunk) > 0


@given(tokens=st_token_list)
@settings(max_examples=20)
def test_split_tokens_no_separators_single_chunk(tokens):
    splitter = SeparatorSplitter(separators=None, chunk_size=10000, chunk_overlap=0)
    chunks = splitter.split_tokens(tokens)
    assert len(chunks) == 1
    assert chunks[0] == tokens


@given(ids=st.lists(st_content_string, min_size=1, max_size=20))
@settings(max_examples=20)
def test_serialize_deserialize_source_ids_roundtrip(ids):
    serialized = serialize_source_ids(ids)
    deserialized = deserialize_source_ids(serialized)
    assert sorted(set(ids)) == deserialized


@given(text=st_content_string)
@settings(max_examples=20)
def test_clean_str_removes_control_chars(text):
    result = clean_str(text)
    for ch in result:
        assert not (0x00 <= ord(ch) <= 0x1F) and not (0x7F <= ord(ch) <= 0x9F)


@given(text=st_content_string)
@settings(max_examples=20)
def test_split_string_preserves_content(text):
    markers = [", ", "; "]
    parts = split_string_by_multi_markers(text, markers)
    joined = "".join(parts)
    for ch in joined:
        assert ch in text


@given(text=st.text(min_size=0, max_size=100))
@settings(max_examples=50)
def test_safe_filename_no_invalid_chars(text):
    from nano_graphrag._vault.export import _safe_filename

    result = _safe_filename(text)
    for ch in result:
        assert ch not in ("/", ":", "\\", "\0", "*", "?", '"', "<", ">", "|")


@given(tokens=st.lists(st.integers(min_value=0, max_value=50000), min_size=1, max_size=200))
@settings(max_examples=20)
def test_split_tokens_chunk_size_bound(tokens):
    chunk_size = 50
    splitter = SeparatorSplitter(separators=None, chunk_size=chunk_size, chunk_overlap=0)
    chunks = splitter.split_tokens(tokens)
    for chunk in chunks:
        assert len(chunk) <= chunk_size


@given(
    name=st_entity_name,
    ns=st_namespace,
    whitespace=st.text(alphabet=" \t", min_size=0, max_size=3),
)
@settings(max_examples=20)
def test_stable_entity_id_whitespace_stable(name, ns, whitespace):
    padded = whitespace + name + whitespace
    assert generate_stable_entity_id(padded, namespace=ns) == generate_stable_entity_id(
        padded, namespace=ns
    )


@given(ids=st.lists(st_content_string, min_size=1, max_size=20))
@settings(max_examples=20)
def test_serialize_source_ids_deduplicates(ids):
    serialized = serialize_source_ids(ids)
    deserialized = deserialize_source_ids(serialized)
    assert len(deserialized) == len(set(ids))


@given(
    a=st.sets(
        st.text(alphabet=string.ascii_letters, min_size=1, max_size=3), min_size=2, max_size=10
    ),
    b=st.sets(
        st.text(alphabet=string.ascii_letters, min_size=1, max_size=3), min_size=1, max_size=10
    ),
)
@settings(max_examples=30)
def test_iou_monotonic_with_overlap(a, b):
    from nano_graphrag._ops.extraction import _compute_neighborhood_iou

    a_only = a - b
    if not a_only or a == b:
        return
    extra = next(iter(a_only))
    b_bigger = b | {extra}
    iou_bigger, _ = _compute_neighborhood_iou(a, b_bigger)
    iou_base, _ = _compute_neighborhood_iou(a, b)
    assert iou_bigger >= iou_base

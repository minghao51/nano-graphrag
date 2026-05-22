from __future__ import annotations

import string
from datetime import date

import numpy as np
from hypothesis import strategies as st

from nano_graphrag._schemas import (
    RELATION_VOCABULARY,
    ExtractedEntity,
    ExtractedRelationship,
)

st_entity_name = st.text(
    alphabet=string.ascii_letters + string.digits + " _-",
    min_size=1,
    max_size=50,
).map(str.strip)

st_entity_type = st.sampled_from(
    ["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "CONCEPT", "TECHNOLOGY", "PRODUCT"]
)

st_description = st.text(min_size=1, max_size=500)

st_alias = st.text(
    alphabet=string.ascii_letters + string.digits + " _-",
    min_size=1,
    max_size=30,
)

st_relation_type = st.sampled_from(sorted(RELATION_VOCABULARY))

st_confidence = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

st_weight = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)

st_date_str = st.dates(min_value=date(1900, 1, 1), max_value=date(2030, 12, 31)).map(
    lambda d: d.isoformat()
)

st_extracted_entity = st.builds(
    ExtractedEntity,
    name=st_entity_name,
    type=st_entity_type,
    description=st_description,
    aliases=st.lists(st_alias, max_size=5),
)

st_extracted_relationship = st.builds(
    ExtractedRelationship,
    source=st_entity_name,
    target=st_entity_name,
    description=st_description,
    relation_type=st_relation_type,
    weight=st_weight,
    confidence=st_confidence,
)

st_content_string = st.text(min_size=1, max_size=200)

st_namespace = st.text(
    alphabet=string.ascii_lowercase + string.digits + "_",
    min_size=1,
    max_size=20,
)

st_embedding_vector = st.lists(
    st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=8,
    max_size=8,
).map(np.array)

st_embedding_matrix = st.integers(min_value=1, max_value=10).flatmap(
    lambda n: st.lists(
        st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=8,
            max_size=8,
        ),
        min_size=n,
        max_size=n,
    ).map(lambda rows: np.array(rows))
)

st_token_list = st.lists(st.integers(min_value=0, max_value=50000), min_size=1, max_size=100)

st_chunk_id = st.text(
    alphabet=string.ascii_lowercase + string.digits + "-",
    min_size=8,
    max_size=32,
)

st_node_data = st.fixed_dictionaries(
    {
        "entity_name": st_entity_name,
        "entity_type": st_entity_type,
        "description": st_description,
        "source_id": st.just('["chunk_0"]'),
    }
)

st_edge_data = st.fixed_dictionaries(
    {
        "description": st_description,
        "weight": st_weight,
        "relation_type": st_relation_type,
        "confidence": st_confidence,
    }
)

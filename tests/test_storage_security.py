import pytest

pytest.importorskip("neo4j")

from nano_graphrag._storage.gdb_neo4j import sanitize_neo4j_label

pytestmark = pytest.mark.unit


def test_sanitize_neo4j_label_strips_unsafe_chars():
    assert (
        sanitize_neo4j_label('ORG"` MATCH (n) DETACH DELETE n //') == "ORG_MATCH_n_DETACH_DELETE_n"
    )


def test_sanitize_neo4j_label_handles_empty_and_numeric():
    assert sanitize_neo4j_label("") == "UNKNOWN"
    assert sanitize_neo4j_label("123-org") == "L_123_org"

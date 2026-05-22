import random

import pytest

from nano_graphrag._ops.extraction import _compute_neighborhood_iou
from nano_graphrag._ops.refinement.infer import _select_candidates_weighted
from nano_graphrag._ops.refinement.pipeline import _get_rejection_ttl

pytestmark = [pytest.mark.unit, pytest.mark.metamorphic]


class TestEntityLinkingMonotonicity:
    def test_more_shared_neighbors_higher_iou(self):
        a = {"x", "y", "z"}
        b_identical = {"x", "y", "z"}
        b_empty: set[str] = set()
        b_partial = {"x", "w"}

        iou_identical, _ = _compute_neighborhood_iou(a, b_identical)
        iou_empty, _ = _compute_neighborhood_iou(a, b_empty)
        iou_partial, _ = _compute_neighborhood_iou(a, b_partial)

        assert iou_identical == 1.0
        assert iou_empty == 0.0
        assert 0.0 < iou_partial < 1.0

    def test_iou_symmetric(self):
        a = {"alpha", "beta", "gamma"}
        b = {"beta", "delta"}
        iou_ab, common_ab = _compute_neighborhood_iou(a, b)
        iou_ba, common_ba = _compute_neighborhood_iou(b, a)
        assert iou_ab == iou_ba
        assert common_ab == common_ba

    def test_iou_self_is_one(self):
        s = {"a", "b"}
        iou, common = _compute_neighborhood_iou(s, s)
        assert iou == 1.0
        assert common == s


class TestSelectCandidatesWeighted:
    def test_select_candidates_top_heavy(self):
        random.seed(42)
        candidate_pairs = [(f"a_{i}", f"b_{i}", i + 1) for i in range(100)]
        candidate_pairs.sort(key=lambda x: x[2], reverse=True)
        result = _select_candidates_weighted(candidate_pairs, batch_size=10)
        assert len(result) == 10
        top_8 = result[:8]
        rest = result[8:]
        for pair in top_8:
            assert pair in candidate_pairs[:80]
        assert len(rest) == 2


class TestRejectionTTL:
    def test_higher_threshold_fewer_candidates(self):
        ttl_small = _get_rejection_ttl(10)
        ttl_medium = _get_rejection_ttl(200)
        ttl_large = _get_rejection_ttl(1000)
        assert ttl_large >= ttl_medium
        assert ttl_medium >= ttl_small

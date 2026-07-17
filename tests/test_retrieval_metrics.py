"""Retrieval metric tests with hand-computed expected values."""

from __future__ import annotations

import math

import pytest

from rag_eval.metrics.retrieval import (
    dcg_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieval_metrics_for_query,
)

RETRIEVED = ["a", "b", "c", "d"]
RELEVANT = {"b", "d"}


def test_precision_at_k_basic():
    # top-2 = [a, b]; one relevant out of k=2.
    assert precision_at_k(RETRIEVED, RELEVANT, 2) == pytest.approx(0.5)


def test_precision_at_k_denominator_is_k():
    # Only one item retrieved but k=3, so denominator stays 3.
    assert precision_at_k(["a"], {"a"}, 3) == pytest.approx(1.0 / 3.0)


def test_precision_at_k_rejects_nonpositive_k():
    with pytest.raises(ValueError):
        precision_at_k(RETRIEVED, RELEVANT, 0)


def test_recall_at_k_basic():
    # top-2 intersects relevant in {b}; two relevant total.
    assert recall_at_k(RETRIEVED, RELEVANT, 2) == pytest.approx(0.5)


def test_recall_at_k_full_depth():
    assert recall_at_k(RETRIEVED, RELEVANT, 4) == pytest.approx(1.0)


def test_recall_at_k_no_relevant_returns_zero():
    assert recall_at_k(RETRIEVED, set(), 3) == 0.0


def test_hit_rate_at_k():
    assert hit_rate_at_k(RETRIEVED, RELEVANT, 1) == 0.0  # top-1 is 'a'
    assert hit_rate_at_k(RETRIEVED, RELEVANT, 2) == 1.0  # 'b' at rank 2
    assert hit_rate_at_k(RETRIEVED, {"z"}, 4) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RETRIEVED, RELEVANT) == pytest.approx(0.5)  # 'b' at rank 2
    assert reciprocal_rank(["b"], {"b"}) == pytest.approx(1.0)
    assert reciprocal_rank(["x", "y"], {"z"}) == 0.0


def test_dcg_at_k_hand_value():
    # 3/log2(2) + 2/log2(3) + 1/log2(4) = 3 + 1.261859 + 0.5
    expected = 3.0 + 2.0 / math.log2(3) + 1.0 / math.log2(4)
    assert dcg_at_k([3, 2, 1], 3) == pytest.approx(expected)


def test_ndcg_at_k_binary_hand_value():
    # DCG = 1/log2(3) + 1/log2(5); IDCG = 1/log2(2) + 1/log2(3)
    dcg = 1.0 / math.log2(3) + 1.0 / math.log2(5)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert ndcg_at_k(RETRIEVED, RELEVANT, 4) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(RETRIEVED, RELEVANT, 4) == pytest.approx(0.6509, abs=1e-4)


def test_ndcg_at_k_graded_gains():
    gains = {"A": 3, "B": 2, "C": 0, "D": 1}
    retrieved = ["A", "B", "C", "D"]
    dcg = 3.0 + 2.0 / math.log2(3) + 0.0 + 1.0 / math.log2(5)
    idcg = 3.0 + 2.0 / math.log2(3) + 1.0 / math.log2(4)
    assert ndcg_at_k(retrieved, set(gains), 4, gains=gains) == pytest.approx(dcg / idcg)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["b", "d", "a", "c"], RELEVANT, 4) == pytest.approx(1.0)


def test_retrieval_metrics_for_query_keys_and_values():
    scores = retrieval_metrics_for_query(RETRIEVED, RELEVANT, ks=(1, 3))
    assert set(scores) == {
        "precision@1", "recall@1", "ndcg@1", "hit_rate@1",
        "precision@3", "recall@3", "ndcg@3", "hit_rate@3",
        "mrr",
    }
    assert scores["mrr"] == pytest.approx(0.5)
    assert scores["hit_rate@1"] == 0.0
    assert scores["recall@3"] == pytest.approx(0.5)  # only 'b' in top-3

"""Generation metric scoring tests with hand-computed expected values."""

from __future__ import annotations

import numpy as np
import pytest

from rag_eval.metrics.generation import (
    answer_relevance_score,
    context_utilization_score,
    faithfulness_score,
    max_support,
)


def test_max_support_rows():
    matrix = np.array([[0.1, 0.9], [0.5, 0.2], [0.0, 0.0]])
    assert max_support(matrix).tolist() == pytest.approx([0.9, 0.5, 0.0])


def test_max_support_empty_context_axis():
    # No evidence columns -> zero support for every row.
    assert max_support(np.zeros((3, 0))).tolist() == [0.0, 0.0, 0.0]


def test_max_support_empty_query_axis():
    assert max_support(np.zeros((0, 4))).tolist() == []


def test_max_support_requires_2d():
    with pytest.raises(ValueError):
        max_support(np.array([1.0, 2.0]))


def test_faithfulness_fraction_supported():
    # Two of three answer sentences clear the 0.5 threshold.
    assert faithfulness_score([0.9, 0.4, 0.7], threshold=0.5) == pytest.approx(2 / 3)


def test_faithfulness_threshold_is_inclusive():
    assert faithfulness_score([0.5, 0.5], threshold=0.5) == pytest.approx(1.0)


def test_faithfulness_empty_answer_is_zero():
    assert faithfulness_score([], threshold=0.5) == 0.0


def test_answer_relevance_clamps():
    assert answer_relevance_score(0.6) == pytest.approx(0.6)
    assert answer_relevance_score(1.3) == 1.0
    assert answer_relevance_score(-0.2) == 0.0


def test_context_utilization_fraction_used():
    # Two of four context sentences are echoed by the answer.
    scores = [0.8, 0.1, 0.6, 0.05]
    assert context_utilization_score(scores, threshold=0.5) == pytest.approx(0.5)


def test_context_utilization_empty_is_zero():
    assert context_utilization_score([], threshold=0.5) == 0.0

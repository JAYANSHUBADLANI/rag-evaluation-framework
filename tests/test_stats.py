"""Tests for bootstrap confidence intervals and the permutation test."""

from __future__ import annotations

import numpy as np
import pytest

from rag_eval.stats import (
    bootstrap_ci,
    compare_metric,
    paired_permutation_test,
)


def test_bootstrap_point_equals_mean():
    values = [0.0, 1.0, 1.0, 0.5, 0.25]
    result = bootstrap_ci(values, n_boot=1000, seed=1)
    assert result.point == pytest.approx(float(np.mean(values)))


def test_bootstrap_constant_collapses_interval():
    result = bootstrap_ci([1.0, 1.0, 1.0, 1.0], n_boot=500, seed=1)
    assert result.low == pytest.approx(1.0)
    assert result.high == pytest.approx(1.0)


def test_bootstrap_is_reproducible():
    a = bootstrap_ci([0, 1, 0, 1, 1, 0, 1], n_boot=2000, seed=42)
    b = bootstrap_ci([0, 1, 0, 1, 1, 0, 1], n_boot=2000, seed=42)
    assert (a.low, a.high) == (b.low, b.high)


def test_bootstrap_interval_brackets_point():
    result = bootstrap_ci([0.2, 0.4, 0.6, 0.8, 1.0], n_boot=2000, seed=3)
    assert result.low <= result.point <= result.high


def test_bootstrap_rejects_empty():
    with pytest.raises(ValueError):
        bootstrap_ci([], n_boot=100)


def test_permutation_identical_gives_p_one():
    values = [0.3, 0.6, 0.9, 0.1]
    result = paired_permutation_test(values, values, n_perm=500, seed=0)
    assert result.observed_diff == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)


def test_permutation_detects_clear_difference():
    a = [1.0] * 20
    b = [0.0] * 20
    result = paired_permutation_test(a, b, n_perm=2000, seed=0)
    assert result.observed_diff == pytest.approx(1.0)
    assert result.p_value < 0.01


def test_permutation_length_mismatch_raises():
    with pytest.raises(ValueError):
        paired_permutation_test([1.0, 2.0], [1.0], n_perm=100)


def test_permutation_p_value_bounds():
    rng = np.random.default_rng(0)
    a = rng.random(15)
    b = rng.random(15)
    result = paired_permutation_test(a, b, n_perm=1000, seed=1)
    assert 0.0 < result.p_value <= 1.0


def test_compare_metric_significant_winner():
    a = [1.0, 0.9, 1.0, 0.8, 1.0, 0.95, 0.85, 0.9]
    b = [0.2, 0.1, 0.3, 0.0, 0.2, 0.15, 0.05, 0.1]
    comparison = compare_metric(
        "recall@5", a, b, config_a="A", config_b="B",
        n_boot=1000, n_perm=2000, alpha=0.05, seed=0,
    )
    assert comparison.mean_a > comparison.mean_b
    assert comparison.diff > 0
    assert comparison.significant is True
    assert "A is significantly better" in comparison.verdict


def test_compare_metric_no_difference():
    values = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    comparison = compare_metric(
        "recall@5", values, values, config_a="A", config_b="B",
        n_boot=500, n_perm=500, alpha=0.05, seed=0,
    )
    assert comparison.diff == pytest.approx(0.0)
    assert comparison.significant is False
    assert "no significant difference" in comparison.verdict

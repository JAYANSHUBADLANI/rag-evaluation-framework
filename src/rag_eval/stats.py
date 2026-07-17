"""Statistical tooling for reporting and comparing evaluation results.

Point estimates on their own hide sampling noise: a Recall@5 of 0.82 measured
on 40 questions could plausibly be 0.72 or 0.90 on a different 40 questions.
This module quantifies that uncertainty and tests whether the gap between two
pipeline configurations is real.

It provides three things:

* :func:`bootstrap_ci` -- a percentile bootstrap confidence interval for the
  mean of a per-query metric.
* :func:`paired_permutation_test` -- an exact-style paired randomisation test
  (sign flipping) for the mean difference between two configurations evaluated
  on the *same* queries.
* :func:`compare_metric` -- combines the two into a single verdict object that
  the report generator renders.

Everything is implemented with :mod:`numpy` only and is fully seeded, so the
numbers are reproducible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    """A point estimate together with a bootstrap confidence interval."""

    point: float
    low: float
    high: float
    confidence: float
    n_boot: int

    @property
    def margin(self) -> float:
        """Half-width of the interval around the point estimate (mean side)."""
        return max(self.high - self.point, self.point - self.low)


@dataclass(frozen=True)
class PermutationResult:
    """Outcome of a paired permutation test on a mean difference."""

    observed_diff: float
    p_value: float
    n_perm: int
    alternative: str


@dataclass(frozen=True)
class MetricComparison:
    """Full comparison of one metric between configuration A and B."""

    metric: str
    config_a: str
    config_b: str
    mean_a: float
    mean_b: float
    diff: float
    diff_ci_low: float
    diff_ci_high: float
    p_value: float
    significant: bool
    verdict: str


def _as_1d(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one observation")
    return array


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    *,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile bootstrap confidence interval for the mean.

    Resamples ``values`` with replacement ``n_boot`` times, recomputes the mean
    of each resample and reads the empirical percentiles at
    ``(1 ± confidence) / 2``.

    Parameters
    ----------
    values:
        Per-query metric values.
    n_boot:
        Number of bootstrap resamples.
    confidence:
        Target coverage, e.g. ``0.95`` for a 95% interval.
    seed:
        Seed for the random generator, making the interval reproducible.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_boot <= 0:
        raise ValueError(f"n_boot must be positive, got {n_boot}")

    array = _as_1d(values, "values")
    n = array.size
    point = float(array.mean())

    if n == 1:
        # No variability to resample; the interval collapses to the point.
        return BootstrapResult(point, point, point, confidence, n_boot)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_boot, n))
    boot_means = array[indices].mean(axis=1)

    alpha = 1.0 - confidence
    low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return BootstrapResult(point, low, high, confidence, n_boot)


def paired_permutation_test(
    values_a: Sequence[float] | np.ndarray,
    values_b: Sequence[float] | np.ndarray,
    *,
    n_perm: int = 10_000,
    alternative: str = "two-sided",
    seed: int = 0,
) -> PermutationResult:
    """Paired permutation (randomisation) test for a mean difference.

    The two inputs are paired: ``values_a[i]`` and ``values_b[i]`` come from the
    same query ``i``. Under the null hypothesis the two configurations are
    exchangeable for each query, so the sign of each per-query difference is
    equally likely to be positive or negative. The test builds the null
    distribution by randomly flipping those signs ``n_perm`` times.

    A Monte-Carlo p-value with the standard ``(hits + 1) / (n_perm + 1)``
    correction is returned so the p-value is never exactly zero.

    Parameters
    ----------
    values_a, values_b:
        Equal-length paired per-query metric values.
    n_perm:
        Number of sign-flip permutations.
    alternative:
        ``"two-sided"``, ``"greater"`` (A > B) or ``"less"`` (A < B).
    seed:
        Seed for the random generator.
    """
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(f"unknown alternative {alternative!r}")
    if n_perm <= 0:
        raise ValueError(f"n_perm must be positive, got {n_perm}")

    array_a = _as_1d(values_a, "values_a")
    array_b = _as_1d(values_b, "values_b")
    if array_a.size != array_b.size:
        raise ValueError(
            "paired test requires equal length inputs, "
            f"got {array_a.size} and {array_b.size}"
        )

    diff = array_a - array_b
    observed = float(diff.mean())
    n = diff.size

    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(n_perm, n)) * 2 - 1
    perm_means = (signs * diff).mean(axis=1)

    if alternative == "two-sided":
        hits = int(np.count_nonzero(np.abs(perm_means) >= abs(observed)))
    elif alternative == "greater":
        hits = int(np.count_nonzero(perm_means >= observed))
    else:  # "less"
        hits = int(np.count_nonzero(perm_means <= observed))

    p_value = (hits + 1) / (n_perm + 1)
    return PermutationResult(observed, p_value, n_perm, alternative)


def compare_metric(
    metric: str,
    values_a: Sequence[float] | np.ndarray,
    values_b: Sequence[float] | np.ndarray,
    *,
    config_a: str,
    config_b: str,
    n_boot: int = 10_000,
    n_perm: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> MetricComparison:
    """Compare one metric between two configurations on paired queries.

    Computes both means, a bootstrap confidence interval on the *paired*
    difference and a two-sided permutation p-value, then renders a plain
    language verdict at significance level ``alpha`` (higher metric is better).
    """
    array_a = _as_1d(values_a, "values_a")
    array_b = _as_1d(values_b, "values_b")
    if array_a.size != array_b.size:
        raise ValueError("comparison requires equal length paired inputs")

    mean_a = float(array_a.mean())
    mean_b = float(array_b.mean())
    diff = mean_a - mean_b

    diff_ci = bootstrap_ci(
        array_a - array_b, n_boot=n_boot, confidence=1.0 - alpha, seed=seed
    )
    perm = paired_permutation_test(
        array_a, array_b, n_perm=n_perm, alternative="two-sided", seed=seed
    )
    significant = perm.p_value < alpha

    if significant:
        winner = config_a if diff > 0 else config_b
        verdict = (
            f"{winner} is significantly better "
            f"(delta={diff:+.4f}; p={perm.p_value:.4f})"
        )
    else:
        verdict = (
            f"no significant difference "
            f"(delta={diff:+.4f}; p={perm.p_value:.4f})"
        )

    return MetricComparison(
        metric=metric,
        config_a=config_a,
        config_b=config_b,
        mean_a=mean_a,
        mean_b=mean_b,
        diff=diff,
        diff_ci_low=diff_ci.low,
        diff_ci_high=diff_ci.high,
        p_value=perm.p_value,
        significant=significant,
        verdict=verdict,
    )

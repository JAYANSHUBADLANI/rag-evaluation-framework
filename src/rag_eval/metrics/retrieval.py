"""Retrieval quality metrics.

All functions operate on a ranked list of *retrieved identifiers* (highest
ranked first) and a set of *relevant identifiers*. Identifiers are opaque
strings; in the demo pipeline they are document ids obtained by collapsing a
ranked list of retrieved chunks onto their parent documents.

The functions are intentionally free of any embedding, model or framework
dependency so that every metric can be checked against a hand computed value.

Conventions
-----------
* ``retrieved`` is assumed to contain unique identifiers ordered by rank. When
  a metric only needs set membership (recall, hit rate) duplicates are handled
  gracefully anyway.
* ``k`` is one-indexed cut-off depth and must be a positive integer.
* Every metric returns a float in ``[0, 1]``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


def _check_k(k: int) -> None:
    if not isinstance(k, int):
        raise TypeError(f"k must be an int, got {type(k)!r}")
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Precision@k: fraction of the top ``k`` positions that are relevant.

    Defined as ``|{top-k retrieved} ∩ relevant| / k``. The denominator is the
    fixed cut-off ``k`` (the textbook definition), so retrieving fewer than
    ``k`` items is penalised.
    """
    _check_k(k)
    relevant_set = set(relevant)
    top_k = list(retrieved[:k])
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / k


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Recall@k: fraction of all relevant items found within the top ``k``.

    Defined as ``|{top-k retrieved} ∩ relevant| / |relevant|``. Returns ``0.0``
    when there are no relevant items (the metric is undefined but ``0.0`` keeps
    aggregation total-safe).
    """
    _check_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = set(retrieved[:k])
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)


def hit_rate_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Hit Rate@k: ``1.0`` if any relevant item is in the top ``k`` else ``0.0``.

    Averaged over queries this is the fraction of queries for which retrieval
    surfaced at least one relevant document (also called success@k).
    """
    _check_k(k)
    relevant_set = set(relevant)
    top_k = set(retrieved[:k])
    return 1.0 if top_k & relevant_set else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Reciprocal rank of the first relevant item (``0.0`` if none retrieved).

    The mean of this quantity over a set of queries is the Mean Reciprocal Rank
    (MRR).
    """
    relevant_set = set(relevant)
    for index, item in enumerate(retrieved):
        if item in relevant_set:
            return 1.0 / (index + 1)
    return 0.0


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """Discounted Cumulative Gain over the first ``k`` graded gains.

    ``DCG@k = Σ_{i=1..k} gain_i / log2(i + 1)`` with one-indexed positions.
    """
    _check_k(k)
    total = 0.0
    for index, gain in enumerate(gains[:k]):
        total += float(gain) / math.log2(index + 2)
    return total


def ndcg_at_k(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: int,
    gains: Mapping[str, float] | None = None,
) -> float:
    """Normalised DCG@k.

    With binary relevance (``gains is None``) every relevant identifier has a
    gain of ``1``. When ``gains`` is supplied it maps identifiers to graded
    relevance values and the ideal ranking is built from those grades.
    """
    _check_k(k)
    relevant_set = set(relevant)

    if gains is None:
        ranked_gains = [1.0 if item in relevant_set else 0.0 for item in retrieved[:k]]
        ideal_gains = [1.0] * min(len(relevant_set), k)
    else:
        ranked_gains = [float(gains.get(item, 0.0)) for item in retrieved[:k]]
        ideal_gains = sorted((float(g) for g in gains.values()), reverse=True)[:k]

    dcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(ranked_gains))
    idcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(ideal_gains))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def retrieval_metrics_for_query(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    ks: Sequence[int],
) -> dict[str, float]:
    """Compute every retrieval metric for a single query.

    Returns a flat mapping such as ``{"precision@1": .., "recall@5": ..,
    "ndcg@5": .., "hit_rate@3": .., "mrr": ..}``. The ``mrr`` key holds the
    per-query reciprocal rank; averaging it across queries yields MRR.
    """
    relevant_set = set(relevant)
    scores: dict[str, float] = {}
    for k in ks:
        scores[f"precision@{k}"] = precision_at_k(retrieved, relevant_set, k)
        scores[f"recall@{k}"] = recall_at_k(retrieved, relevant_set, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant_set, k)
        scores[f"hit_rate@{k}"] = hit_rate_at_k(retrieved, relevant_set, k)
    scores["mrr"] = reciprocal_rank(retrieved, relevant_set)
    return scores

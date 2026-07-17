"""Generation quality metric definitions.

These functions express the *scoring formulas* used to turn similarity
evidence into the three generation metrics. They deliberately take plain
numeric arrays rather than text or embeddings so that each formula can be unit
tested against hand computed values. The judges in :mod:`rag_eval.judges`
supply the evidence (a similarity matrix between answer sentences and context
sentences, plus an answer/question similarity) and call into these functions.

Metric summary
--------------
* **Faithfulness** -- fraction of answer sentences that are supported by at
  least one context sentence (grounding / hallucination proxy).
* **Answer relevance** -- similarity between the answer and the question,
  clamped to ``[0, 1]``.
* **Context utilization** -- fraction of retrieved context sentences that are
  echoed by the answer (retrieval efficiency proxy).
"""

from __future__ import annotations

import numpy as np


def max_support(similarity: np.ndarray) -> np.ndarray:
    """Best support score per row of a similarity matrix.

    Given an ``(m, n)`` matrix of similarities between ``m`` query units and
    ``n`` evidence units, return a length ``m`` vector holding, for each query
    unit, its maximum similarity to any evidence unit. An empty evidence axis
    yields an all-zero vector.
    """
    matrix = np.asarray(similarity, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"similarity must be 2-D, got shape {matrix.shape}")
    if matrix.shape[0] == 0:
        return np.zeros((0,), dtype=float)
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0],), dtype=float)
    return matrix.max(axis=1)


def faithfulness_score(answer_support: np.ndarray, threshold: float) -> float:
    """Fraction of answer sentences whose best support meets ``threshold``.

    ``answer_support[i]`` is the maximum similarity of answer sentence ``i`` to
    any context sentence. An answer with no sentences scores ``0.0``.
    """
    support = np.asarray(answer_support, dtype=float)
    if support.size == 0:
        return 0.0
    supported = np.count_nonzero(support >= threshold)
    return float(supported) / float(support.size)


def answer_relevance_score(answer_question_similarity: float) -> float:
    """Clamp an answer/question similarity into ``[0, 1]``.

    Non-negative embedding spaces (such as TF-IDF) already live in ``[0, 1]``;
    the clamp keeps the metric well defined for spaces that admit negative
    cosine values.
    """
    value = float(answer_question_similarity)
    return max(0.0, min(1.0, value))


def context_utilization_score(context_support: np.ndarray, threshold: float) -> float:
    """Fraction of context sentences that are echoed by the answer.

    ``context_support[j]`` is the maximum similarity of context sentence ``j``
    to any answer sentence. A high score means most of the retrieved context
    made it into the answer; a low score means retrieval returned material the
    answer never used. Empty context scores ``0.0``.
    """
    support = np.asarray(context_support, dtype=float)
    if support.size == 0:
        return 0.0
    used = np.count_nonzero(support >= threshold)
    return float(used) / float(support.size)

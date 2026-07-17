"""Offline generation judge based on embedding similarity heuristics.

This judge scores an answer without any external service. It embeds the
question, the answer sentences and the context sentences into a shared TF-IDF
space (fitted per example so it captures the lexical overlap that matters for
grounding) and turns the resulting cosine similarities into the three
generation metrics defined in :mod:`rag_eval.metrics.generation`.

The similarity-threshold heuristic is an entailment proxy: a sentence is
treated as "supported" when a context sentence is sufficiently close to it in
the embedding space. This is cheap and deterministic, which is what makes the
whole evaluation runnable offline.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from rag_eval.embeddings import Embedder, TfidfEmbedder, cosine_similarity_matrix
from rag_eval.metrics.generation import (
    answer_relevance_score,
    context_utilization_score,
    faithfulness_score,
    max_support,
)
from rag_eval.judges.base import GenerationScores, Judge
from rag_eval.text import flatten_sentences, split_sentences


def _default_embedder_factory() -> Embedder:
    """Build the per-example embedder used by the local judge.

    Raw TF-IDF (no LSA reduction) is used so that identical sentences map to
    identical vectors and therefore a cosine similarity of exactly ``1``.
    """
    return TfidfEmbedder(svd_dim=None)


class LocalJudge(Judge):
    """Score generation quality from embedding similarities, fully offline.

    Parameters
    ----------
    embedder_factory:
        Zero-argument callable returning a fresh :class:`Embedder`. A new
        embedder is fitted per example on that example's sentences. Defaults to
        raw TF-IDF.
    faithfulness_threshold:
        Minimum answer-to-context similarity for an answer sentence to count as
        grounded.
    utilization_threshold:
        Minimum context-to-answer similarity for a context sentence to count as
        used by the answer.
    """

    def __init__(
        self,
        *,
        embedder_factory: Callable[[], Embedder] | None = None,
        faithfulness_threshold: float = 0.5,
        utilization_threshold: float = 0.5,
    ) -> None:
        self._embedder_factory = embedder_factory or _default_embedder_factory
        self.faithfulness_threshold = faithfulness_threshold
        self.utilization_threshold = utilization_threshold

    def score(
        self, question: str, answer: str, contexts: Sequence[str]
    ) -> GenerationScores:
        """Score one triple; see :class:`~rag_eval.judges.base.Judge`."""
        answer_sentences = split_sentences(answer)
        context_sentences = flatten_sentences(contexts)

        if not answer_sentences:
            return GenerationScores(0.0, 0.0, 0.0)

        corpus = [question, *answer_sentences, *context_sentences]
        embedder = self._embedder_factory()
        embedder.fit(corpus)

        question_vector = embedder.encode([question])
        answer_vectors = embedder.encode(answer_sentences)

        # Answer relevance: mean cosine similarity of answer sentences to the
        # question.
        answer_question = cosine_similarity_matrix(answer_vectors, question_vector)[:, 0]
        relevance = answer_relevance_score(float(answer_question.mean()))

        if context_sentences:
            context_vectors = embedder.encode(context_sentences)
            answer_context = cosine_similarity_matrix(answer_vectors, context_vectors)
            answer_support = max_support(answer_context)
            context_support = max_support(answer_context.T)
        else:
            answer_support = np.zeros(len(answer_sentences), dtype=float)
            context_support = np.zeros(0, dtype=float)

        faithfulness = faithfulness_score(answer_support, self.faithfulness_threshold)
        utilization = context_utilization_score(
            context_support, self.utilization_threshold
        )
        return GenerationScores(faithfulness, relevance, utilization)

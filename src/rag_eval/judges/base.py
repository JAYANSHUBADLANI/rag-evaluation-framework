"""Shared types for generation-quality judges."""

from __future__ import annotations

import abc
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationScores:
    """The three generation-quality scores for a single answer.

    Every field lies in ``[0, 1]`` where higher is better.
    """

    faithfulness: float
    answer_relevance: float
    context_utilization: float

    def as_dict(self) -> dict[str, float]:
        """Return the scores as a plain mapping keyed by metric name."""
        return {
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_utilization": self.context_utilization,
        }


class Judge(abc.ABC):
    """Abstract base class for generation judges.

    A judge scores a generated ``answer`` against the ``question`` it answers
    and the ``contexts`` (retrieved passages) it was given.
    """

    @abc.abstractmethod
    def score(
        self, question: str, answer: str, contexts: Sequence[str]
    ) -> GenerationScores:
        """Score a single ``(question, answer, contexts)`` triple."""

    def score_batch(
        self, samples: Iterable[tuple[str, str, Sequence[str]]]
    ) -> list[GenerationScores]:
        """Score many triples, returning one :class:`GenerationScores` each."""
        return [self.score(question, answer, contexts) for question, answer, contexts in samples]

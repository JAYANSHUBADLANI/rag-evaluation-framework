"""Small, dependency-free text helpers shared across the package."""

from __future__ import annotations

import re
from collections.abc import Sequence

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\w+")


def word_tokens(text: str) -> list[str]:
    """Split text into whitespace-delimited word tokens.

    Used by the chunker to measure length in tokens. Whitespace splitting keeps
    chunk boundaries deterministic and independent of any external tokenizer.
    """
    return text.split()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on ``.``, ``!`` or ``?`` boundaries.

    A deliberately lightweight splitter: it avoids a natural-language toolkit
    dependency while being good enough for grounding heuristics on clean,
    encyclopedia-style prose. Empty fragments are dropped.
    """
    stripped = text.strip()
    if not stripped:
        return []
    parts = _SENTENCE_BOUNDARY.split(stripped)
    return [part.strip() for part in parts if part.strip()]


def has_content(text: str) -> bool:
    """Return ``True`` when the text contains at least one word character."""
    return bool(_WORD.search(text))


def flatten_sentences(texts: Sequence[str]) -> list[str]:
    """Split each item into sentences and concatenate the results."""
    sentences: list[str] = []
    for text in texts:
        sentences.extend(split_sentences(text))
    return sentences

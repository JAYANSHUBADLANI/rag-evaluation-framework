"""Generation-quality judges: an offline heuristic backend and an LLM backend."""

from __future__ import annotations

from rag_eval.judges.base import GenerationScores, Judge
from rag_eval.judges.llm import LLMJudge, build_judge_prompt, parse_scores
from rag_eval.judges.local import LocalJudge

__all__ = [
    "GenerationScores",
    "Judge",
    "LocalJudge",
    "LLMJudge",
    "build_judge_prompt",
    "parse_scores",
]


def build_judge(name: str, **kwargs) -> Judge:
    """Factory mapping a config name to a judge backend.

    ``"local"`` builds the offline embedding judge; ``"llm"`` builds the
    language-model judge (keyword arguments are forwarded to it).
    """
    key = name.lower()
    if key == "local":
        return LocalJudge(**kwargs)
    if key == "llm":
        return LLMJudge(**kwargs)
    raise ValueError(f"unknown judge backend {name!r}")

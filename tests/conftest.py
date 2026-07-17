"""Shared pytest fixtures: a tiny on-disk project and a path to the demo data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_eval.config import ExperimentConfig, RetrievalConfig

_TINY_DOCS = {
    "apple": (
        "Apples are sweet fruits that grow on apple trees in orchards. "
        "They come in many colours such as red, green, and yellow."
    ),
    "python": (
        "Python is a popular programming language used for data analysis and "
        "web development. It is known for readable, simple syntax."
    ),
    "everest": (
        "Mount Everest is the tallest mountain on Earth and lies in the "
        "Himalaya range on the border of Nepal and Tibet."
    ),
    "ocean": (
        "The ocean is a large body of salt water covering most of the surface "
        "of the planet. It is home to countless marine species."
    ),
    "guitar": (
        "A guitar is a stringed musical instrument played by plucking or "
        "strumming its strings with the fingers or a pick."
    ),
}

_TINY_EVAL = [
    {"id": "t1", "question": "Which fruit grows on trees in orchards?", "relevant_doc_ids": ["apple"]},
    {"id": "t2", "question": "What programming language is used for data analysis?", "relevant_doc_ids": ["python"]},
    {"id": "t3", "question": "What is the tallest mountain on Earth?", "relevant_doc_ids": ["everest"]},
    {"id": "t4", "question": "What covers most of the planet with salt water?", "relevant_doc_ids": ["ocean"]},
    {"id": "t5", "question": "Which stringed instrument is played by strumming?", "relevant_doc_ids": ["guitar"]},
    {"id": "t6", "question": "Where do apples grow?", "relevant_doc_ids": ["apple"]},
]


@pytest.fixture()
def demo_dir() -> Path:
    """Path to the repository's demo data directory."""
    return Path(__file__).resolve().parents[1] / "demo"


@pytest.fixture()
def tiny_project(tmp_path: Path) -> dict[str, Path]:
    """Write a small corpus and eval set to disk and return their paths."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    for doc_id, text in _TINY_DOCS.items():
        (corpus_dir / f"{doc_id}.txt").write_text(text + "\n", encoding="utf-8")

    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps(_TINY_EVAL, indent=2), encoding="utf-8")

    output_dir = tmp_path / "results"
    return {"corpus": corpus_dir, "eval": eval_path, "output": output_dir}


@pytest.fixture()
def tiny_config(tiny_project: dict[str, Path]) -> ExperimentConfig:
    """A fast, deterministic experiment config over the tiny project."""
    return ExperimentConfig(
        corpus_path=tiny_project["corpus"],
        eval_path=tiny_project["eval"],
        output_dir=tiny_project["output"],
        configs=[
            RetrievalConfig(name="small", chunk_size=8, overlap=2, top_k=3),
            RetrievalConfig(name="large", chunk_size=16, overlap=2, top_k=3),
        ],
        comparisons=[("small", "large")],
        embedder_name="tfidf",
        embedder_kwargs={"svd_dim": None},
        index_kind="faiss",
        judge_name="local",
        judge_kwargs={},
        ks=(1, 3),
        n_boot=400,
        n_perm=400,
        alpha=0.05,
        answer_sentences=2,
        primary_metric="recall@3",
        seed=7,
    )

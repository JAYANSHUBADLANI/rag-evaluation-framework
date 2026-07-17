"""Tests for corpus and evaluation-set loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval.dataset import (
    Document,
    EvalExample,
    load_corpus,
    load_eval_set,
    validate_eval_set,
)


def test_load_corpus_reads_all_documents(tiny_project):
    corpus = load_corpus(tiny_project["corpus"])
    doc_ids = [d.doc_id for d in corpus]
    assert doc_ids == sorted(doc_ids)  # deterministic ordering
    assert set(doc_ids) == {"apple", "python", "everest", "ocean", "guitar"}


def test_load_corpus_missing_directory():
    with pytest.raises(NotADirectoryError):
        load_corpus(Path("/no/such/corpus/dir"))


def test_load_and_validate_eval_set(tiny_project):
    corpus = load_corpus(tiny_project["corpus"])
    examples = load_eval_set(tiny_project["eval"])
    assert len(examples) == 6
    validate_eval_set(examples, corpus)  # should not raise


def test_validate_rejects_unknown_document():
    corpus = [Document("apple", "text")]
    examples = [EvalExample("q1", "Q?", ("banana",))]
    with pytest.raises(ValueError, match="unknown documents"):
        validate_eval_set(examples, corpus)


def test_validate_rejects_duplicate_ids():
    corpus = [Document("apple", "text")]
    examples = [
        EvalExample("dup", "Q1?", ("apple",)),
        EvalExample("dup", "Q2?", ("apple",)),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        validate_eval_set(examples, corpus)


def test_validate_rejects_empty_relevant():
    corpus = [Document("apple", "text")]
    examples = [EvalExample("q1", "Q?", ())]
    with pytest.raises(ValueError, match="no relevant"):
        validate_eval_set(examples, corpus)


def test_demo_dataset_is_valid(demo_dir):
    corpus = load_corpus(demo_dir / "corpus")
    examples = load_eval_set(demo_dir / "eval_set.json")
    assert len(corpus) == 20
    assert len(examples) == 40
    validate_eval_set(examples, corpus)  # the shipped demo data is clean

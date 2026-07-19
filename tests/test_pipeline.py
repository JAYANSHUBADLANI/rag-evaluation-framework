"""Tests for chunking, vector indexes, retrieval and the extractive answerer."""

from __future__ import annotations

import numpy as np
import pytest

from rag_eval.config import RetrievalConfig
from rag_eval.embeddings import TfidfEmbedder, l2_normalize
from rag_eval.pipeline import (
    Chunk,
    CrossEncoderReranker,
    FaissIndex,
    NumpyIndex,
    Retriever,
    chunk_document,
    extractive_answer,
    ranked_doc_ids,
    RetrievedChunk,
)

_MINI_CORPUS = {
    "apple": "Apples are sweet fruits that grow on trees in orchards.",
    "everest": "Mount Everest is the tallest mountain on Earth in the Himalaya.",
    "guitar": "A guitar is a stringed instrument played by strumming.",
}


def _mini_chunks() -> list[Chunk]:
    return [Chunk(f"{doc}#0", doc, text) for doc, text in _MINI_CORPUS.items()]


def test_chunk_document_boundaries_and_ids():
    text = " ".join(str(i) for i in range(10))
    chunks = chunk_document("doc", text, chunk_size=4, overlap=1)
    assert [c.chunk_id for c in chunks] == ["doc#0", "doc#1", "doc#2"]
    assert chunks[0].text == "0 1 2 3"  # first window
    assert chunks[1].text == "3 4 5 6"  # step = chunk_size - overlap = 3
    assert chunks[2].text == "6 7 8 9"
    assert all(c.doc_id == "doc" for c in chunks)


def test_chunk_document_short_text_single_chunk():
    chunks = chunk_document("doc", "only three words", chunk_size=50)
    assert len(chunks) == 1
    assert chunks[0].text == "only three words"


def test_chunk_document_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_document("doc", "a b c d", chunk_size=4, overlap=4)


def test_ranked_doc_ids_dedupes_preserving_order():
    retrieved = [
        RetrievedChunk(Chunk("a#0", "a", "x"), 0.9),
        RetrievedChunk(Chunk("a#1", "a", "y"), 0.8),
        RetrievedChunk(Chunk("b#0", "b", "z"), 0.7),
    ]
    assert ranked_doc_ids(retrieved) == ["a", "b"]


def test_numpy_and_faiss_indexes_agree():
    rng = np.random.default_rng(0)
    vectors = l2_normalize(rng.random((6, 8)).astype(np.float32))
    queries = vectors[:1]

    numpy_index = NumpyIndex(8)
    numpy_index.build(vectors)
    faiss_index = FaissIndex(8)
    faiss_index.build(vectors)

    n_idx, n_scores = numpy_index.search(queries, 3)
    f_idx, f_scores = faiss_index.search(queries, 3)

    assert n_idx[0][0] == 0 and f_idx[0][0] == 0  # nearest to itself
    assert list(n_idx[0]) == list(f_idx[0])
    assert n_scores[0][0] == pytest.approx(1.0, abs=1e-5)
    assert f_scores[0][0] == pytest.approx(1.0, abs=1e-5)


def test_numpy_index_caps_k_at_corpus_size():
    index = NumpyIndex(4)
    index.build(l2_normalize(np.eye(3, 4).astype(np.float32)))
    indices, scores = index.search(np.eye(1, 4).astype(np.float32), k=10)
    assert indices.shape[1] == 3  # only three vectors available


@pytest.mark.parametrize("index_kind", ["numpy", "faiss"])
def test_retriever_finds_relevant_document(index_kind):
    retriever = Retriever(TfidfEmbedder(svd_dim=None), index_kind=index_kind)
    retriever.index(_mini_chunks())
    results = retriever.retrieve("What is the tallest mountain on Earth?", k=1)
    assert results[0].chunk.doc_id == "everest"


def test_extractive_answer_selects_relevant_sentence():
    embedder = TfidfEmbedder(svd_dim=None)
    chunks = _mini_chunks()
    retriever = Retriever(embedder, index_kind="numpy")
    retriever.index(chunks)
    retrieved = retriever.retrieve("Which instrument is played by strumming?", k=3)
    answer = extractive_answer("Which instrument is played by strumming?", retrieved, embedder, max_sentences=1)
    assert "guitar" in answer.lower()


class _StubCrossEncoder:
    """Offline stand-in for a cross-encoder: scores by shared-word count."""

    def predict(self, pairs):
        return [
            float(len(set(query.lower().split()) & set(text.lower().split())))
            for query, text in pairs
        ]


def test_reranker_reorders_by_cross_encoder_score():
    retrieved = [
        RetrievedChunk(Chunk("a#0", "a", "nothing relevant here"), 0.9),
        RetrievedChunk(Chunk("b#0", "b", "the tallest mountain is everest"), 0.5),
    ]
    reranker = CrossEncoderReranker()
    reranker._model = _StubCrossEncoder()  # keep the test offline
    result = reranker.rerank("what is the tallest mountain", retrieved)
    assert [item.chunk.doc_id for item in result] == ["b", "a"]
    assert result[0].score > result[1].score


def test_reranker_empty_input_returns_empty():
    reranker = CrossEncoderReranker()
    assert reranker.rerank("anything", []) == []


def test_retrieval_config_rejects_rerank_candidates_below_top_k():
    with pytest.raises(ValueError):
        RetrievalConfig(
            name="bad",
            chunk_size=128,
            overlap=0,
            top_k=5,
            rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            rerank_candidates=3,
        )


def test_retrieval_config_allows_rerank_candidates_at_or_above_top_k():
    config = RetrievalConfig(
        name="good",
        chunk_size=128,
        overlap=0,
        top_k=5,
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_candidates=12,
    )
    assert config.rerank_candidates == 12

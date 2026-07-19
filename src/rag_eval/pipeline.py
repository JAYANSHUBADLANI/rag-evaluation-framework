"""A small, self-contained retrieval-augmented generation pipeline.

The pipeline exists so the framework has a real system to evaluate. It is
deliberately simple and fully offline:

1. :func:`chunk_document` splits documents into overlapping token windows.
2. :class:`Retriever` embeds the chunks, stores them in a vector index (FAISS
   by default, with a NumPy fallback) and returns the top ``k`` chunks for a
   query.
3. :func:`extractive_answer` composes an answer by selecting the retrieved
   sentences most similar to the question -- a grounded generator that needs no
   language-model API, which keeps the demo reproducible.

Because the answerer is extractive, generated answers are grounded in the
retrieved context by construction; the generation metrics correctly reflect
this. The language-model judge (:mod:`rag_eval.judges.llm`) is provided for
evaluating genuinely generative pipelines.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rag_eval.embeddings import Embedder, cosine_similarity_matrix
from rag_eval.text import split_sentences, word_tokens


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of a source document."""

    chunk_id: str
    doc_id: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by the retriever together with its similarity score."""

    chunk: Chunk
    score: float


def chunk_document(
    doc_id: str,
    text: str,
    chunk_size: int,
    overlap: int = 0,
) -> list[Chunk]:
    """Split a document into overlapping windows of ``chunk_size`` tokens.

    Consecutive windows advance by ``chunk_size - overlap`` tokens. A document
    shorter than one window becomes a single chunk. Chunk ids have the form
    ``"{doc_id}#{index}"``.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            f"overlap must satisfy 0 <= overlap < chunk_size, got {overlap}"
        )

    tokens = word_tokens(text)
    if not tokens:
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    start = 0
    index = 0
    n_tokens = len(tokens)
    while start < n_tokens:
        window = tokens[start : start + chunk_size]
        chunks.append(Chunk(f"{doc_id}#{index}", doc_id, " ".join(window)))
        index += 1
        if start + chunk_size >= n_tokens:
            break
        start += step
    return chunks


class NumpyIndex:
    """Brute-force inner-product index using NumPy.

    A dependency-light alternative to FAISS with identical semantics for
    normalised vectors (inner product equals cosine similarity). Used to keep
    the pipeline portable and to exercise both index code paths in the tests.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors: np.ndarray | None = None

    def build(self, vectors: np.ndarray) -> None:
        """Store the corpus vectors to search against."""
        array = np.ascontiguousarray(vectors, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.dim:
            raise ValueError(
                f"expected vectors with {self.dim} columns, got {array.shape}"
            )
        self._vectors = array

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(indices, scores)`` for the top ``k`` matches per query."""
        if self._vectors is None:
            raise RuntimeError("NumpyIndex.search called before build")
        query = np.ascontiguousarray(queries, dtype=np.float32)
        sims = query @ self._vectors.T
        k_eff = min(k, self._vectors.shape[0])
        order = np.argsort(-sims, axis=1, kind="stable")[:, :k_eff]
        scores = np.take_along_axis(sims, order, axis=1)
        return order, scores


class FaissIndex:
    """A thin wrapper around a FAISS flat inner-product index."""

    def __init__(self, dim: int) -> None:
        import faiss

        self._faiss = faiss
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)

    def build(self, vectors: np.ndarray) -> None:
        """Reset the index and add the corpus vectors."""
        array = np.ascontiguousarray(vectors, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.dim:
            raise ValueError(
                f"expected vectors with {self.dim} columns, got {array.shape}"
            )
        self._index.reset()
        self._index.add(array)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(indices, scores)`` for the top ``k`` matches per query."""
        query = np.ascontiguousarray(queries, dtype=np.float32)
        k_eff = min(k, self._index.ntotal)
        scores, indices = self._index.search(query, k_eff)
        return indices, scores


def build_index(kind: str, dim: int):
    """Build a vector index of the requested ``kind`` (``"faiss"``/``"numpy"``)."""
    key = kind.lower()
    if key == "faiss":
        return FaissIndex(dim)
    if key == "numpy":
        return NumpyIndex(dim)
    raise ValueError(f"unknown index kind {kind!r}")


class Retriever:
    """Embeds a chunked corpus into a vector index and retrieves by similarity."""

    def __init__(self, embedder: Embedder, index_kind: str = "faiss") -> None:
        self.embedder = embedder
        self.index_kind = index_kind
        self._chunks: list[Chunk] = []
        self._index = None

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Fit the embedder on the chunks and populate the vector index."""
        if not chunks:
            raise ValueError("cannot index an empty chunk list")
        self._chunks = list(chunks)
        texts = [chunk.text for chunk in self._chunks]
        self.embedder.fit(texts)
        vectors = self.embedder.encode(texts)
        dim = int(vectors.shape[1])
        self._index = build_index(self.index_kind, dim)
        self._index.build(vectors)

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return the top ``k`` chunks for ``query`` ranked by similarity."""
        if self._index is None:
            raise RuntimeError("Retriever.retrieve called before index")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        query_vector = self.embedder.encode([query])
        indices, scores = self._index.search(query_vector, k)
        results: list[RetrievedChunk] = []
        for position, score in zip(indices[0].tolist(), scores[0].tolist()):
            if position < 0:
                continue
            results.append(RetrievedChunk(self._chunks[position], float(score)))
        return results

    @property
    def chunks(self) -> list[Chunk]:
        """The indexed chunks."""
        return list(self._chunks)


class CrossEncoderReranker:
    """Optional second-stage reranker backed by a cross-encoder.

    A cross-encoder scores each (query, chunk) pair jointly instead of
    comparing pre-computed embeddings, which is slower but considerably more
    precise. The intended use is two-stage retrieval: over-retrieve a candidate
    pool with the vector index, rerank it here, then keep the top ``k``.

    The heavy dependency and its model weights are only imported and loaded on
    first use, mirroring :class:`~rag_eval.embeddings.SentenceTransformerEmbedder`,
    so the framework stays fully offline unless a reranker is explicitly
    configured.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)

    def rerank(
        self, query: str, retrieved: Sequence[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Re-score ``retrieved`` against ``query`` and return them re-ordered.

        Scores come from the cross-encoder; the sort is stable so ties keep
        their original (vector-similarity) order. The returned
        :class:`RetrievedChunk` scores are the cross-encoder scores.
        """
        if not retrieved:
            return []
        self._ensure_model()
        assert self._model is not None
        pairs = [(query, item.chunk.text) for item in retrieved]
        scores = self._model.predict(pairs)
        order = np.argsort(-np.asarray(scores, dtype=np.float32), kind="stable")
        return [
            RetrievedChunk(retrieved[int(i)].chunk, float(scores[int(i)]))
            for i in order
        ]


def ranked_doc_ids(retrieved: Sequence[RetrievedChunk]) -> list[str]:
    """Collapse a ranked chunk list onto a ranked list of unique document ids.

    The first (highest ranked) chunk of each document sets that document's rank.
    This is what lets the retrieval metrics judge relevance at the document
    level and compare chunking strategies fairly.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for item in retrieved:
        doc_id = item.chunk.doc_id
        if doc_id not in seen:
            seen.add(doc_id)
            ordered.append(doc_id)
    return ordered


def extractive_answer(
    question: str,
    retrieved: Sequence[RetrievedChunk],
    embedder: Embedder,
    *,
    max_sentences: int = 2,
) -> str:
    """Compose an answer from the retrieved sentences closest to the question.

    Candidate sentences are gathered from the retrieved chunks (de-duplicated),
    scored by cosine similarity to the question in the embedder's space, and the
    top ``max_sentences`` are returned in their original reading order.
    """
    if max_sentences <= 0:
        raise ValueError(f"max_sentences must be positive, got {max_sentences}")

    candidates: list[str] = []
    seen: set[str] = set()
    for item in retrieved:
        for sentence in split_sentences(item.chunk.text):
            key = sentence.lower()
            if key not in seen:
                seen.add(key)
                candidates.append(sentence)

    if not candidates:
        return ""

    question_vector = embedder.encode([question])
    candidate_vectors = embedder.encode(candidates)
    sims = cosine_similarity_matrix(question_vector, candidate_vectors)[0]
    top = np.argsort(-sims, kind="stable")[:max_sentences]
    chosen = sorted(int(i) for i in top)
    return " ".join(candidates[i] for i in chosen)

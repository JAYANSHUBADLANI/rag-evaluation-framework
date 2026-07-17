"""Embedding backends and cosine-similarity helpers.

The retrieval pipeline and the local judge both need to turn text into vectors
and compare them. Two backends implement the :class:`Embedder` protocol:

* :class:`TfidfEmbedder` -- a fully offline TF-IDF (optionally reduced with
  latent semantic analysis) backend. It requires no downloads, is
  deterministic and is the default so the whole project runs without network
  access or API keys.
* :class:`SentenceTransformerEmbedder` -- an optional dense backend that wraps
  ``sentence-transformers``. The import is lazy so the dependency is only
  needed when this backend is explicitly selected.

Both backends return L2-normalised ``float32`` vectors, so an inner product is
exactly the cosine similarity.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return a copy of ``matrix`` with each row scaled to unit L2 norm.

    Zero rows are left as zeros (their norm is clamped by ``eps``), which makes
    downstream cosine similarities with empty text well defined at ``0``.
    """
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, eps)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between every row of ``a`` and every row of ``b``.

    Robust to zero-norm rows (which yield a similarity of ``0``). Returns an
    ``(len(a), len(b))`` matrix.
    """
    a_norm = l2_normalize(a)
    b_norm = l2_normalize(b)
    return (a_norm @ b_norm.T).astype(np.float32)


@runtime_checkable
class Embedder(Protocol):
    """Protocol for text embedding backends.

    ``fit`` is called once with the corpus to be indexed (a no-op for
    pretrained models); ``encode`` maps texts into the fitted vector space and
    returns L2-normalised ``float32`` vectors.
    """

    def fit(self, texts: Sequence[str]) -> None:
        ...

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        ...

    @property
    def dimension(self) -> int:
        ...


class TfidfEmbedder:
    """Offline TF-IDF (+ optional LSA) embedding backend.

    The vectoriser is fitted on the supplied corpus. When ``svd_dim`` is set and
    the corpus is large enough, a :class:`~sklearn.decomposition.TruncatedSVD`
    projection (latent semantic analysis) produces dense, lower-dimensional
    vectors; otherwise the raw TF-IDF weights are used. Cosine similarity in the
    resulting space is invariant to the sign ambiguity of the SVD basis, so a
    fixed ``random_state`` makes the backend reproducible.
    """

    def __init__(
        self,
        svd_dim: int | None = 128,
        *,
        min_df: int = 1,
        ngram_range: tuple[int, int] = (1, 1),
        random_state: int = 0,
    ) -> None:
        self.svd_dim = svd_dim
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.random_state = random_state
        self._vectorizer = None
        self._svd = None
        self._dimension = 0

    def fit(self, texts: Sequence[str]) -> None:
        """Fit the TF-IDF vocabulary (and optional LSA projection)."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        corpus = list(texts)
        if not corpus:
            raise ValueError("cannot fit TfidfEmbedder on an empty corpus")

        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            min_df=self.min_df,
            ngram_range=self.ngram_range,
        )
        tfidf = self._vectorizer.fit_transform(corpus)
        n_samples, n_features = tfidf.shape

        components = 0
        if self.svd_dim:
            components = min(self.svd_dim, n_features - 1, n_samples - 1)

        if components >= 1:
            from sklearn.decomposition import TruncatedSVD

            self._svd = TruncatedSVD(
                n_components=components, random_state=self.random_state
            )
            self._svd.fit(tfidf)
            self._dimension = components
        else:
            self._svd = None
            self._dimension = n_features

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Transform texts into the fitted, L2-normalised vector space."""
        if self._vectorizer is None:
            raise RuntimeError("TfidfEmbedder.encode called before fit")
        tfidf = self._vectorizer.transform(list(texts))
        if self._svd is not None:
            dense = self._svd.transform(tfidf)
        else:
            dense = tfidf.toarray()
        return l2_normalize(np.asarray(dense, dtype=np.float32))

    @property
    def dimension(self) -> int:
        """Dimensionality of the encoded vectors (valid after :meth:`fit`)."""
        return self._dimension


class SentenceTransformerEmbedder:
    """Optional dense backend backed by ``sentence-transformers``.

    The heavy dependency and its model weights are only imported and loaded on
    first use, so selecting this backend is opt-in. Requires the model to be
    available locally or downloadable at runtime.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dimension = 0

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._dimension = int(self._model.get_sentence_embedding_dimension())

    def fit(self, texts: Sequence[str]) -> None:
        """Load the model. No corpus fitting is required for a pretrained model."""
        self._ensure_model()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts with the sentence transformer, L2-normalised."""
        self._ensure_model()
        assert self._model is not None
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    @property
    def dimension(self) -> int:
        """Embedding dimensionality (valid after the model is loaded)."""
        return self._dimension


def build_embedder(name: str, **kwargs) -> Embedder:
    """Factory mapping a config name to an embedding backend.

    ``"tfidf"`` builds the offline backend; ``"sentence-transformers"`` (aliased
    ``"st"``) builds the dense backend.
    """
    key = name.lower().replace("_", "-")
    if key == "tfidf":
        return TfidfEmbedder(**kwargs)
    if key in {"sentence-transformers", "st"}:
        return SentenceTransformerEmbedder(**kwargs)
    raise ValueError(f"unknown embedder backend {name!r}")

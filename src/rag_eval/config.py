"""Configuration dataclasses and YAML loading for the experiment runner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RetrievalConfig:
    """One retrieval configuration to evaluate.

    ``embedder_name`` optionally overrides the experiment-wide embedder for this
    configuration only, which is what makes embedding-capacity ablations
    possible alongside chunk-size and top-k sweeps.
    """

    name: str
    chunk_size: int
    overlap: int
    top_k: int
    embedder_name: str | None = None
    embedder_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    """Everything needed to run and compare a set of retrieval configurations."""

    corpus_path: Path
    eval_path: Path
    output_dir: Path
    configs: list[RetrievalConfig]
    comparisons: list[tuple[str, str]]
    embedder_name: str = "tfidf"
    embedder_kwargs: dict[str, Any] = field(default_factory=dict)
    index_kind: str = "faiss"
    judge_name: str = "local"
    judge_kwargs: dict[str, Any] = field(default_factory=dict)
    ks: tuple[int, ...] = (1, 3, 5)
    n_boot: int = 10_000
    n_perm: int = 10_000
    alpha: float = 0.05
    answer_sentences: int = 2
    primary_metric: str = "recall@5"
    seed: int = 0

    def config_by_name(self, name: str) -> RetrievalConfig:
        """Return the retrieval configuration with the given ``name``."""
        for config in self.configs:
            if config.name == name:
                return config
        raise KeyError(f"no configuration named {name!r}")


def _parse_retrieval_config(raw: Mapping[str, Any]) -> RetrievalConfig:
    embedder = raw.get("embedder")
    if embedder is None:
        embedder_name, embedder_kwargs = None, {}
    else:
        embedder_name, embedder_kwargs = _split_named_block(embedder, "tfidf")
    return RetrievalConfig(
        name=str(raw["name"]),
        chunk_size=int(raw["chunk_size"]),
        overlap=int(raw.get("overlap", 0)),
        top_k=int(raw["top_k"]),
        embedder_name=embedder_name,
        embedder_kwargs=embedder_kwargs,
    )


def _split_named_block(raw: Any, default_name: str) -> tuple[str, dict[str, Any]]:
    """Split a ``{name: ..., **kwargs}`` block into its name and keyword args."""
    if raw is None:
        return default_name, {}
    block = dict(raw)
    name = str(block.pop("name", default_name))
    return name, block


def config_from_dict(raw: Mapping[str, Any], *, base_dir: Path) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` from a parsed mapping.

    Relative ``corpus_path``, ``eval_path`` and ``output_dir`` are resolved
    against ``base_dir``.
    """
    def _resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (base_dir / path)

    configs = [_parse_retrieval_config(item) for item in raw["configs"]]
    if not configs:
        raise ValueError("at least one retrieval configuration is required")

    comparisons: list[tuple[str, str]] = []
    for pair in raw.get("comparisons", []):
        if len(pair) != 2:
            raise ValueError(f"each comparison must have exactly two names: {pair}")
        comparisons.append((str(pair[0]), str(pair[1])))

    embedder_name, embedder_kwargs = _split_named_block(raw.get("embedder"), "tfidf")
    judge_name, judge_kwargs = _split_named_block(raw.get("judge"), "local")

    return ExperimentConfig(
        corpus_path=_resolve(raw["corpus_path"]),
        eval_path=_resolve(raw["eval_path"]),
        output_dir=_resolve(raw.get("output_dir", "results")),
        configs=configs,
        comparisons=comparisons,
        embedder_name=embedder_name,
        embedder_kwargs=embedder_kwargs,
        index_kind=str(raw.get("index_kind", "faiss")),
        judge_name=judge_name,
        judge_kwargs=judge_kwargs,
        ks=tuple(int(k) for k in raw.get("ks", (1, 3, 5))),
        n_boot=int(raw.get("n_boot", 10_000)),
        n_perm=int(raw.get("n_perm", 10_000)),
        alpha=float(raw.get("alpha", 0.05)),
        answer_sentences=int(raw.get("answer_sentences", 2)),
        primary_metric=str(raw.get("primary_metric", "recall@5")),
        seed=int(raw.get("seed", 0)),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a YAML file.

    Paths inside the file are resolved relative to the current working
    directory (run the CLI from the project root).
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"config file {config_path} must contain a mapping")
    return config_from_dict(raw, base_dir=Path.cwd())

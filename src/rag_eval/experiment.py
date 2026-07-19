"""Experiment runner: evaluate and compare retrieval configurations.

The runner ties the whole framework together. For each retrieval configuration
it chunks the corpus, builds a retriever, computes per-query retrieval and
generation metrics, and summarises each metric with a bootstrap confidence
interval. It then runs paired permutation tests for every requested pair of
configurations and writes machine-readable CSVs plus a human-readable report.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from rag_eval.config import ExperimentConfig, RetrievalConfig
from rag_eval.dataset import (
    Document,
    EvalExample,
    load_corpus,
    load_eval_set,
    validate_eval_set,
)
from rag_eval.embeddings import build_embedder
from rag_eval.judges import Judge, build_judge
from rag_eval.metrics.retrieval import retrieval_metrics_for_query
from rag_eval.pipeline import (
    CrossEncoderReranker,
    Retriever,
    chunk_document,
    extractive_answer,
    ranked_doc_ids,
)
from rag_eval.stats import BootstrapResult, MetricComparison, bootstrap_ci, compare_metric

GENERATION_METRICS: tuple[str, ...] = (
    "faithfulness",
    "answer_relevance",
    "context_utilization",
)


def retrieval_metric_names(ks: tuple[int, ...]) -> list[str]:
    """Ordered retrieval metric names for the requested cut-off depths."""
    names: list[str] = []
    for k in ks:
        names.extend(
            [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", f"hit_rate@{k}"]
        )
    names.append("mrr")
    return names


def all_metric_names(ks: tuple[int, ...]) -> list[str]:
    """All metric names (retrieval followed by generation), in report order."""
    return [*retrieval_metric_names(ks), *GENERATION_METRICS]


@dataclass
class ConfigRun:
    """Per-configuration results: per-query values and their summaries."""

    config: RetrievalConfig
    example_ids: list[str]
    per_query: dict[str, list[float]]
    summaries: dict[str, BootstrapResult]
    n_chunks: int
    embedder_label: str


def _resolve_embedder(cfg: ExperimentConfig, retrieval_config: RetrievalConfig):
    """Return the embedder and a human-readable label for one configuration.

    A per-configuration override takes full precedence over the experiment-wide
    embedder, enabling embedding-capacity ablations.
    """
    if retrieval_config.embedder_name is not None:
        name = retrieval_config.embedder_name
        kwargs = retrieval_config.embedder_kwargs
    else:
        name = cfg.embedder_name
        kwargs = cfg.embedder_kwargs
    detail = ", ".join(f"{key}={value}" for key, value in kwargs.items())
    label = f"{name}({detail})" if detail else name
    return build_embedder(name, **kwargs), label


@dataclass
class ExperimentResult:
    """The full outcome of an experiment across all configurations."""

    config: ExperimentConfig
    corpus_size: int
    eval_size: int
    runs: dict[str, ConfigRun]
    comparisons: dict[tuple[str, str], list[MetricComparison]]
    metric_names: list[str]


def run_config(
    cfg: ExperimentConfig,
    retrieval_config: RetrievalConfig,
    corpus: list[Document],
    examples: list[EvalExample],
    judge: Judge,
) -> ConfigRun:
    """Evaluate a single retrieval configuration over the whole eval set."""
    chunks = []
    for document in corpus:
        chunks.extend(
            chunk_document(
                document.doc_id,
                document.text,
                retrieval_config.chunk_size,
                retrieval_config.overlap,
            )
        )

    embedder, embedder_label = _resolve_embedder(cfg, retrieval_config)
    retriever = Retriever(embedder, cfg.index_kind)
    retriever.index(chunks)

    reranker = None
    if retrieval_config.rerank_model is not None:
        reranker = CrossEncoderReranker(retrieval_config.rerank_model)
        embedder_label += f" + rerank({retrieval_config.rerank_model})"

    metric_names = all_metric_names(cfg.ks)
    per_query: dict[str, list[float]] = {name: [] for name in metric_names}
    example_ids: list[str] = []
    judge_samples: list[tuple[str, str, list[str]]] = []

    for example in examples:
        if reranker is not None:
            candidates = retriever.retrieve(
                example.question, retrieval_config.rerank_candidates
            )
            retrieved = reranker.rerank(example.question, candidates)
            retrieved = retrieved[: retrieval_config.top_k]
        else:
            retrieved = retriever.retrieve(example.question, retrieval_config.top_k)
        doc_ranking = ranked_doc_ids(retrieved)
        query_metrics = retrieval_metrics_for_query(
            doc_ranking, example.relevant_doc_ids, cfg.ks
        )
        for name, value in query_metrics.items():
            per_query[name].append(value)

        answer = extractive_answer(
            example.question,
            retrieved,
            embedder,
            max_sentences=cfg.answer_sentences,
        )
        contexts = [item.chunk.text for item in retrieved]
        judge_samples.append((example.question, answer, contexts))
        example_ids.append(example.example_id)

    for scores in judge.score_batch(judge_samples):
        values = scores.as_dict()
        for name in GENERATION_METRICS:
            per_query[name].append(values[name])

    summaries = {
        name: bootstrap_ci(
            per_query[name],
            n_boot=cfg.n_boot,
            confidence=1.0 - cfg.alpha,
            seed=cfg.seed,
        )
        for name in metric_names
    }
    return ConfigRun(
        config=retrieval_config,
        example_ids=example_ids,
        per_query=per_query,
        summaries=summaries,
        n_chunks=len(chunks),
        embedder_label=embedder_label,
    )


def run_experiment(cfg: ExperimentConfig) -> ExperimentResult:
    """Run every configuration and every requested pairwise comparison."""
    corpus = load_corpus(cfg.corpus_path)
    examples = load_eval_set(cfg.eval_path)
    validate_eval_set(examples, corpus)

    judge = build_judge(cfg.judge_name, **cfg.judge_kwargs)

    runs: dict[str, ConfigRun] = {}
    for retrieval_config in cfg.configs:
        runs[retrieval_config.name] = run_config(
            cfg, retrieval_config, corpus, examples, judge
        )

    metric_names = all_metric_names(cfg.ks)
    comparisons: dict[tuple[str, str], list[MetricComparison]] = {}
    for name_a, name_b in cfg.comparisons:
        if name_a not in runs or name_b not in runs:
            raise KeyError(f"comparison references unknown config: {name_a}, {name_b}")
        pair_results = [
            compare_metric(
                metric,
                runs[name_a].per_query[metric],
                runs[name_b].per_query[metric],
                config_a=name_a,
                config_b=name_b,
                n_boot=cfg.n_boot,
                n_perm=cfg.n_perm,
                alpha=cfg.alpha,
                seed=cfg.seed,
            )
            for metric in metric_names
        ]
        comparisons[(name_a, name_b)] = pair_results

    return ExperimentResult(
        config=cfg,
        corpus_size=len(corpus),
        eval_size=len(examples),
        runs=runs,
        comparisons=comparisons,
        metric_names=metric_names,
    )


def _write_per_query_csv(result: ExperimentResult, path: Path) -> None:
    header = ["config", "example_id", *result.metric_names]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for name, run in result.runs.items():
            for row_index, example_id in enumerate(run.example_ids):
                row = [name, example_id]
                row.extend(
                    f"{run.per_query[metric][row_index]:.6f}"
                    for metric in result.metric_names
                )
                writer.writerow(row)


def _write_summary_csv(result: ExperimentResult, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config", "metric", "mean", "ci_low", "ci_high", "confidence"])
        for name, run in result.runs.items():
            for metric in result.metric_names:
                summary = run.summaries[metric]
                writer.writerow(
                    [
                        name,
                        metric,
                        f"{summary.point:.6f}",
                        f"{summary.low:.6f}",
                        f"{summary.high:.6f}",
                        f"{summary.confidence:.2f}",
                    ]
                )


def _write_comparisons_csv(result: ExperimentResult, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "config_a",
                "config_b",
                "metric",
                "mean_a",
                "mean_b",
                "diff",
                "diff_ci_low",
                "diff_ci_high",
                "p_value",
                "significant",
                "verdict",
            ]
        )
        for comparisons in result.comparisons.values():
            for comparison in comparisons:
                writer.writerow(
                    [
                        comparison.config_a,
                        comparison.config_b,
                        comparison.metric,
                        f"{comparison.mean_a:.6f}",
                        f"{comparison.mean_b:.6f}",
                        f"{comparison.diff:.6f}",
                        f"{comparison.diff_ci_low:.6f}",
                        f"{comparison.diff_ci_high:.6f}",
                        f"{comparison.p_value:.6f}",
                        str(comparison.significant).lower(),
                        comparison.verdict,
                    ]
                )


def write_results(result: ExperimentResult, output_dir: Path) -> dict[str, Path]:
    """Write CSVs and the Markdown report, returning the output file paths."""
    from rag_eval.report import build_markdown_report

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_query": output_dir / "per_query_metrics.csv",
        "summary": output_dir / "summary.csv",
        "comparisons": output_dir / "comparisons.csv",
        "report": output_dir / "report.md",
    }
    _write_per_query_csv(result, paths["per_query"])
    _write_summary_csv(result, paths["summary"])
    _write_comparisons_csv(result, paths["comparisons"])
    paths["report"].write_text(build_markdown_report(result), encoding="utf-8")
    return paths

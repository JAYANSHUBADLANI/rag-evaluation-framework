"""Render an :class:`~rag_eval.experiment.ExperimentResult` as Markdown."""

from __future__ import annotations

from rag_eval.experiment import ExperimentResult


def _percent(confidence: float) -> str:
    return f"{round(confidence * 100)}%"


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _overview(result: ExperimentResult) -> list[str]:
    cfg = result.config
    lines = ["# Retrieval-augmented generation evaluation report", ""]
    lines.append(
        f"- Corpus: **{result.corpus_size}** documents "
        f"| Evaluation set: **{result.eval_size}** questions"
    )
    lines.append(
        f"- Embedder: `{cfg.embedder_name}` "
        f"| Index: `{cfg.index_kind}` "
        f"| Judge: `{cfg.judge_name}`"
    )
    lines.append(
        f"- Bootstrap resamples: {cfg.n_boot:,} "
        f"| Permutations: {cfg.n_perm:,} "
        f"| Significance level: {cfg.alpha}"
    )
    lines.append("")
    return lines


def _configurations_table(result: ExperimentResult) -> list[str]:
    lines = ["## Configurations", ""]
    lines.append(
        "| Configuration | Chunk size | Overlap | Top-k | Embedder | Chunks indexed |"
    )
    lines.append("| --- | ---: | ---: | ---: | --- | ---: |")
    for name, run in result.runs.items():
        config = run.config
        lines.append(
            f"| `{name}` | {config.chunk_size} | {config.overlap} "
            f"| {config.top_k} | `{run.embedder_label}` | {run.n_chunks} |"
        )
    lines.append("")
    return lines


def _per_config_metrics(result: ExperimentResult) -> list[str]:
    confidence = _percent(1.0 - result.config.alpha)
    lines = [f"## Per-configuration metrics (mean with {confidence} bootstrap CI)", ""]
    for name, run in result.runs.items():
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append(f"| Metric | Mean | {confidence} CI |")
        lines.append("| --- | ---: | :---: |")
        for metric in result.metric_names:
            summary = run.summaries[metric]
            lines.append(
                f"| {metric} | {_fmt(summary.point)} "
                f"| [{_fmt(summary.low)}, {_fmt(summary.high)}] |"
            )
        lines.append("")
    return lines


def _comparisons(result: ExperimentResult) -> list[str]:
    confidence = _percent(1.0 - result.config.alpha)
    lines = ["## Pairwise comparisons", ""]
    lines.append(
        "Each pair is evaluated on the same questions. The difference is "
        "config A minus config B; the p-value comes from a two-sided paired "
        "permutation test."
    )
    lines.append("")
    for (name_a, name_b), comparisons in result.comparisons.items():
        lines.append(f"### `{name_a}` vs `{name_b}`")
        lines.append("")
        lines.append(
            f"| Metric | Mean A | Mean B | Δ (A−B) | {confidence} CI of Δ "
            f"| p-value | Verdict |"
        )
        lines.append("| --- | ---: | ---: | ---: | :---: | ---: | --- |")
        for comparison in comparisons:
            marker = " **\\***" if comparison.significant else ""
            lines.append(
                f"| {comparison.metric} | {_fmt(comparison.mean_a)} "
                f"| {_fmt(comparison.mean_b)} | {comparison.diff:+.4f} "
                f"| [{_fmt(comparison.diff_ci_low)}, {_fmt(comparison.diff_ci_high)}] "
                f"| {comparison.p_value:.4f}{marker} "
                f"| {comparison.verdict} |"
            )
        lines.append("")
    lines.append(
        f"**\\*** marks a statistically significant difference at "
        f"the {result.config.alpha} level."
    )
    lines.append("")
    return lines


def _headline(result: ExperimentResult) -> list[str]:
    primary = result.config.primary_metric
    if not result.comparisons or primary not in result.metric_names:
        return []
    lines = ["## Headline", ""]
    for (name_a, name_b), comparisons in result.comparisons.items():
        match = next((c for c in comparisons if c.metric == primary), None)
        if match is not None:
            lines.append(f"- **{primary}**, `{name_a}` vs `{name_b}`: {match.verdict}")
    lines.append("")
    return lines


def build_markdown_report(result: ExperimentResult) -> str:
    """Build the full Markdown evaluation report as a single string."""
    sections: list[str] = []
    sections.extend(_overview(result))
    sections.extend(_headline(result))
    sections.extend(_configurations_table(result))
    sections.extend(_per_config_metrics(result))
    sections.extend(_comparisons(result))
    return "\n".join(sections).rstrip() + "\n"

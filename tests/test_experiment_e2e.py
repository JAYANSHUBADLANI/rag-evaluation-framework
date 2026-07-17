"""End-to-end experiment tests on a tiny fixture, plus the CLI entry point."""

from __future__ import annotations

import json

import yaml

from rag_eval.__main__ import main
from rag_eval.experiment import all_metric_names, run_experiment, write_results


def test_run_experiment_produces_valid_result(tiny_config):
    result = run_experiment(tiny_config)

    assert set(result.runs) == {"small", "large"}
    assert result.eval_size == 6
    assert result.corpus_size == 5

    expected_metrics = all_metric_names(tiny_config.ks)
    for run in result.runs.values():
        assert len(run.example_ids) == 6
        for metric in expected_metrics:
            values = run.per_query[metric]
            assert len(values) == 6
            assert all(0.0 <= v <= 1.0 for v in values)
            summary = run.summaries[metric]
            assert summary.low <= summary.point <= summary.high

    # A comparison exists for every metric of the requested pair.
    comparisons = result.comparisons[("small", "large")]
    assert len(comparisons) == len(expected_metrics)


def test_retrieval_recovers_relevant_docs(tiny_config):
    # On this easy fixture, hit_rate@3 should be perfect for both configs.
    result = run_experiment(tiny_config)
    for run in result.runs.values():
        assert run.summaries["hit_rate@3"].point == 1.0


def test_write_results_creates_all_files(tiny_config):
    result = run_experiment(tiny_config)
    paths = write_results(result, tiny_config.output_dir)

    for path in paths.values():
        assert path.exists()

    report = paths["report"].read_text(encoding="utf-8")
    assert "## Configurations" in report
    assert "## Pairwise comparisons" in report

    header = paths["summary"].read_text(encoding="utf-8").splitlines()[0]
    assert header == "config,metric,mean,ci_low,ci_high,confidence"


def test_cli_run_writes_report(tiny_project, tmp_path):
    config = {
        "corpus_path": str(tiny_project["corpus"]),
        "eval_path": str(tiny_project["eval"]),
        "output_dir": str(tiny_project["output"]),
        "seed": 1,
        "ks": [1, 3],
        "n_boot": 200,
        "n_perm": 200,
        "embedder": {"name": "tfidf", "svd_dim": None},
        "index_kind": "numpy",
        "judge": {"name": "local"},
        "configs": [
            {"name": "a", "chunk_size": 8, "overlap": 2, "top_k": 3},
            {"name": "b", "chunk_size": 16, "overlap": 2, "top_k": 3},
        ],
        "comparisons": [["a", "b"]],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    exit_code = main(["run", "--config", str(config_path)])
    assert exit_code == 0
    assert (tiny_project["output"] / "report.md").exists()
    assert (tiny_project["output"] / "per_query_metrics.csv").exists()


def test_per_query_csv_row_count(tiny_config):
    result = run_experiment(tiny_config)
    paths = write_results(result, tiny_config.output_dir)
    rows = paths["per_query"].read_text(encoding="utf-8").splitlines()
    # header + (2 configs * 6 questions)
    assert len(rows) == 1 + 2 * 6

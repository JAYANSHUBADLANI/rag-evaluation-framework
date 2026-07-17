"""Command line entry point: ``python -m rag_eval run --config <file>``."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from collections.abc import Sequence

from rag_eval.config import load_config
from rag_eval.experiment import ExperimentResult, run_experiment, write_results


def _print_summary(result: ExperimentResult, paths: dict[str, Path]) -> None:
    primary = result.config.primary_metric
    print(
        f"Evaluated {len(result.runs)} configuration(s) on "
        f"{result.eval_size} questions over {result.corpus_size} documents."
    )
    if primary in result.metric_names:
        print(f"\n{primary} by configuration:")
        for name, run in result.runs.items():
            summary = run.summaries[primary]
            print(
                f"  {name:>16}: {summary.point:.4f} "
                f"[{summary.low:.4f}, {summary.high:.4f}]"
            )
    for (name_a, name_b), comparisons in result.comparisons.items():
        match = next((c for c in comparisons if c.metric == primary), None)
        if match is not None:
            print(f"\n{name_a} vs {name_b} on {primary}: {match.verdict}")
    print("\nWritten:")
    for label, path in paths.items():
        print(f"  {label:>11}: {path}")


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.output is not None:
        config = replace(config, output_dir=Path(args.output))
    result = run_experiment(config)
    paths = write_results(result, config.output_dir)
    _print_summary(result, paths)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="rag_eval",
        description="Evaluate and statistically compare RAG retrieval configurations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the evaluation experiment")
    run_parser.add_argument(
        "--config",
        required=True,
        help="path to a YAML experiment configuration",
    )
    run_parser.add_argument(
        "--output",
        default=None,
        help="override the output directory from the config",
    )
    run_parser.set_defaults(handler=_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch to the selected sub-command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

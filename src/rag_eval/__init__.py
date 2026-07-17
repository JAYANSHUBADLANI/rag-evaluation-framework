"""Statistically grounded evaluation for retrieval-augmented generation pipelines.

The package is organised into three layers:

* :mod:`rag_eval.metrics` -- retrieval and generation metric definitions.
* :mod:`rag_eval.judges` -- backends that score generation quality (offline
  embedding heuristics or an OpenAI-compatible language-model judge).
* :mod:`rag_eval.stats` -- bootstrap confidence intervals and paired
  permutation testing used to compare pipeline configurations.

The remaining modules wire these together into a runnable demo pipeline
(:mod:`rag_eval.pipeline`), an experiment runner (:mod:`rag_eval.experiment`)
and a command line entry point (``python -m rag_eval``).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]

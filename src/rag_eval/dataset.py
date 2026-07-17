"""Loading and validating the demo corpus and the labelled evaluation set.

The corpus is a directory of ``.txt`` files, one document per file; the file
stem is the document id. The evaluation set is a JSON list of objects with an
``id``, a ``question`` and the ``relevant_doc_ids`` that answer it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """A source document identified by its file stem."""

    doc_id: str
    text: str


@dataclass(frozen=True)
class EvalExample:
    """A labelled evaluation question with its relevant document ids."""

    example_id: str
    question: str
    relevant_doc_ids: tuple[str, ...]


def load_corpus(path: str | Path) -> list[Document]:
    """Load every ``.txt`` file in ``path`` as a :class:`Document`.

    Documents are returned sorted by id for deterministic ordering.
    """
    directory = Path(path)
    if not directory.is_dir():
        raise NotADirectoryError(f"corpus path is not a directory: {directory}")

    documents: list[Document] = []
    for file_path in sorted(directory.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()
        if text:
            documents.append(Document(doc_id=file_path.stem, text=text))
    if not documents:
        raise ValueError(f"no non-empty .txt documents found in {directory}")
    return documents


def load_eval_set(path: str | Path) -> list[EvalExample]:
    """Load the evaluation set from a JSON file into :class:`EvalExample` objects."""
    file_path = Path(path)
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"eval set {file_path} must be a JSON list")

    examples: list[EvalExample] = []
    for entry in raw:
        examples.append(
            EvalExample(
                example_id=str(entry["id"]),
                question=str(entry["question"]),
                relevant_doc_ids=tuple(str(d) for d in entry["relevant_doc_ids"]),
            )
        )
    if not examples:
        raise ValueError(f"eval set {file_path} is empty")
    return examples


def validate_eval_set(
    examples: list[EvalExample], corpus: list[Document]
) -> None:
    """Check the evaluation set is well formed against ``corpus``.

    Raises :class:`ValueError` if any example has a duplicate id, no relevant
    documents, or references a document id that is not present in the corpus.
    """
    known_ids = {document.doc_id for document in corpus}
    seen_ids: set[str] = set()
    for example in examples:
        if example.example_id in seen_ids:
            raise ValueError(f"duplicate example id: {example.example_id}")
        seen_ids.add(example.example_id)

        if not example.relevant_doc_ids:
            raise ValueError(f"example {example.example_id} has no relevant documents")

        missing = set(example.relevant_doc_ids) - known_ids
        if missing:
            raise ValueError(
                f"example {example.example_id} references unknown documents: "
                f"{sorted(missing)}"
            )

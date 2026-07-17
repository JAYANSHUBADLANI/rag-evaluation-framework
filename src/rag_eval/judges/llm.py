"""Language-model-as-judge backend for generation quality.

This judge asks an instruction-following model to rate an answer on the three
generation metrics and to reply with strict JSON. It targets any
OpenAI-compatible ``/chat/completions`` endpoint, so the base URL and model are
configurable and it works against hosted or self-hosted servers.

The network call is isolated behind :meth:`LLMJudge._chat`. Tests (and any
caller wanting determinism) inject a ``chat_fn`` to bypass HTTP entirely, which
is why no API key is required to exercise the class.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable, Sequence

from rag_eval.judges.base import GenerationScores, Judge

Message = dict[str, str]
ChatFn = Callable[[Sequence[Message]], str]

_SYSTEM_PROMPT = (
    "You are a strict evaluator of question-answering systems. You grade an "
    "answer only against the question and the provided context passages. You "
    "never use outside knowledge and you always reply with a single JSON object."
)

_METRIC_KEYS = ("faithfulness", "answer_relevance", "context_utilization")


def build_judge_prompt(question: str, answer: str, contexts: Sequence[str]) -> str:
    """Render the user prompt asking for the three generation scores.

    The model is asked to return a JSON object with float fields in ``[0, 1]``
    for faithfulness, answer relevance and context utilization.
    """
    joined = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    if not joined:
        joined = "(no context was retrieved)"
    return (
        "Rate the answer on three criteria, each a number from 0.0 to 1.0:\n"
        "- faithfulness: every claim in the answer is supported by the context.\n"
        "- answer_relevance: the answer directly addresses the question.\n"
        "- context_utilization: the answer makes use of the retrieved context.\n\n"
        f"Question:\n{question}\n\n"
        f"Context passages:\n{joined}\n\n"
        f"Answer:\n{answer}\n\n"
        'Reply with only JSON, for example: '
        '{"faithfulness": 0.9, "answer_relevance": 0.8, "context_utilization": 0.7}'
    )


def parse_scores(content: str) -> GenerationScores:
    """Parse a model reply into :class:`GenerationScores`.

    Tolerates surrounding prose or Markdown code fences by extracting the first
    ``{...}`` block. Missing fields default to ``0.0`` and all values are
    clamped to ``[0, 1]``.
    """
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object found in judge reply: {content!r}")
    payload = json.loads(content[start : end + 1])

    def _clamp(key: str) -> float:
        try:
            value = float(payload.get(key, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        return max(0.0, min(1.0, value))

    return GenerationScores(*(_clamp(key) for key in _METRIC_KEYS))


class LLMJudge(Judge):
    """Judge that delegates scoring to an OpenAI-compatible chat model.

    Parameters
    ----------
    model:
        Model identifier passed to the endpoint.
    base_url:
        Root of the OpenAI-compatible API (``/chat/completions`` is appended).
    api_key:
        Explicit key; if omitted it is read from ``api_key_env``.
    api_key_env:
        Environment variable holding the key (default ``OPENAI_API_KEY``).
    temperature:
        Sampling temperature; ``0.0`` keeps grading as deterministic as the
        server allows.
    timeout:
        Per-request timeout in seconds.
    chat_fn:
        Optional callable ``(messages) -> content`` used instead of HTTP. When
        supplied the judge performs no network access, which makes it fully
        mockable in tests.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.0,
        timeout: float = 30.0,
        chat_fn: ChatFn | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.timeout = timeout
        self._api_key = api_key
        self._chat_fn = chat_fn

    def _resolve_api_key(self) -> str:
        key = self._api_key or os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"no API key: pass api_key or set ${self.api_key_env}"
            )
        return key

    def _chat(self, messages: Sequence[Message]) -> str:
        """Return the reply message content for ``messages``.

        Uses the injected ``chat_fn`` when present, otherwise POSTs to the
        configured endpoint with the standard library HTTP client.
        """
        if self._chat_fn is not None:
            return self._chat_fn(messages)

        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._resolve_api_key()}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]

    def score(
        self, question: str, answer: str, contexts: Sequence[str]
    ) -> GenerationScores:
        """Score one triple by prompting the model and parsing its JSON reply."""
        messages: list[Message] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": build_judge_prompt(question, answer, contexts)},
        ]
        content = self._chat(messages)
        return parse_scores(content)

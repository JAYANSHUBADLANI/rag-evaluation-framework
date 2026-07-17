"""Tests for the offline local judge and the mockable language-model judge."""

from __future__ import annotations

import pytest

from rag_eval.judges import (
    GenerationScores,
    LLMJudge,
    LocalJudge,
    build_judge,
    build_judge_prompt,
    parse_scores,
)


# --- Local judge ---------------------------------------------------------

def test_local_judge_extractive_answer_is_faithful():
    judge = LocalJudge()
    scores = judge.score(
        question="What colour is the sky?",
        answer="The sky is blue.",
        contexts=["The sky is blue. Grass is green."],
    )
    # The answer sentence appears verbatim in the context.
    assert scores.faithfulness == pytest.approx(1.0)
    # One of the two context sentences is echoed by the answer.
    assert scores.context_utilization == pytest.approx(0.5)
    # The answer shares content with the question.
    assert scores.answer_relevance > 0.0


def test_local_judge_empty_answer_scores_zero():
    judge = LocalJudge()
    scores = judge.score("Any question?", "", ["Some context sentence here."])
    assert scores == GenerationScores(0.0, 0.0, 0.0)


def test_local_judge_without_context_is_ungrounded():
    judge = LocalJudge()
    scores = judge.score("What colour is the sky?", "The sky is blue.", [])
    assert scores.faithfulness == 0.0
    assert scores.context_utilization == 0.0


def test_local_judge_unsupported_answer_has_low_faithfulness():
    judge = LocalJudge()
    scores = judge.score(
        question="What colour is the sky?",
        answer="Bananas are a good source of potassium.",
        contexts=["The sky is blue. Grass is green."],
    )
    assert scores.faithfulness == 0.0


def test_local_judge_batch():
    judge = LocalJudge()
    results = judge.score_batch(
        [
            ("Q1?", "The sky is blue.", ["The sky is blue."]),
            ("Q2?", "", ["irrelevant"]),
        ]
    )
    assert len(results) == 2
    assert results[0].faithfulness == pytest.approx(1.0)


# --- LLM judge parsing ---------------------------------------------------

def test_parse_scores_plain_json():
    scores = parse_scores(
        '{"faithfulness": 0.9, "answer_relevance": 0.8, "context_utilization": 0.7}'
    )
    assert scores == GenerationScores(0.9, 0.8, 0.7)


def test_parse_scores_with_code_fence_and_prose():
    reply = (
        "Here is my grading:\n```json\n"
        '{"faithfulness": 1.0, "answer_relevance": 0.5, "context_utilization": 0.25}\n'
        "```\nThanks."
    )
    scores = parse_scores(reply)
    assert scores == GenerationScores(1.0, 0.5, 0.25)


def test_parse_scores_clamps_and_defaults_missing():
    scores = parse_scores('{"faithfulness": 1.7, "answer_relevance": -0.4}')
    assert scores.faithfulness == 1.0
    assert scores.answer_relevance == 0.0
    assert scores.context_utilization == 0.0  # missing -> default


def test_parse_scores_without_json_raises():
    with pytest.raises(ValueError):
        parse_scores("no json here")


# --- LLM judge with an injected chat function (no network) ---------------

def test_llm_judge_uses_injected_chat_fn():
    captured: dict[str, object] = {}

    def fake_chat(messages):
        captured["messages"] = messages
        return '{"faithfulness": 0.6, "answer_relevance": 0.9, "context_utilization": 0.4}'

    judge = LLMJudge(model="test-model", chat_fn=fake_chat)
    scores = judge.score("Why is the sky blue?", "Rayleigh scattering.", ["Context."])

    assert scores == GenerationScores(0.6, 0.9, 0.4)
    # The system + user messages were passed through.
    messages = captured["messages"]
    assert messages[0]["role"] == "system"
    assert "Why is the sky blue?" in messages[1]["content"]


def test_build_judge_prompt_contains_inputs():
    prompt = build_judge_prompt("The question", "The answer", ["Passage one", "Passage two"])
    assert "The question" in prompt
    assert "The answer" in prompt
    assert "Passage one" in prompt and "Passage two" in prompt


def test_build_judge_factory():
    assert isinstance(build_judge("local"), LocalJudge)
    assert isinstance(build_judge("llm", model="m", chat_fn=lambda m: "{}"), LLMJudge)
    with pytest.raises(ValueError):
        build_judge("nonsense")

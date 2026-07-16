"""Tests para credixai.copilot.evaluator.llm_judge (paso 6, prd.md 9.1).

TDD: llm_judge es la segunda capa del evaluator (tras el precheck
deterministico), un juez LLM que verifica que el memo no afirme nada no
soportado por el contexto. Se testea con chat_fn inyectado (stub), mismo
manejo de fences markdown que ya resolvimos en el reranker.
"""

import pytest

from credixai.copilot.evaluator import llm_judge


def test_llm_judge_returns_approved_verdict_and_feedback():
    def fake_chat_fn(messages):
        return '{"approved": true, "feedback": "El memo es consistente con el contexto."}'

    approved, feedback = llm_judge(
        memo="Memo consistente.",
        context="score=alto_riesgo, reason_codes=[EXT_SOURCE_1 bajo]",
        chat_fn=fake_chat_fn,
    )

    assert approved is True
    assert feedback == "El memo es consistente con el contexto."


def test_llm_judge_returns_rejected_verdict_and_feedback():
    def fake_chat_fn(messages):
        return '{"approved": false, "feedback": "El memo afirma algo no soportado por el contexto."}'

    approved, feedback = llm_judge(memo="Memo dudoso.", context="score=alto_riesgo", chat_fn=fake_chat_fn)

    assert approved is False
    assert feedback == "El memo afirma algo no soportado por el contexto."


def test_llm_judge_strips_markdown_fences_before_parsing():
    def fake_chat_fn(messages):
        return '```json\n{"approved": true, "feedback": "ok"}\n```'

    approved, feedback = llm_judge(memo="Memo.", context="score=riesgo_aceptable", chat_fn=fake_chat_fn)

    assert approved is True
    assert feedback == "ok"


def test_llm_judge_raises_on_malformed_response():
    def fake_chat_fn(messages):
        return "esto no es JSON"

    with pytest.raises(ValueError):
        llm_judge(memo="Memo.", context="score=alto_riesgo", chat_fn=fake_chat_fn)

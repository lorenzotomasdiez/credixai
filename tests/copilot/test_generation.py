"""Tests para credixai.copilot.generation.draft_memo (paso 6).

TDD: draft_memo redacta el memo en base a los reason_codes y citas ya
calculados por las tools, sin inventar razones nuevas; se testea con un
chat_fn inyectado (stub), mismo patron que PolicyAnswerer/LLMReranker.
"""

from credixai.copilot.generation import draft_memo


def test_draft_memo_sends_score_reason_codes_and_citations_in_the_prompt():
    captured_messages = []

    def fake_chat_fn(messages):
        captured_messages.extend(messages)
        return "Memo redactado."

    memo = draft_memo(
        score={"sk_id_curr": 100002, "probability": 0.72, "threshold": 0.5, "decision": "alto_riesgo"},
        reason_codes=["EXT_SOURCE_1 bajo", "AMT_CREDIT alto"],
        policy_citations=["[Politica interna] umbral de decision"],
        chat_fn=fake_chat_fn,
    )

    assert memo == "Memo redactado."
    user_content = captured_messages[-1]["content"]
    assert "alto_riesgo" in user_content
    assert "EXT_SOURCE_1 bajo" in user_content
    assert "umbral de decision" in user_content


def test_draft_memo_includes_revision_feedback_when_given():
    captured_messages = []

    def fake_chat_fn(messages):
        captured_messages.extend(messages)
        return "Memo corregido."

    draft_memo(
        score={"sk_id_curr": 100002, "probability": 0.72, "threshold": 0.5, "decision": "alto_riesgo"},
        reason_codes=["EXT_SOURCE_1 bajo"],
        policy_citations=[],
        chat_fn=fake_chat_fn,
        revision_feedback="Mencionaste genero, sacalo.",
    )

    user_content = captured_messages[-1]["content"]
    assert "Mencionaste genero, sacalo." in user_content


def test_draft_memo_without_revision_feedback_does_not_mention_it():
    def fake_chat_fn(messages):
        return "Memo."

    memo = draft_memo(
        score={"sk_id_curr": 100002, "probability": 0.2, "threshold": 0.5, "decision": "riesgo_aceptable"},
        reason_codes=[],
        policy_citations=[],
        chat_fn=fake_chat_fn,
    )

    assert memo == "Memo."

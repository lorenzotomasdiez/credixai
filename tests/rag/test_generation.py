"""Tests para credixai.rag.generation (paso 5).

TDD: se escribe antes que credixai/rag/generation.py. PolicyAnswerer recibe
chat_fn inyectado, igual que LLMReranker: el test stubea la respuesta del
LLM y no llama a OpenRouter.
"""

from credixai.rag.generation import PolicyAnswerer
from credixai.rag.retrieval import RetrievedChunk


def _retrieved():
    return [
        RetrievedChunk(doc_id="d1", chunk_id="d1::0", title="Adverse action", text="maximo 4 reason codes"),
        RetrievedChunk(doc_id="d2", chunk_id="d2::0", title="BCRA", text="clasificacion de deudores"),
    ]


def test_answer_returns_llm_text_and_citations_from_retrieved_chunks():
    def fake_chat_fn(messages):
        return "Se comunican como maximo 4 reason codes [Adverse action]."

    answerer = PolicyAnswerer(chat_fn=fake_chat_fn)

    result = answerer.answer("cuantos reason codes se comunican", _retrieved())

    assert "4 reason codes" in result.answer
    assert {c.doc_title for c in result.citations} == {"Adverse action", "BCRA"}


def test_answer_citations_include_chunk_id_and_snippet():
    def fake_chat_fn(messages):
        return "respuesta"

    answerer = PolicyAnswerer(chat_fn=fake_chat_fn)

    result = answerer.answer("pregunta", _retrieved())

    citation = next(c for c in result.citations if c.chunk_id == "d1::0")
    assert citation.doc_title == "Adverse action"
    assert "reason codes" in citation.snippet


def test_answer_with_no_retrieved_chunks_still_returns_result_without_citations():
    def fake_chat_fn(messages):
        return "No encontre informacion en el corpus para responder esa pregunta."

    answerer = PolicyAnswerer(chat_fn=fake_chat_fn)

    result = answerer.answer("pregunta sin contexto", [])

    assert result.citations == []
    assert "No encontre" in result.answer


def test_answer_prompt_passed_to_chat_fn_includes_retrieved_context():
    captured = {}

    def fake_chat_fn(messages):
        captured["messages"] = messages
        return "respuesta"

    answerer = PolicyAnswerer(chat_fn=fake_chat_fn)
    answerer.answer("pregunta", _retrieved())

    joined = " ".join(m["content"] for m in captured["messages"])
    assert "maximo 4 reason codes" in joined
    assert "clasificacion de deudores" in joined

"""Tests para credixai.dashboard_client (paso 9, prd.md 9.1: UI de RAG/copiloto).

TDD: se escribe antes que credixai/dashboard_client.py. El dashboard llama
a la API real por HTTP (mismo principio ya usado en credixai/copilot/tools.py
para las tools del copiloto: consumir por HTTP en vez de importar directo),
asi que se testea con httpx.MockTransport -- respuestas simuladas, sin red
real ni Streamlit.
"""

import httpx
import pytest

from credixai.dashboard_client import query_policy, request_copilot_memo


def _client_with(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")


def test_query_policy_returns_answer_and_citations():
    def handler(request):
        assert request.url.path == "/rag/query"
        assert request.method == "POST"
        return httpx.Response(
            200,
            json={
                "answer": "hasta 4 reason codes",
                "citations": [{"doc_title": "Politica interna", "chunk_id": "doc::0", "snippet": "maximo 4"}],
            },
        )

    client = _client_with(handler)

    result = query_policy(client, "¿cuantos reason codes como maximo?")

    assert result["answer"] == "hasta 4 reason codes"
    assert result["citations"][0]["doc_title"] == "Politica interna"


def test_query_policy_raises_on_error_status():
    def handler(request):
        return httpx.Response(500, json={"detail": "error interno"})

    client = _client_with(handler)

    with pytest.raises(httpx.HTTPStatusError):
        query_policy(client, "pregunta")


def test_request_copilot_memo_returns_memo_and_status():
    def handler(request):
        assert request.url.path == "/copilot/memo/100002"
        assert request.method == "POST"
        return httpx.Response(
            200,
            json={
                "sk_id_curr": 100002,
                "decision": "alto_riesgo",
                "memo": "Memo redactado.",
                "citations": [],
                "status": "approved",
                "evaluator_feedback": "memo consistente",
            },
        )

    client = _client_with(handler)

    result = request_copilot_memo(client, 100002)

    assert result["memo"] == "Memo redactado."
    assert result["status"] == "approved"


def test_request_copilot_memo_raises_on_404():
    def handler(request):
        return httpx.Response(404, json={"detail": "SK_ID_CURR 999999999 no encontrado"})

    client = _client_with(handler)

    with pytest.raises(httpx.HTTPStatusError):
        request_copilot_memo(client, 999999999)

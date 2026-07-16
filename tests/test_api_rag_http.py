"""Tests HTTP para /rag/query en app/api.py (paso 5, prd.md 9.1).

TDD: se escribe antes de tocar app/api.py. Igual que test_api_http.py, usa
TestClient con dependencias inyectadas via dependency override, sin llamar a
OpenRouter ni a Qdrant reales.
"""

import pytest
from fastapi.testclient import TestClient

from credixai.rag.generation import AnswerResult, Citation


class _FakeRagPipeline:
    def query(self, question: str) -> AnswerResult:
        return AnswerResult(
            answer=f"respuesta a: {question}",
            citations=[Citation(doc_title="Politica de ejemplo", chunk_id="doc::0", snippet="fragmento citado")],
        )


@pytest.fixture
def client():
    from app.api import app, get_rag_pipeline

    app.dependency_overrides[get_rag_pipeline] = lambda: _FakeRagPipeline()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_rag_query_returns_answer_and_citations(client):
    response = client.post("/rag/query", json={"question": "¿cuantos reason codes se comunican?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "respuesta a: ¿cuantos reason codes se comunican?"
    assert body["citations"] == [
        {"doc_title": "Politica de ejemplo", "chunk_id": "doc::0", "snippet": "fragmento citado"}
    ]


def test_rag_query_rejects_empty_question(client):
    response = client.post("/rag/query", json={"question": ""})

    assert response.status_code == 422


def test_rag_query_missing_body_field_returns_422(client):
    response = client.post("/rag/query", json={})

    assert response.status_code == 422


def test_openapi_exposes_rag_query_path(client):
    response = client.get("/openapi.json")

    assert "/rag/query" in response.json()["paths"]

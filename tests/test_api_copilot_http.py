"""Tests HTTP para /copilot/memo/{sk_id_curr} en app/api.py (paso 6).

TDD: se escribe antes de tocar app/api.py. Mismo patron que
test_api_rag_http.py: TestClient con el grafo LangGraph inyectado via
dependency override (fake con .ainvoke), sin llamar a OpenRouter real.
"""

import httpx
import pytest
from fastapi.testclient import TestClient


class _FakeGraph:
    def __init__(self, result=None, raises=None):
        self._result = result
        self._raises = raises

    async def ainvoke(self, state):
        if self._raises:
            raise self._raises
        return self._result


def _approved_result(sk_id_curr):
    return {
        "sk_id_curr": sk_id_curr,
        "score": {"sk_id_curr": sk_id_curr, "probability": 0.8, "threshold": 0.5, "decision": "alto_riesgo"},
        "explanation": {"reason_codes": ["EXT_SOURCE_1 bajo"]},
        "policy_result": {
            "answer": "...",
            "citations": [{"doc_title": "Politica interna", "chunk_id": "doc::0", "snippet": "maximo 4 reason codes"}],
        },
        "memo": "Memo redactado.",
        "evaluator_feedback": "memo consistente",
        "iteration": 0,
        "status": "approved",
    }


@pytest.fixture
def client_factory():
    from app.api import app, get_copilot_graph

    def _make(graph):
        app.dependency_overrides[get_copilot_graph] = lambda: graph
        test_client = TestClient(app)
        return test_client

    yield _make
    app.dependency_overrides.clear()


def test_copilot_memo_returns_approved_memo_with_citations(client_factory):
    client = client_factory(_FakeGraph(result=_approved_result(100002)))

    response = client.post("/copilot/memo/100002")

    assert response.status_code == 200
    body = response.json()
    assert body["sk_id_curr"] == 100002
    assert body["decision"] == "alto_riesgo"
    assert body["memo"] == "Memo redactado."
    assert body["status"] == "approved"
    assert body["citations"] == [{"doc_title": "Politica interna", "chunk_id": "doc::0", "snippet": "maximo 4 reason codes"}]


def test_copilot_memo_returns_needs_human_review_status(client_factory):
    result = _approved_result(100003)
    result["status"] = "needs_human_review"
    result["evaluator_feedback"] = "sigue mencionando atributos protegidos"
    client = client_factory(_FakeGraph(result=result))

    response = client.post("/copilot/memo/100003")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "needs_human_review"
    assert body["evaluator_feedback"] == "sigue mencionando atributos protegidos"


def test_copilot_memo_returns_404_when_application_not_found(client_factory):
    not_found = httpx.HTTPStatusError(
        "404", request=httpx.Request("GET", "http://x/score/999"), response=httpx.Response(404, request=httpx.Request("GET", "http://x/score/999"))
    )
    client = client_factory(_FakeGraph(raises=not_found))

    response = client.post("/copilot/memo/999999999")

    assert response.status_code == 404


def test_openapi_exposes_copilot_memo_path(client_factory):
    client = client_factory(_FakeGraph(result=_approved_result(100002)))

    response = client.get("/openapi.json")

    assert "/copilot/memo/{sk_id_curr}" in response.json()["paths"]

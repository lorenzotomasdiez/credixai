"""Tests para los root spans Langfuse en /rag/query y /copilot/memo (paso 7).

TDD: se escribe antes de tocar app/api.py. Los endpoints envuelven su
trabajo en un span raiz de Langfuse (get_langfuse_client, inyectable via
dependency override igual que get_service/get_rag_pipeline/get_copilot_graph)
para que las generations de OpenRouterClient, invocadas mas abajo en la
misma llamada, se aniden automaticamente via contexto OTEL. Se testea con
un fake Langfuse client, sin red.
"""

import httpx
import pytest
from fastapi.testclient import TestClient

from credixai.rag.generation import AnswerResult, Citation


class _FakeObservation:
    def __init__(self):
        self.updates = []
        self.scores = []

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def score(self, **kwargs):
        self.scores.append(kwargs)


class _FakeObservationContext:
    def __init__(self, observation):
        self._observation = observation

    def __enter__(self):
        return self._observation

    def __exit__(self, *exc_info):
        return False


class _FakeLangfuseClient:
    def __init__(self):
        self.calls = []
        self.observation = _FakeObservation()

    def start_as_current_observation(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeObservationContext(self.observation)


class _FakeRagPipeline:
    def query(self, question: str) -> AnswerResult:
        return AnswerResult(
            answer=f"respuesta a: {question}",
            citations=[Citation(doc_title="Politica de ejemplo", chunk_id="doc::0", snippet="fragmento")],
        )


class _FakeGraph:
    def __init__(self, result=None, raises=None):
        self._result = result
        self._raises = raises

    async def ainvoke(self, state):
        if self._raises:
            raise self._raises
        return self._result


def _approved_result(sk_id_curr, status="approved", iteration=0):
    return {
        "sk_id_curr": sk_id_curr,
        "score": {"sk_id_curr": sk_id_curr, "probability": 0.8, "threshold": 0.5, "decision": "alto_riesgo"},
        "explanation": {"reason_codes": ["EXT_SOURCE_1 bajo"]},
        "policy_result": {"answer": "...", "citations": []},
        "memo": "Memo redactado.",
        "evaluator_feedback": "memo consistente",
        "iteration": iteration,
        "status": status,
    }


@pytest.fixture
def fake_langfuse():
    return _FakeLangfuseClient()


@pytest.fixture
def client_factory(fake_langfuse):
    from app.api import app, get_copilot_graph, get_langfuse_client, get_rag_pipeline

    def _make(graph=None, pipeline=None):
        app.dependency_overrides[get_langfuse_client] = lambda: fake_langfuse
        if graph is not None:
            app.dependency_overrides[get_copilot_graph] = lambda: graph
        if pipeline is not None:
            app.dependency_overrides[get_rag_pipeline] = lambda: pipeline
        return TestClient(app)

    yield _make
    app.dependency_overrides.clear()


def test_rag_query_opens_a_langfuse_root_span(client_factory, fake_langfuse):
    client = client_factory(pipeline=_FakeRagPipeline())

    response = client.post("/rag/query", json={"question": "¿que dice la politica de adverse action?"})

    assert response.status_code == 200
    assert len(fake_langfuse.calls) == 1
    call = fake_langfuse.calls[0]
    assert call["name"] == "rag_query"
    assert call["as_type"] == "span"
    assert call["input"] == {"question": "¿que dice la politica de adverse action?"}
    assert fake_langfuse.observation.updates[0]["output"] == "respuesta a: ¿que dice la politica de adverse action?"


def test_copilot_memo_opens_a_langfuse_root_span_and_scores_first_try_success(client_factory, fake_langfuse):
    client = client_factory(graph=_FakeGraph(result=_approved_result(100002, status="approved", iteration=0)))

    response = client.post("/copilot/memo/100002")

    assert response.status_code == 200
    call = fake_langfuse.calls[0]
    assert call["name"] == "copilot_memo"
    assert call["as_type"] == "span"
    assert call["input"] == {"sk_id_curr": 100002}
    score = fake_langfuse.observation.scores[0]
    assert score["name"] == "evaluator_passed_first_try"
    assert score["value"] == 1.0


def test_copilot_memo_scores_zero_when_needs_human_review(client_factory, fake_langfuse):
    client = client_factory(graph=_FakeGraph(result=_approved_result(100003, status="needs_human_review", iteration=1)))

    response = client.post("/copilot/memo/100003")

    assert response.status_code == 200
    score = fake_langfuse.observation.scores[0]
    assert score["value"] == 0.0


def test_copilot_memo_404_still_propagates_through_the_span(client_factory, fake_langfuse):
    not_found = httpx.HTTPStatusError(
        "404",
        request=httpx.Request("GET", "http://x/score/999"),
        response=httpx.Response(404, request=httpx.Request("GET", "http://x/score/999")),
    )
    client = client_factory(graph=_FakeGraph(raises=not_found))

    response = client.post("/copilot/memo/999999999")

    assert response.status_code == 404
    assert len(fake_langfuse.calls) == 1

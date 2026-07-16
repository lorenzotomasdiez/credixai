"""Tests para credixai.copilot.tools (paso 6, prd.md 9.1).

TDD: las tools del copiloto llaman a los endpoints REST ya existentes
(/score, /explain, /rag/query) via httpx contra una app ASGI, nunca
importando ScoringService/RagPipeline directamente (prd.md linea 254).
Se testea contra una app FastAPI de juguete, sin cargar el modelo real.
Son async (httpx.ASGITransport solo soporta clientes async); se manejan
con asyncio.run en vez de sumar pytest-asyncio como dependencia nueva.
"""

import asyncio

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from credixai.copilot.tools import explain_shap, retrieve_policy, score_application


@pytest.fixture
def http_client():
    stub_app = FastAPI()

    @stub_app.get("/score/{sk_id_curr}")
    def score(sk_id_curr: int):
        if sk_id_curr == 999999999:
            raise HTTPException(status_code=404, detail="no encontrado")
        return {"sk_id_curr": sk_id_curr, "probability": 0.72, "threshold": 0.5, "decision": "alto_riesgo"}

    @stub_app.get("/explain/{sk_id_curr}")
    def explain(sk_id_curr: int, top_n: int = 4):
        return {
            "sk_id_curr": sk_id_curr,
            "probability": 0.72,
            "threshold": 0.5,
            "decision": "alto_riesgo",
            "base_value": 0.1,
            "shap_values": [0.3, 0.2],
            "feature_names": ["EXT_SOURCE_1", "AMT_CREDIT"],
            "reason_codes": ["EXT_SOURCE_1 bajo", "AMT_CREDIT alto"],
        }

    @stub_app.post("/rag/query")
    def rag_query(payload: dict):
        return {
            "answer": f"respuesta a: {payload['question']}",
            "citations": [{"doc_title": "Politica", "chunk_id": "doc::0", "snippet": "fragmento"}],
        }

    transport = httpx.ASGITransport(app=stub_app)
    client = httpx.AsyncClient(transport=transport, base_url="http://copilot-internal")
    yield client
    asyncio.run(client.aclose())


def test_score_application_calls_score_endpoint(http_client):
    result = asyncio.run(score_application(http_client, sk_id_curr=100002))

    assert result["sk_id_curr"] == 100002
    assert result["decision"] == "alto_riesgo"


def test_explain_shap_calls_explain_endpoint_with_top_n(http_client):
    result = asyncio.run(explain_shap(http_client, sk_id_curr=100002, top_n=4))

    assert result["reason_codes"] == ["EXT_SOURCE_1 bajo", "AMT_CREDIT alto"]


def test_retrieve_policy_calls_rag_query_endpoint(http_client):
    result = asyncio.run(retrieve_policy(http_client, query="cuantos reason codes maximo"))

    assert result["answer"] == "respuesta a: cuantos reason codes maximo"
    assert result["citations"][0]["doc_title"] == "Politica"


def test_score_application_raises_on_not_found(http_client):
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(score_application(http_client, sk_id_curr=999999999))

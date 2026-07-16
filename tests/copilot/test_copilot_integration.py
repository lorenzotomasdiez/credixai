"""Verificacion real end-to-end del copiloto (paso 6).

Marcado integration: no corre en CI, requiere OPENROUTER_API_KEY real,
Qdrant corriendo con el corpus ya ingestado (scripts/06_rag_ingest.py) y
carga el modelo real (~30s la primera vez). Se corre manualmente antes de
dar por cerrado el paso 6, mismo criterio que
tests/rag/test_openrouter_client_integration.py.

Usa un sk_id_curr de riesgo_aceptable para mantener el caso rapido (el
orchestrator no llama explain_shap ni retrieve_policy en ese camino); el
camino alto_riesgo completo (con tool-calling real de las 3 tools y el loop
evaluator-optimizer) se verifico manualmente via curl, ver
docs/informe-final.md seccion 8.6.
"""

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    from app.api import app

    with TestClient(app) as test_client:
        yield test_client


def test_copilot_memo_end_to_end_riesgo_aceptable(client):
    response = client.post("/copilot/memo/100003")

    assert response.status_code == 200
    body = response.json()
    assert body["sk_id_curr"] == 100003
    assert body["decision"] == "riesgo_aceptable"
    assert body["status"] in {"approved", "needs_human_review"}
    assert len(body["memo"]) > 0

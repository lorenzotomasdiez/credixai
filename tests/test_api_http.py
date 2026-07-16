"""Tests HTTP para app/api.py (paso 2).

TDD: este archivo se escribe antes que app/api.py. Usa TestClient con un
ScoringService sintetico inyectado via dependency override, para no
reentrenar el modelo real ni tocar data/processed en la suite de tests.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from credixai.api import ScoringService
from credixai.modeling import build_xgboost


def _synthetic_service():
    rng = np.random.RandomState(7)
    n = 300
    signal = rng.normal(0, 1, n)
    X = pd.DataFrame({
        "EXT_SOURCE_2": signal + rng.normal(0, 0.2, n),
        "credit_to_income": rng.uniform(0.5, 5, n),
    })
    y = pd.Series((signal > np.median(signal)).astype(int), name="TARGET")
    model = build_xgboost(X, y)
    proba_full = model.predict_proba(X)[:, 1]
    threshold = float(np.quantile(proba_full, 1 - y.mean()))
    bundle = {
        "model": model,
        "X": X,
        "y": y,
        "train_full": pd.DataFrame({"SK_ID_CURR": np.arange(1, n + 1)}),
        "proba_full": proba_full,
        "threshold": threshold,
    }
    return ScoringService(bundle)


@pytest.fixture
def client(tmp_path):
    from app.api import app, get_prediction_log_path, get_service

    service = _synthetic_service()
    app.dependency_overrides[get_service] = lambda: service
    app.dependency_overrides[get_prediction_log_path] = lambda: str(tmp_path / "prediction_log.jsonl")
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_score_known_id_returns_200_with_expected_shape(client):
    response = client.get("/score/1")

    assert response.status_code == 200
    body = response.json()
    assert body["sk_id_curr"] == 1
    assert 0.0 <= body["probability"] <= 1.0
    assert body["decision"] in {"alto_riesgo", "riesgo_aceptable"}


def test_score_unknown_id_returns_404(client):
    response = client.get("/score/999999")

    assert response.status_code == 404


def test_explain_known_id_returns_shap_and_reason_codes(client):
    response = client.get("/explain/1")

    assert response.status_code == 200
    body = response.json()
    assert len(body["shap_values"]) == len(body["feature_names"])
    assert isinstance(body["reason_codes"], list)


def test_explain_unknown_id_returns_404(client):
    response = client.get("/explain/999999")

    assert response.status_code == 404


def test_explain_top_n_query_param(client):
    response = client.get("/explain/1", params={"top_n": 1})

    assert response.status_code == 200
    assert len(response.json()["reason_codes"]) <= 1


def test_score_non_numeric_id_returns_422(client):
    response = client.get("/score/not-a-number")

    assert response.status_code == 422


def test_explain_zero_or_negative_top_n_returns_422(client):
    # pandas Series.head(negative) silently drops rows from the end instead of
    # erroring, so the API must reject non-positive top_n at the boundary.
    assert client.get("/explain/1", params={"top_n": 0}).status_code == 422
    assert client.get("/explain/1", params={"top_n": -1}).status_code == 422


def test_openapi_docs_are_exposed(client):
    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/score/{sk_id_curr}" in paths
    assert "/explain/{sk_id_curr}" in paths

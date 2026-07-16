"""Tests para el logging de predicciones de /score (RNF-4, paso 8, prd.md 9.1).

TDD: se escribe antes de tocar app/api.py. Mismo patron de DI que el resto
de las dependencias del modulo: get_prediction_log_path se sobreescribe via
dependency_overrides para que el test escriba a un archivo temporal, no al
log real del proyecto.
"""

import json

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
def client_factory(tmp_path):
    from app.api import app, get_prediction_log_path, get_service

    log_path = tmp_path / "prediction_log.jsonl"
    service = _synthetic_service()
    app.dependency_overrides[get_service] = lambda: service
    app.dependency_overrides[get_prediction_log_path] = lambda: str(log_path)

    def _make():
        return TestClient(app), log_path

    yield _make
    app.dependency_overrides.clear()


def test_score_appends_a_log_entry_with_prediction(client_factory):
    client, log_path = client_factory()

    response = client.get("/score/1")

    assert response.status_code == 200
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["sk_id_curr"] == 1
    assert entry["decision"] == response.json()["decision"]
    assert entry["probability"] == response.json()["probability"]
    assert "timestamp" in entry


def test_score_appends_one_entry_per_call(client_factory):
    client, log_path = client_factory()

    client.get("/score/1")
    client.get("/score/2")

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2


def test_score_unknown_id_does_not_write_a_log_entry(client_factory):
    client, log_path = client_factory()

    response = client.get("/score/999999")

    assert response.status_code == 404
    assert not log_path.exists()

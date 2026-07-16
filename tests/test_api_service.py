"""Tests para src/credixai/api.py: ScoringService (paso 2, prd.md 9.1).

TDD: este archivo se escribe antes que credixai/api.py. Define el contrato
de ScoringService con datos sinteticos (sin cargar el dataset real ni
reentrenar el modelo completo), reutilizando build_xgboost/compute_shap/
reason_codes ya validados en modeling.py y explainability.py.
"""

import numpy as np
import pandas as pd
import pytest

from credixai.explainability import PROTECTED_FEATURES
from credixai.modeling import build_xgboost


def _synthetic_bundle(n: int = 300):
    rng = np.random.RandomState(42)
    signal = rng.normal(0, 1, n)
    X = pd.DataFrame({
        "EXT_SOURCE_2": signal + rng.normal(0, 0.2, n),
        "credit_to_income": rng.uniform(0.5, 5, n),
        "CODE_GENDER_M": rng.randint(0, 2, n),
    })
    y = pd.Series((signal > np.median(signal)).astype(int), name="TARGET")
    model = build_xgboost(X, y)
    proba_full = model.predict_proba(X)[:, 1]
    threshold = float(np.quantile(proba_full, 1 - y.mean()))
    train_full = pd.DataFrame({"SK_ID_CURR": np.arange(1, n + 1)})

    return {
        "model": model,
        "X": X,
        "y": y,
        "train_full": train_full,
        "proba_full": proba_full,
        "threshold": threshold,
    }


@pytest.fixture
def service():
    from credixai.api import ScoringService

    return ScoringService(_synthetic_bundle())


def test_score_returns_probability_matching_bundle_and_correct_decision(service):
    sk_id = int(service.bundle["train_full"]["SK_ID_CURR"].iloc[0])
    expected_proba = float(service.bundle["proba_full"][0])

    result = service.score(sk_id)

    assert result.sk_id_curr == sk_id
    assert result.probability == pytest.approx(expected_proba)
    assert result.threshold == pytest.approx(service.bundle["threshold"])
    expected_decision = "alto_riesgo" if expected_proba >= service.bundle["threshold"] else "riesgo_aceptable"
    assert result.decision == expected_decision


def test_score_unknown_sk_id_raises_keyerror(service):
    with pytest.raises(KeyError):
        service.score(999_999)


def test_explain_returns_one_shap_value_per_feature(service):
    sk_id = int(service.bundle["train_full"]["SK_ID_CURR"].iloc[0])

    result = service.explain(sk_id)

    assert result.sk_id_curr == sk_id
    assert len(result.shap_values) == len(result.feature_names)
    assert set(result.feature_names) == set(service.bundle["X"].columns)
    assert isinstance(result.base_value, float)


def test_explain_never_leaks_protected_features_in_reason_codes(service):
    # find (or force) a high-risk row so reason_codes has content to check
    proba = service.bundle["proba_full"]
    threshold = service.bundle["threshold"]
    high_risk_positions = np.where(proba >= threshold)[0]
    assert len(high_risk_positions) > 0, "el fixture debe tener al menos una fila de alto riesgo"
    sk_id = int(service.bundle["train_full"]["SK_ID_CURR"].iloc[high_risk_positions[0]])

    result = service.explain(sk_id)

    assert result.decision == "alto_riesgo"
    joined = " ".join(result.reason_codes)
    assert all(feat not in joined for feat in PROTECTED_FEATURES)


def test_explain_low_risk_row_has_no_reason_codes(service):
    proba = service.bundle["proba_full"]
    threshold = service.bundle["threshold"]
    low_risk_positions = np.where(proba < threshold)[0]
    assert len(low_risk_positions) > 0
    sk_id = int(service.bundle["train_full"]["SK_ID_CURR"].iloc[low_risk_positions[0]])

    result = service.explain(sk_id)

    assert result.decision == "riesgo_aceptable"
    assert result.reason_codes == []


def test_explain_unknown_sk_id_raises_keyerror(service):
    with pytest.raises(KeyError):
        service.explain(999_999)


def test_explain_respects_top_n(service):
    proba = service.bundle["proba_full"]
    threshold = service.bundle["threshold"]
    high_risk_positions = np.where(proba >= threshold)[0]
    sk_id = int(service.bundle["train_full"]["SK_ID_CURR"].iloc[high_risk_positions[0]])

    result = service.explain(sk_id, top_n=1)

    assert len(result.reason_codes) <= 1

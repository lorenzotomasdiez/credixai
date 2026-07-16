"""Tests para src/credixai/modeling.py (Tarea 4: modelado supervisado)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from credixai.modeling import NON_FEATURE_COLS, build_baseline, build_xgboost, evaluate, feature_columns


def test_evaluate_prints_when_label_is_given(capsys):
    y_true = [0, 1, 0, 1]
    y_proba = [0.1, 0.9, 0.2, 0.8]

    evaluate(y_true, y_proba, label="test-label")

    assert "[test-label]" in capsys.readouterr().out


def test_feature_columns_excludes_id_target_and_split():
    features = pd.DataFrame(columns=["SK_ID_CURR", "TARGET", "IS_TRAIN", "EXT_SOURCE_2", "credit_to_income"])

    cols = feature_columns(features)

    assert cols == ["EXT_SOURCE_2", "credit_to_income"]
    assert not set(cols) & set(NON_FEATURE_COLS)


def test_evaluate_matches_sklearn_reference_metrics():
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, 200)
    y_proba = np.clip(y_true + rng.normal(0, 0.4, 200), 0, 1)

    metrics = evaluate(y_true, y_proba)

    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(y_true, y_proba))
    assert metrics["pr_auc"] == pytest.approx(average_precision_score(y_true, y_proba))
    assert metrics["brier"] == pytest.approx(brier_score_loss(y_true, y_proba))
    assert 0.0 <= metrics["ks"] <= 1.0


def _synthetic_training_data(n: int = 200):
    rng = np.random.RandomState(42)
    signal = rng.normal(0, 1, n)
    X = pd.DataFrame({
        "EXT_SOURCE_2": signal + rng.normal(0, 0.2, n),
        "credit_to_income": rng.uniform(0.5, 5, n),
        "with_nan": np.where(rng.uniform(size=n) < 0.1, np.nan, rng.normal(0, 1, n)),
    })
    y = pd.Series((signal > np.median(signal)).astype(int), name="TARGET")
    return X, y


def test_build_baseline_fits_and_predicts_probabilities():
    X, y = _synthetic_training_data()

    pipeline = build_baseline(X, y)
    proba = pipeline.predict_proba(X)[:, 1]

    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()
    # the baseline must learn better-than-random separation on this synthetic signal
    assert roc_auc_score(y, proba) > 0.6


def test_build_xgboost_handles_native_nan_and_predicts_probabilities():
    X, y = _synthetic_training_data()
    assert X["with_nan"].isna().any(), "el fixture debe incluir NaN para probar el manejo nativo de XGBoost"

    model = build_xgboost(X, y)
    proba = model.predict_proba(X)[:, 1]

    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert roc_auc_score(y, proba) > 0.6

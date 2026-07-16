"""Tests para src/credixai/explainability.py (Tarea 5: XAI y fairness)."""

import numpy as np
import pandas as pd
import pytest
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio

from credixai.explainability import (
    PROTECTED_FEATURES,
    compute_shap,
    fairness_report,
    mean_abs_shap,
    reason_codes,
)
from credixai.modeling import build_xgboost


def test_mean_abs_shap_ranks_by_average_magnitude():
    shap_values = np.array([
        [1.0, -0.1, 0.0],
        [-1.0, 0.1, 0.0],
        [0.5, -5.0, 0.0],
    ])
    feature_names = ["a", "b", "c"]

    ranking = mean_abs_shap(shap_values, feature_names)

    assert ranking.iloc[0]["feature"] == "b"  # mean(|.|) = (0.1+0.1+5.0)/3, highest
    assert ranking.iloc[-1]["feature"] == "c"  # always zero, lowest
    assert list(ranking["mean_abs_shap"]) == sorted(ranking["mean_abs_shap"], reverse=True)


def test_reason_codes_excludes_protected_features_and_negative_contributions():
    feature_names = ["EXT_SOURCE_2", "CODE_GENDER_M", "DAYS_BIRTH", "bureau_debt_mean", "credit_to_goods"]
    # positive SHAP = pushes toward higher risk; protected features have the
    # largest positive contribution here, and must still never appear.
    shap_row = np.array([0.5, 0.9, 0.8, 0.3, -0.2])

    reasons = reason_codes(shap_row, feature_names, top_n=4)

    assert len(reasons) == 2  # only 2 features have positive contribution and aren't protected
    assert all(feat not in " ".join(reasons) for feat in PROTECTED_FEATURES)
    # EXT_SOURCE_2 (0.5) must be listed before bureau_debt_mean (0.3): descending order
    assert reasons[0] == "Score de riesgo externo (fuente 2) por debajo del promedio."


def test_reason_codes_falls_back_to_generic_text_for_unmapped_feature():
    reasons = reason_codes(np.array([0.7]), ["some_unmapped_feature"], top_n=4)

    assert reasons == ["Valor de 'some_unmapped_feature' desfavorable para el perfil."]


def test_reason_codes_respects_top_n():
    feature_names = [f"f{i}" for i in range(10)]
    shap_row = np.arange(10, dtype=float)  # all positive, increasing

    reasons = reason_codes(shap_row, feature_names, top_n=3)

    assert len(reasons) == 3


def test_fairness_report_matches_fairlearn_reference_and_flags_amplified_disparity():
    rng = np.random.RandomState(0)
    n = 500
    group = rng.randint(0, 2, n)
    # group 1 has a much higher predicted "high risk" rate than group 0,
    # simulating the amplification pattern documented for the real model.
    y_pred = np.where(group == 1, rng.uniform(size=n) < 0.5, rng.uniform(size=n) < 0.1).astype(int)
    y_true = np.where(group == 1, rng.uniform(size=n) < 0.3, rng.uniform(size=n) < 0.2).astype(int)

    report = fairness_report(y_true, y_pred, group)

    assert report["statistical_parity_difference"] == pytest.approx(
        demographic_parity_difference(y_true, y_pred, sensitive_features=group)
    )
    assert report["disparate_impact"] == pytest.approx(
        demographic_parity_ratio(y_true, y_pred, sensitive_features=group)
    )
    assert set(report["selection_rate_by_group"].keys()) == {0, 1}
    # the synthetic setup makes group 1's selection rate much higher than group 0's
    assert report["selection_rate_by_group"][1] > report["selection_rate_by_group"][0]


def test_compute_shap_returns_one_row_per_input_sample():
    rng = np.random.RandomState(0)
    n = 60
    X = pd.DataFrame({
        "EXT_SOURCE_2": rng.normal(0, 1, n),
        "credit_to_income": rng.uniform(0.5, 5, n),
    })
    y = pd.Series((X["EXT_SOURCE_2"] > 0).astype(int))
    model = build_xgboost(X, y)

    explainer, shap_values = compute_shap(model, X)

    assert shap_values.shape == (n, X.shape[1])
    assert isinstance(explainer.expected_value, (float, np.floating))

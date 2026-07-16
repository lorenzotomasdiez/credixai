"""Tests para src/credixai/dashboard.py: el modulo que app/dashboard.py y
app/api.py consumen para entrenar el modelo final, segmentar y auditar
fairness. Hasta este archivo no tenia tests directos, aunque es el unico
punto de entrada compartido por ambos front-ends.
"""

import numpy as np
import pandas as pd
import pytest

from credixai.clustering import CLUSTER_COLS
from credixai.dashboard import (
    application_reason_codes,
    compute_fairness,
    compute_segments,
    sample_for_shap,
    segment_profile,
    train_full_model,
)
from credixai.explainability import PROTECTED_FEATURES
from credixai.modeling import NON_FEATURE_COLS


def _synthetic_features(n: int = 300) -> pd.DataFrame:
    """Tabla sintetica con las mismas columnas que features.parquet real
    necesita: base del modelo (NON_FEATURE_COLS + señal predictiva) mas
    todas las columnas que build_segments requiere (CLUSTER_COLS).
    """
    rng = np.random.RandomState(42)
    signal = rng.normal(0, 1, n)
    gender = rng.randint(0, 2, n)

    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(1, n + 1),
        "IS_TRAIN": 1,
        "TARGET": (signal > np.median(signal)).astype(int),
        "DAYS_BIRTH": rng.randint(-25000, -7500, n),
        "DAYS_EMPLOYED": rng.randint(-9000, -100, n),
        "AMT_INCOME_TOTAL": rng.uniform(20_000, 300_000, n),
        "AMT_CREDIT": rng.uniform(30_000, 500_000, n),
        "CODE_GENDER_M": gender,
        "EXT_SOURCE_2": signal + rng.normal(0, 0.2, n),
        "CNT_CHILDREN": rng.randint(0, 3, n),
        "CNT_FAM_MEMBERS": rng.randint(1, 5, n),
        "credit_to_income": rng.uniform(0.5, 5, n),
        "annuity_to_income": rng.uniform(0.05, 0.5, n),
        "credit_to_goods": rng.uniform(0.8, 1.5, n),
        "bureau_active_count": rng.randint(0, 5, n),
        "bb_months_count_mean": rng.uniform(0, 24, n),
        "prev_approval_rate": rng.uniform(0, 1, n),
        "pos_dpd_rate": rng.uniform(0, 0.3, n),
        "cc_utilization_mean": rng.uniform(0, 1, n),
        "inst_late_rate": rng.uniform(0, 0.3, n),
        "bureau_no_record": rng.choice([True, False], n),
        "bb_no_record": rng.choice([True, False], n),
        "prev_no_record": rng.choice([True, False], n),
        "pos_no_record": rng.choice([True, False], n),
        "cc_no_record": rng.choice([True, False], n),
        "inst_no_record": rng.choice([True, False], n),
    })
    assert set(CLUSTER_COLS) - {"AGE_YEARS", "EMPLOYED_YEARS"} <= set(df.columns)
    return df


@pytest.fixture(scope="module")
def features_df():
    return _synthetic_features()


@pytest.fixture(scope="module")
def bundle(features_df):
    return train_full_model(features_df)


def test_train_full_model_returns_expected_shape_and_valid_probabilities(bundle):
    expected_keys = {"model", "feature_cols", "train_full", "X", "y", "proba_full", "threshold", "holdout_metrics"}
    assert expected_keys <= bundle.keys()
    assert not set(bundle["feature_cols"]) & set(NON_FEATURE_COLS)
    assert len(bundle["proba_full"]) == len(bundle["train_full"])
    assert ((bundle["proba_full"] >= 0) & (bundle["proba_full"] <= 1)).all()
    assert set(bundle["holdout_metrics"].keys()) == {"roc_auc", "pr_auc", "ks", "brier"}
    assert 0.0 <= bundle["holdout_metrics"]["roc_auc"] <= 1.0


def test_train_full_model_threshold_is_the_documented_quantile(bundle):
    y = bundle["y"]
    expected_threshold = np.quantile(bundle["proba_full"], 1 - y.mean())
    assert bundle["threshold"] == pytest.approx(expected_threshold)


def test_compute_segments_keeps_only_train_rows_with_cluster_assigned(features_df, bundle):
    train_segments = compute_segments(features_df, bundle["train_full"])

    assert len(train_segments) == (features_df["IS_TRAIN"] == 1).sum()
    assert {"cluster", "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "SK_ID_CURR"} <= set(train_segments.columns)
    assert train_segments["SK_ID_CURR"].is_unique


def test_segment_profile_aggregates_match_manual_groupby():
    train_segments = pd.DataFrame({
        "SK_ID_CURR": range(6),
        "cluster": [0, 0, 0, 1, 1, 1],
        "TARGET": [1, 0, 0, 0, 0, 1],
        "AMT_INCOME_TOTAL": [10_000, 20_000, 30_000, 100_000, 200_000, 300_000],
        "AMT_CREDIT": [5_000, 15_000, 25_000, 50_000, 60_000, 70_000],
    })

    profile = segment_profile(train_segments)

    row0 = profile.set_index("cluster").loc[0]
    assert row0["n_clientes"] == 3
    assert row0["tasa_default"] == pytest.approx(1 / 3)
    assert row0["ingreso_promedio"] == pytest.approx(20_000)
    assert profile["pct_poblacion"].sum() == pytest.approx(1.0)


def test_compute_fairness_real_rates_match_manual_groupby(bundle):
    fairness = compute_fairness(bundle)

    train_full = bundle["train_full"]
    expected_gender_rates = train_full.assign(
        gender=train_full["CODE_GENDER_M"].astype(int)
    ).groupby("gender")["TARGET"].mean().to_dict()

    assert fairness["real_rate_by_gender"] == pytest.approx(expected_gender_rates)
    assert fairness["threshold"] == pytest.approx(bundle["threshold"])
    assert set(fairness["gender"].keys()) >= {
        "statistical_parity_difference", "disparate_impact", "equal_opportunity_difference"
    }


def test_sample_for_shap_returns_consistent_shapes(bundle):
    shap_bundle = sample_for_shap(bundle, sample_size=50)

    assert len(shap_bundle["X_sample"]) == 50
    assert len(shap_bundle["sk_ids"]) == 50
    assert shap_bundle["shap_values"].shape == (50, bundle["X"].shape[1])
    assert len(shap_bundle["proba_sample"]) == 50
    assert isinstance(shap_bundle["base_value"], (float, np.floating))


def test_sample_for_shap_is_reproducible_with_same_random_state(bundle):
    a = sample_for_shap(bundle, sample_size=20, random_state=1)
    b = sample_for_shap(bundle, sample_size=20, random_state=1)

    assert a["sk_ids"].tolist() == b["sk_ids"].tolist()


def test_application_reason_codes_never_leaks_protected_attributes(bundle):
    shap_bundle = sample_for_shap(bundle, sample_size=50)
    # pick whichever row has the highest predicted probability, most likely to have reasons
    row_idx = int(np.argmax(shap_bundle["proba_sample"]))

    reasons = application_reason_codes(shap_bundle, row_idx, top_n=4)

    joined = " ".join(reasons)
    assert all(feat not in joined for feat in PROTECTED_FEATURES)
    assert len(reasons) <= 4

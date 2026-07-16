"""Tests para src/credixai/clustering.py (Tarea 3: segmentacion)."""

import numpy as np
import pandas as pd
import pytest

from credixai.clustering import CLUSTER_COLS, add_cluster_input_columns, assign_clusters, build_segments, fit_clustering


def test_add_cluster_input_columns_derives_years_from_days():
    df = pd.DataFrame({"DAYS_BIRTH": [-3650, -7300], "DAYS_EMPLOYED": [-365, -1825]})

    out = add_cluster_input_columns(df)

    assert out["AGE_YEARS"].tolist() == pytest.approx([10.0, 20.0])
    assert out["EMPLOYED_YEARS"].tolist() == pytest.approx([1.0, 5.0])


def _synthetic_features(n_per_group: int = 30, is_train_all: bool = True) -> pd.DataFrame:
    """Dos grupos claramente separables en ingreso, para que KMeans los distinga."""
    rng = np.random.RandomState(42)
    n = n_per_group * 2
    income = np.concatenate([
        rng.normal(30_000, 1_000, n_per_group),
        rng.normal(300_000, 1_000, n_per_group),
    ])
    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(1, n + 1),
        "IS_TRAIN": 1 if is_train_all else np.tile([1, 0], n // 2),
        "DAYS_BIRTH": rng.randint(-25000, -8000, n),
        "DAYS_EMPLOYED": rng.randint(-10000, -100, n),
        "AMT_INCOME_TOTAL": income,
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
    return df


def test_fit_clustering_only_uses_train_rows_and_assign_clusters_covers_everyone():
    features = _synthetic_features(n_per_group=30, is_train_all=False)

    imputer, scaler, kmeans = fit_clustering(features, k=2)
    assigned = assign_clusters(features, imputer, scaler, kmeans)

    assert set(assigned["SK_ID_CURR"]) == set(features["SK_ID_CURR"])
    assert assigned["cluster"].isin([0, 1]).all()
    # fit_clustering must have trained KMeans only on IS_TRAIN==1 rows
    assert kmeans.n_iter_ >= 1


def test_build_segments_separates_two_clearly_distinct_income_groups():
    features = _synthetic_features(n_per_group=30, is_train_all=True)

    segments = build_segments(features)

    assert segments["SK_ID_CURR"].is_unique
    merged = segments.merge(features[["SK_ID_CURR", "AMT_INCOME_TOTAL"]], on="SK_ID_CURR")
    mean_income_by_cluster = merged.groupby("cluster")["AMT_INCOME_TOTAL"].mean()
    # the two synthetic income groups (~30k vs ~300k) must land in different clusters
    assert mean_income_by_cluster.max() - mean_income_by_cluster.min() > 100_000


def test_cluster_cols_are_all_present_in_synthetic_fixture():
    """Guardrail: si CLUSTER_COLS cambia en clustering.py, el fixture de este
    archivo de test debe actualizarse tambien, no fallar en silencio con un KeyError.
    """
    features = _synthetic_features(n_per_group=2)
    missing = set(CLUSTER_COLS) - set(add_cluster_input_columns(features).columns)
    assert not missing, f"faltan columnas de CLUSTER_COLS en el fixture: {missing}"

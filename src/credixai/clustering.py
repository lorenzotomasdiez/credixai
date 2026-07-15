"""Clustering / segmentacion de perfiles (Tarea 3) para Home Credit Default Risk.

Reproduce, en forma de funciones reutilizables, la logica ya validada
interactivamente en notebooks/03_clustering.ipynb. Ver docs/informe-final.md
seccion 3 para la justificacion e interpretacion de cada paso.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

CLUSTER_COLS = [
    "AGE_YEARS", "EMPLOYED_YEARS", "AMT_INCOME_TOTAL", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    "credit_to_income", "annuity_to_income", "credit_to_goods",
    "bureau_active_count", "bb_months_count_mean", "prev_approval_rate",
    "pos_dpd_rate", "cc_utilization_mean", "inst_late_rate",
    "bureau_no_record", "bb_no_record", "prev_no_record",
    "pos_no_record", "cc_no_record", "inst_no_record",
]

K_FINAL = 5


def add_cluster_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva AGE_YEARS/EMPLOYED_YEARS a partir de DAYS_BIRTH/DAYS_EMPLOYED."""
    df = df.copy()
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365
    return df


def fit_clustering(features: pd.DataFrame, k: int = K_FINAL):
    """Ajusta imputer/scaler/KMeans sobre la porcion train (IS_TRAIN == 1).

    Devuelve (imputer, scaler, kmeans) ya ajustados, para aplicar despues
    con transform/predict sobre cualquier poblacion (train o test).
    """
    train_df = add_cluster_input_columns(features.loc[features["IS_TRAIN"] == 1])

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

    X_imputed = imputer.fit_transform(train_df[CLUSTER_COLS])
    X_scaled = scaler.fit_transform(X_imputed)
    kmeans.fit(X_scaled)

    return imputer, scaler, kmeans


def assign_clusters(features: pd.DataFrame, imputer, scaler, kmeans) -> pd.DataFrame:
    """Aplica (transform/predict) el pipeline ya ajustado a toda la poblacion."""
    df = add_cluster_input_columns(features)
    X_imputed = imputer.transform(df[CLUSTER_COLS])
    X_scaled = scaler.transform(X_imputed)
    df["cluster"] = kmeans.predict(X_scaled)
    return df[["SK_ID_CURR", "IS_TRAIN", "cluster"]]


def build_segments(features: pd.DataFrame) -> pd.DataFrame:
    """Orquesta el pipeline completo de la Tarea 3: fit sobre train, predict sobre todos."""
    imputer, scaler, kmeans = fit_clustering(features)
    segments = assign_clusters(features, imputer, scaler, kmeans)
    assert segments["SK_ID_CURR"].is_unique, "SK_ID_CURR duplicado en segments"
    return segments

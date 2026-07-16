"""Visualizacion e informe ejecutivo (Tarea 6) para Home Credit Default Risk.

Funciones de carga y preparacion de datos que consume app/dashboard.py
(Streamlit). Reutiliza el modelo (Tarea 4), la segmentacion (Tarea 3) y la
explicabilidad/fairness (Tarea 5) ya validados, sin reimplementar logica.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from credixai.clustering import build_segments
from credixai.explainability import compute_shap, fairness_report, reason_codes
from credixai.modeling import build_xgboost, evaluate, feature_columns

DATA_DIR = "data/processed"
SEGMENT_LABELS = {
    0: "Segmento 0",
    1: "Segmento 1",
    2: "Segmento 2",
    3: "Segmento 3",
    4: "Segmento 4",
}


def load_features() -> pd.DataFrame:
    return pd.read_parquet(f"{DATA_DIR}/features.parquet")


def train_full_model(features: pd.DataFrame):
    """Reentrena XGBoost final sobre el 100% de train (misma logica que Tarea 4/5)."""
    train_full = features.loc[features["IS_TRAIN"] == 1].reset_index(drop=True)
    feature_cols = feature_columns(features)
    X = train_full[feature_cols]
    y = train_full["TARGET"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    holdout_model = build_xgboost(X_train, y_train)
    holdout_metrics = evaluate(y_val, holdout_model.predict_proba(X_val)[:, 1])

    model = build_xgboost(X, y)
    proba_full = model.predict_proba(X)[:, 1]
    threshold = np.quantile(proba_full, 1 - y.mean())

    return {
        "model": model,
        "feature_cols": feature_cols,
        "train_full": train_full,
        "X": X,
        "y": y,
        "proba_full": proba_full,
        "threshold": threshold,
        "holdout_metrics": holdout_metrics,
    }


def compute_segments(features: pd.DataFrame, train_full: pd.DataFrame) -> pd.DataFrame:
    segments = build_segments(features)
    train_segments = segments.loc[segments["IS_TRAIN"] == 1].merge(
        train_full[["SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT"]],
        on="SK_ID_CURR",
    )
    return train_segments


def segment_profile(train_segments: pd.DataFrame) -> pd.DataFrame:
    profile = train_segments.groupby("cluster").agg(
        n_clientes=("SK_ID_CURR", "count"),
        tasa_default=("TARGET", "mean"),
        ingreso_promedio=("AMT_INCOME_TOTAL", "mean"),
        credito_promedio=("AMT_CREDIT", "mean"),
    )
    profile["pct_poblacion"] = profile["n_clientes"] / profile["n_clientes"].sum()
    return profile.reset_index()


def compute_fairness(bundle: dict) -> dict:
    train_full = bundle["train_full"]
    y = bundle["y"]
    proba_full = bundle["proba_full"]
    threshold = bundle["threshold"]
    y_pred = (proba_full >= threshold).astype(int)

    gender = train_full["CODE_GENDER_M"].astype(int)
    age_years = -train_full["DAYS_BIRTH"] / 365.25
    age_group = pd.cut(age_years, bins=[0, 30, 45, 60, 100], labels=["<30", "30-45", "45-60", "60+"])

    real_by_gender = train_full.assign(gender=gender).groupby("gender")["TARGET"].mean()
    real_by_age = train_full.assign(age_group=age_group).groupby("age_group", observed=True)["TARGET"].mean()

    return {
        "gender": fairness_report(y, y_pred, gender),
        "age": fairness_report(y, y_pred, age_group),
        "real_rate_by_gender": real_by_gender.to_dict(),
        "real_rate_by_age": real_by_age.to_dict(),
        "threshold": threshold,
    }


def sample_for_shap(bundle: dict, sample_size: int = 2000, random_state: int = 42):
    X = bundle["X"]
    rng = np.random.RandomState(random_state)
    sample_idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    X_sample = X.iloc[sample_idx].reset_index(drop=True)
    sk_ids = bundle["train_full"]["SK_ID_CURR"].iloc[sample_idx].reset_index(drop=True)
    explainer, shap_values = compute_shap(bundle["model"], X_sample)
    proba_sample = bundle["model"].predict_proba(X_sample)[:, 1]
    return {
        "X_sample": X_sample,
        "sk_ids": sk_ids,
        "shap_values": shap_values,
        "proba_sample": proba_sample,
        "base_value": explainer.expected_value,
    }


def application_reason_codes(shap_bundle: dict, row_idx: int, top_n: int = 4) -> list:
    return reason_codes(
        shap_bundle["shap_values"][row_idx], shap_bundle["X_sample"].columns, top_n=top_n
    )

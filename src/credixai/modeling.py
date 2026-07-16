"""Modelado predictivo supervisado (Tarea 4) para Home Credit Default Risk.

Reproduce, en forma de funciones reutilizables, la logica ya validada
interactivamente en notebooks/04_modeling.ipynb. Ver docs/informe-final.md
seccion 4 para la justificacion e interpretacion de cada paso.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

NON_FEATURE_COLS = ["SK_ID_CURR", "TARGET", "IS_TRAIN"]

XGB_PARAMS = dict(
    n_estimators=1086,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=20,
    reg_lambda=1.0,
    scale_pos_weight=1.0,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
)


def feature_columns(features: pd.DataFrame) -> list:
    """Todas las columnas de features.parquet salvo id/etiqueta/split."""
    return [c for c in features.columns if c not in NON_FEATURE_COLS]


def evaluate(y_true, y_proba, label: str = "") -> dict:
    """ROC-AUC, PR-AUC, KS y Brier score, las 4 metricas de evaluacion."""
    auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ks = np.max(tpr - fpr)

    metrics = {"roc_auc": auc, "pr_auc": pr_auc, "ks": ks, "brier": brier}
    if label:
        print(f"[{label}] ROC-AUC={auc:.4f}  PR-AUC={pr_auc:.4f}  KS={ks:.4f}  Brier={brier:.4f}")
    return metrics


def build_baseline(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Baseline interpretable: Regresion Logistica con imputacion+escalado."""
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X, y)
    return pipeline


def build_xgboost(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    """Modelo primario: XGBoost con los hiperparametros finales, sin early stopping.

    n_estimators=1086 es el promedio de best_iteration observado en el holdout
    y en la validacion cruzada 5-fold del notebook (ver docs/informe-final.md S4).
    """
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X, y, verbose=False)
    return model

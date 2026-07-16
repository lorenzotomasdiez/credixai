"""Tarea 4: entrena el baseline (LogReg) y el modelo primario
(XGBoost) de default sobre data/processed/features.parquet, registra ambos
en MLflow y persiste el modelo final para servir en tareas posteriores.

Requiere haber corrido antes scripts/02_features.py.

Uso:
    uv run python scripts/04_modeling.py
"""

import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split

from credixai.modeling import XGB_PARAMS, build_baseline, build_xgboost, evaluate, feature_columns

DATA_DIR = "data/processed"


def main() -> None:
    features = pd.read_parquet(f"{DATA_DIR}/features.parquet")
    train_full = features.loc[features["IS_TRAIN"] == 1]

    feature_cols = feature_columns(features)
    X = train_full[feature_cols]
    y = train_full["TARGET"].astype(int)

    # Holdout solo para reportar metricas honestas (no vistas en entrenamiento);
    # el mismo split (test_size=0.2, random_state=42) usado en el notebook.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("credixai-tarea4-modelado")

    baseline_holdout = build_baseline(X_train, y_train)
    baseline_metrics = evaluate(y_val, baseline_holdout.predict_proba(X_val)[:, 1], "LogReg (val)")

    xgb_holdout = build_xgboost(X_train, y_train)
    xgb_metrics = evaluate(y_val, xgb_holdout.predict_proba(X_val)[:, 1], "XGBoost (val)")

    # Artefactos finales: reentrenados sobre el 100% del train para servir,
    # ya con los hiperparametros fijos validados (no requieren holdout).
    baseline = build_baseline(X, y)
    with mlflow.start_run(run_name="baseline-logreg"):
        mlflow.log_params({"class_weight": "balanced", "max_iter": 1000})
        mlflow.log_metrics(baseline_metrics)
        mlflow.sklearn.log_model(
            baseline, name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )

    xgb_final = build_xgboost(X, y)
    with mlflow.start_run(run_name="xgboost-final"):
        mlflow.log_params(XGB_PARAMS)
        mlflow.log_metrics(xgb_metrics)
        mlflow.xgboost.log_model(xgb_final, name="model")

    print("Runs registrados en MLflow (sqlite:///mlflow.db)")


if __name__ == "__main__":
    main()

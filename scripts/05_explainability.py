"""Tarea 5: SHAP global/local, reason codes de adverse action y
auditoria de fairness (Fairlearn) sobre el modelo XGBoost final de la Tarea 4.

Requiere haber corrido antes scripts/02_features.py. Los contrafacticos (DiCE)
quedan solo en notebooks/05_xai.ipynb (ver seccion 6: no son confiables para
la poblacion con historial incompleto en este modelo, por el manejo nativo
de NaN de XGBoost).

Uso:
    uv run python scripts/05_explainability.py
"""

import numpy as np
import pandas as pd

from credixai.explainability import compute_shap, fairness_report, mean_abs_shap, reason_codes
from credixai.modeling import build_xgboost, feature_columns

DATA_DIR = "data/processed"
SAMPLE_SIZE = 5000
RANDOM_STATE = 42


def main() -> None:
    features = pd.read_parquet(f"{DATA_DIR}/features.parquet")
    train_full = features.loc[features["IS_TRAIN"] == 1].reset_index(drop=True)

    feature_cols = feature_columns(features)
    X = train_full[feature_cols]
    y = train_full["TARGET"].astype(int)

    model = build_xgboost(X, y)

    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(len(X), size=SAMPLE_SIZE, replace=False)
    X_sample = X.iloc[sample_idx].reset_index(drop=True)

    _, shap_values = compute_shap(model, X_sample)
    importance = mean_abs_shap(shap_values, X_sample.columns)
    print("Top-10 features por SHAP (mean_abs_shap):")
    print(importance.head(10).to_string(index=False))

    proba_sample = model.predict_proba(X_sample)[:, 1]
    idx_high = int(np.argmax(proba_sample))
    print(f"\nReason codes, solicitud de mayor riesgo (proba={proba_sample[idx_high]:.4f}):")
    for i, reason in enumerate(reason_codes(shap_values[idx_high], X_sample.columns), start=1):
        print(f"  {i}. {reason}")

    proba_full = model.predict_proba(X)[:, 1]
    threshold = np.quantile(proba_full, 1 - y.mean())
    y_pred = (proba_full >= threshold).astype(int)

    gender = train_full["CODE_GENDER_M"].astype(int)
    age_years = -train_full["DAYS_BIRTH"] / 365.25
    age_group = pd.cut(age_years, bins=[0, 30, 45, 60, 100], labels=["<30", "30-45", "45-60", "60+"])

    print(f"\nUmbral de decision: {threshold:.4f}")
    print("\nFairness por genero:")
    print(fairness_report(y, y_pred, gender))
    print("\nFairness por grupo etario:")
    print(fairness_report(y, y_pred, age_group))


if __name__ == "__main__":
    main()

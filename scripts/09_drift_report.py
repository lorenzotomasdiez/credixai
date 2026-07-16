"""Paso 8 (prd.md 9.1): monitoreo de drift (Evidently) sobre application_test vs application_train.

Home Credit Default Risk es un dataset estatico de Kaggle, sin trafico
productivo real a lo largo del tiempo. La poblacion "actual" mas honesta
disponible es application_test (IS_TRAIN==0 en features.parquet): solicitudes
reales que Kaggle reserva sin TARGET para su competencia, nunca usadas para
entrenar el modelo (IS_TRAIN==1, la referencia).

Este script:
1. Batch-scorea application_test con el modelo ya entrenado (mismo bundle
   que credixai.dashboard.train_full_model produce para /score), logueando
   cada prediccion con credixai.monitoring.logging.append_log_entry -- el
   mismo mecanismo que usa /score en produccion (RNF-4, "logs de
   prediccion") -- a models/monitoring/prediction_log.jsonl.
2. Corre Evidently (DataDriftPreset, PSI por columna) comparando la
   distribucion de features de application_train (referencia) contra
   application_test (actual), guarda un reporte HTML y resume las columnas
   que superan el umbral de alerta (PSI>0.2, RNF-4) via
   credixai.monitoring.drift.alert_columns.

Sin perturbacion sintetica: el PSI que se reporta es el real, sea alto o
bajo (interpretacion en docs/informe-final.md seccion 8.8). Cada corrida
agrega mas entradas al log de predicciones, igual que un job de batch
scoring real.

Requiere haber corrido antes scripts/02_features.py.

Uso:
    uv run python scripts/09_drift_report.py
"""

from datetime import datetime, timezone

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from credixai.api import ScoreResult
from credixai.dashboard import load_features, train_full_model
from credixai.modeling import feature_columns
from credixai.monitoring.drift import DEFAULT_PSI_THRESHOLD, alert_columns
from credixai.monitoring.logging import append_log_entry, format_log_entry

PREDICTION_LOG_PATH = "models/monitoring/prediction_log.jsonl"
DRIFT_REPORT_PATH = "models/monitoring/drift_report.html"


def _decision(probability: float, threshold: float) -> str:
    return "alto_riesgo" if probability >= threshold else "riesgo_aceptable"


def _score_and_log_current_population(model, threshold: float, current_full: pd.DataFrame, feature_cols: list) -> None:
    probabilities = model.predict_proba(current_full[feature_cols])[:, 1]
    for sk_id_curr, probability in zip(current_full["SK_ID_CURR"], probabilities):
        result = ScoreResult(
            sk_id_curr=int(sk_id_curr),
            probability=float(probability),
            threshold=threshold,
            decision=_decision(float(probability), threshold),
        )
        entry = format_log_entry(result, timestamp=datetime.now(timezone.utc).isoformat())
        append_log_entry(entry, PREDICTION_LOG_PATH)


def _run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    report = Report(metrics=[DataDriftPreset(stattest="psi", stattest_threshold=DEFAULT_PSI_THRESHOLD)])
    report.run(reference_data=reference, current_data=current)
    report.save_html(DRIFT_REPORT_PATH)

    drift_table = report.as_dict()["metrics"][1]["result"]
    return {col: info["drift_score"] for col, info in drift_table["drift_by_columns"].items()}


def main() -> None:
    features = load_features()
    bundle = train_full_model(features)
    feature_cols = feature_columns(features)
    threshold = float(bundle["threshold"])

    current_full = features.loc[features["IS_TRAIN"] == 0].reset_index(drop=True)
    print(f"Batch-scoreando {len(current_full)} solicitudes de application_test...")
    _score_and_log_current_population(bundle["model"], threshold, current_full, feature_cols)

    reference = bundle["train_full"][feature_cols]
    current = current_full[feature_cols]

    print("Corriendo Evidently DataDriftPreset (PSI)...")
    psi_by_column = _run_drift_report(reference, current)
    alerts = alert_columns(psi_by_column, threshold=DEFAULT_PSI_THRESHOLD)

    print(f"\n=== Monitoreo de drift (RNF-4, umbral PSI > {DEFAULT_PSI_THRESHOLD}) ===")
    print(f"Reporte HTML: {DRIFT_REPORT_PATH}")
    print(f"Columnas evaluadas: {len(psi_by_column)}")
    print(f"Columnas con alerta de drift: {len(alerts)}")
    for column, psi in sorted(alerts.items(), key=lambda kv: -kv[1]):
        print(f"  ALERTA {column}: PSI={psi:.4f}")
    if not alerts:
        print("  Sin alertas: ninguna columna supera PSI > 0.2.")


if __name__ == "__main__":
    main()

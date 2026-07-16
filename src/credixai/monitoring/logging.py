"""Logs de prediccion (RNF-4, paso 8, prd.md 9.1).

format_log_entry es pura (recibe el timestamp en vez de generarlo) para que
el resto del pipeline (drift, tests) no dependa del reloj real.
append_log_entry es el unico punto con IO real (append-only, crea el
directorio padre si no existe): lo usan tanto /score en app/api.py como el
batch-scoring de application_test en scripts/09_drift_report.py.
"""

import json
from pathlib import Path

from credixai.api import ScoreResult


def format_log_entry(result: ScoreResult, timestamp: str) -> dict:
    return {
        "timestamp": timestamp,
        "sk_id_curr": result.sk_id_curr,
        "probability": result.probability,
        "threshold": result.threshold,
        "decision": result.decision,
    }


def append_log_entry(entry: dict, path: str) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")

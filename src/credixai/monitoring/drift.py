"""Alerta de drift por PSI (RNF-4, paso 8).

alert_columns es pura: el calculo de PSI en si lo hace Evidently
(scripts/09_drift_report.py) contra los datos reales, y se verifica
manualmente contra el reporte real en vez de mockearlo aca.
"""

DEFAULT_PSI_THRESHOLD = 0.2


def alert_columns(psi_by_column: dict[str, float], threshold: float = DEFAULT_PSI_THRESHOLD) -> dict[str, float]:
    return {column: psi for column, psi in psi_by_column.items() if psi > threshold}

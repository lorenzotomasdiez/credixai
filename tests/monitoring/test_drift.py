"""Tests para credixai.monitoring.drift (paso 8, prd.md 9.1).

TDD: se escribe antes que credixai/monitoring/drift.py.
alert_columns es pura: separa la logica de "que columnas superan el umbral
de alerta" del calculo de PSI en si (que hace Evidently, y se verifica
manualmente contra el reporte real, no con un mock del calculo estadistico).
"""

from credixai.monitoring.drift import DEFAULT_PSI_THRESHOLD, alert_columns


def test_alert_columns_returns_columns_above_threshold():
    psi_by_column = {"EXT_SOURCE_2": 0.35, "AMT_INCOME_TOTAL": 0.05, "CREDIT_TO_INCOME": 0.21}

    alerts = alert_columns(psi_by_column, threshold=0.2)

    assert alerts == {"EXT_SOURCE_2": 0.35, "CREDIT_TO_INCOME": 0.21}


def test_alert_columns_excludes_column_exactly_at_threshold():
    psi_by_column = {"AMT_INCOME_TOTAL": 0.2}

    alerts = alert_columns(psi_by_column, threshold=0.2)

    assert alerts == {}


def test_alert_columns_returns_empty_dict_when_nothing_drifted():
    psi_by_column = {"AMT_INCOME_TOTAL": 0.02, "EXT_SOURCE_2": 0.01}

    alerts = alert_columns(psi_by_column, threshold=0.2)

    assert alerts == {}


def test_alert_columns_uses_default_psi_threshold_of_0_2():
    psi_by_column = {"EXT_SOURCE_2": 0.25}

    alerts = alert_columns(psi_by_column)

    assert alerts == {"EXT_SOURCE_2": 0.25}
    assert DEFAULT_PSI_THRESHOLD == 0.2

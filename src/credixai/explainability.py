"""Explicabilidad y fairness (Tarea 5) para Home Credit Default Risk.

Reproduce, en forma de funciones reutilizables, la logica ya validada
interactivamente en notebooks/05_xai.ipynb: SHAP global/local, reason codes
compatibles con adverse action (ECOA/Reg B) y auditoria de fairness con
Fairlearn. Ver docs/informe-final.md seccion 5 para la justificacion e
interpretacion de cada paso, incluido el hallazgo de amplificacion de
disparidad en genero y edad.

Los contrafacticos (DiCE) quedan solo en el notebook: requieren imputar los
NaN de la fila, y el manejo nativo de NaN es parte central de como XGBoost
calcula el riesgo en este dataset, lo que invalida el contrafactico para
la poblacion con historial incompleto (ver notebook seccion 6).
"""

import numpy as np
import pandas as pd
import shap
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equal_opportunity_difference,
    selection_rate,
)
from sklearn.metrics import recall_score

REASON_TEXT = {
    "EXT_SOURCE_1": "Score de riesgo externo (fuente 1) por debajo del promedio.",
    "EXT_SOURCE_2": "Score de riesgo externo (fuente 2) por debajo del promedio.",
    "EXT_SOURCE_3": "Score de riesgo externo (fuente 3) por debajo del promedio.",
    "prev_annuity_mean": "Cuotas promedio de creditos previos elevadas en relacion al perfil.",
    "credit_to_goods": "Relacion entre el monto solicitado y el valor del bien financiado elevada.",
    "prev_credit_mean": "Monto promedio de creditos previos elevado.",
    "DAYS_EMPLOYED": "Antiguedad laboral corta o irregular.",
    "bureau_debt_mean": "Deuda promedio reportada en burós de credito externos elevada.",
    "AMT_ANNUITY": "Cuota mensual solicitada elevada en relacion al perfil.",
    "pos_months_count": "Historial corto de creditos de consumo (point-of-sale) previos.",
    "OWN_CAR_AGE": "Antiguedad del vehiculo declarado atipica para el perfil.",
    "inst_delay_max": "Atrasos maximos detectados en el pago de cuotas de creditos previos.",
    "AMT_GOODS_PRICE": "Valor del bien a financiar elevado en relacion al perfil.",
    "inst_shortfall_mean": "Pagos parciales (por debajo de lo debido) en creditos previos.",
    "prev_days_decision_mean": "Patron de decisiones sobre solicitudes de credito previas.",
    "NAME_EDUCATION_TYPE_Higher education": "Nivel educativo declarado.",
    "DAYS_ID_PUBLISH": "Antiguedad del documento de identidad declarado.",
    "NAME_FAMILY_STATUS_Married": "Estado civil declarado.",
}

# Atributos protegidos/proxy: nunca se muestran como reason code, aunque el
# modelo los use internamente (ver auditoria de fairness). Mostrarle a un
# solicitante "genero" o "edad" como motivo de rechazo violaria ECOA/Reg B.
PROTECTED_FEATURES = {"CODE_GENDER_M", "CODE_GENDER_F", "DAYS_BIRTH"}


def compute_shap(model, X: pd.DataFrame):
    """TreeExplainer (exacto para modelos de arboles) sobre X."""
    explainer = shap.TreeExplainer(model)
    return explainer, explainer.shap_values(X)


def mean_abs_shap(shap_values: np.ndarray, feature_names) -> pd.DataFrame:
    """Ranking de importancia global por magnitud promedio de SHAP."""
    return pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)


def reason_codes(shap_row: np.ndarray, feature_names, top_n: int = 4) -> list:
    """Top-N razones de rechazo (ECOA/Reg B): solo contribuciones que empujan
    hacia mayor riesgo, nunca atributos protegidos, traducidas a texto legible.
    """
    contributions = pd.Series(shap_row, index=feature_names)
    contributions = contributions.drop(index=[f for f in PROTECTED_FEATURES if f in contributions.index])
    top = contributions[contributions > 0].sort_values(ascending=False).head(top_n)
    return [REASON_TEXT.get(feat, f"Valor de '{feat}' desfavorable para el perfil.") for feat in top.index]


def fairness_report(y_true, y_pred, sensitive_features) -> dict:
    """Statistical parity difference, disparate impact (ratio) y equal
    opportunity difference, dentro del rango de referencia
    [-0.1, 0.1] para las diferencias (convencion AIF360).
    """
    mf = MetricFrame(
        metrics={"selection_rate": selection_rate, "recall": recall_score},
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features,
    )
    return {
        "selection_rate_by_group": mf.by_group["selection_rate"].to_dict(),
        "recall_by_group": mf.by_group["recall"].to_dict(),
        "statistical_parity_difference": demographic_parity_difference(
            y_true, y_pred, sensitive_features=sensitive_features
        ),
        "disparate_impact": demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_features),
        "equal_opportunity_difference": equal_opportunity_difference(
            y_true, y_pred, sensitive_features=sensitive_features
        ),
    }

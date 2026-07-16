"""Tarea 6 (prd.md S5): dashboard de visualizacion e informe ejecutivo para
CrediXAI. Cubre RF-7 (segmentacion, metricas globales, fairness y detalle
por solicitud) sobre el modelo XGBoost final de la Tarea 4, la segmentacion
de la Tarea 3 y la explicabilidad/fairness de la Tarea 5.

Requiere haber corrido antes scripts/02_features.py.

Uso:
    uv run streamlit run app/dashboard.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from credixai.dashboard import (
    application_reason_codes,
    compute_fairness,
    compute_segments,
    load_features,
    sample_for_shap,
    segment_profile,
    train_full_model,
)

st.set_page_config(page_title="CrediXAI - Dashboard", layout="wide")


@st.cache_resource(show_spinner="Entrenando modelo final (una sola vez por sesion)...")
def get_bundle():
    features = load_features()
    return features, train_full_model(features)


@st.cache_resource(show_spinner="Calculando segmentacion...")
def get_segments(_features, _train_full):
    train_segments = compute_segments(_features, _train_full)
    return train_segments, segment_profile(train_segments)


@st.cache_resource(show_spinner="Calculando SHAP sobre una muestra...")
def get_shap_bundle(_bundle):
    return sample_for_shap(_bundle)


features, bundle = get_bundle()
train_segments, profile = get_segments(features, bundle["train_full"])
shap_bundle = get_shap_bundle(bundle)

st.title("CrediXAI - Dashboard de scoring crediticio explicable")

tab_resumen, tab_segmentos, tab_fairness, tab_detalle = st.tabs(
    ["Resumen ejecutivo", "Segmentacion", "Fairness", "Detalle por solicitud"]
)

with tab_resumen:
    st.subheader("Metricas del modelo (holdout 20%)")
    m = bundle["holdout_metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{m['roc_auc']:.4f}")
    c2.metric("PR-AUC", f"{m['pr_auc']:.4f}")
    c3.metric("KS", f"{m['ks']:.4f}")
    c4.metric("Brier", f"{m['brier']:.4f}")

    st.subheader("Poblacion")
    n_train = int(bundle["y"].shape[0])
    tasa_default = float(bundle["y"].mean())
    c1, c2, c3 = st.columns(3)
    c1.metric("Solicitudes (train)", f"{n_train:,}")
    c2.metric("Tasa de default real", f"{tasa_default:.2%}")
    c3.metric("Umbral de decision", f"{bundle['threshold']:.4f}")

    st.caption(
        "El umbral de decision se fija como el percentil (1 - tasa de default real) "
        "de la probabilidad predicha, de forma que el modelo marca 'alto riesgo' a la "
        "misma proporcion de solicitudes que el default real observado."
    )

with tab_segmentos:
    st.subheader("Perfil de los 5 segmentos (K-Means, train)")
    display_profile = profile.copy()
    display_profile["tasa_default"] = display_profile["tasa_default"].map("{:.2%}".format)
    display_profile["pct_poblacion"] = display_profile["pct_poblacion"].map("{:.2%}".format)
    display_profile["ingreso_promedio"] = display_profile["ingreso_promedio"].map("{:,.0f}".format)
    display_profile["credito_promedio"] = display_profile["credito_promedio"].map("{:,.0f}".format)
    st.dataframe(display_profile, width="stretch")

    st.subheader("Tasa de default por segmento")
    st.bar_chart(profile.set_index("cluster")["tasa_default"])

with tab_fairness:
    fairness = compute_fairness(bundle)

    st.subheader("Auditoria de fairness por genero")
    g = fairness["gender"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Statistical parity difference", f"{g['statistical_parity_difference']:.4f}")
    c2.metric("Disparate impact", f"{g['disparate_impact']:.4f}")
    c3.metric("Equal opportunity difference", f"{g['equal_opportunity_difference']:.4f}")
    st.caption("Rango de referencia (convencion AIF360) para las diferencias: [-0.1, 0.1].")

    gender_compare = pd.DataFrame({
        "tasa_default_real": fairness["real_rate_by_gender"],
        "tasa_alto_riesgo_predicha": g["selection_rate_by_group"],
    })
    gender_compare.index = gender_compare.index.map({0: "Mujer", 1: "Hombre"})
    st.dataframe(gender_compare.style.format("{:.2%}"), width="stretch")

    st.subheader("Auditoria de fairness por grupo etario")
    a = fairness["age"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Statistical parity difference", f"{a['statistical_parity_difference']:.4f}")
    c2.metric("Disparate impact", f"{a['disparate_impact']:.4f}")
    c3.metric("Equal opportunity difference", f"{a['equal_opportunity_difference']:.4f}")

    age_compare = pd.DataFrame({
        "tasa_default_real": fairness["real_rate_by_age"],
        "tasa_alto_riesgo_predicha": a["selection_rate_by_group"],
    })
    st.dataframe(age_compare.style.format("{:.2%}"), width="stretch")

    st.info(
        "El modelo no solo refleja la disparidad real de default por genero y edad: "
        "la amplifica. La brecha predicha es aproximadamente el doble de la brecha real "
        "en ambos casos. Ver docs/informe-final.md seccion 5.4 para el detalle."
    )

with tab_detalle:
    st.subheader("Detalle por solicitud")
    sk_ids = shap_bundle["sk_ids"]
    proba_sample = shap_bundle["proba_sample"]

    order = proba_sample.argsort()[::-1]
    sk_ids_by_risk = sk_ids.iloc[order].tolist()
    proba_by_sk_id = dict(zip(sk_ids, proba_sample))

    selected_sk_id = st.selectbox(
        "SK_ID_CURR (ordenado de mayor a menor riesgo)",
        sk_ids_by_risk,
        format_func=lambda sk_id: f"{sk_id}  -  proba={proba_by_sk_id[sk_id]:.4f}",
    )
    row_idx = int(sk_ids[sk_ids == selected_sk_id].index[0])

    proba = float(shap_bundle["proba_sample"][row_idx])
    decision = "Alto riesgo" if proba >= bundle["threshold"] else "Riesgo aceptable"
    c1, c2 = st.columns(2)
    c1.metric("Probabilidad de default", f"{proba:.4f}")
    c2.metric("Decision", decision)

    st.subheader("Explicacion local (SHAP)")
    explanation = shap.Explanation(
        values=shap_bundle["shap_values"][row_idx],
        base_values=shap_bundle["base_value"],
        data=shap_bundle["X_sample"].iloc[row_idx].values,
        feature_names=list(shap_bundle["X_sample"].columns),
    )
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots()
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        shap.plots.waterfall(explanation, max_display=10, show=False)
        st.pyplot(fig, facecolor=fig.get_facecolor())
    plt.close(fig)

    if decision == "Alto riesgo":
        st.subheader("Reason codes (adverse action)")
        for i, reason in enumerate(application_reason_codes(shap_bundle, row_idx), start=1):
            st.write(f"{i}. {reason}")

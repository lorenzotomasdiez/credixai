# ADR-003: XGBoost como modelo primario de scoring

## Status
Accepted

## Context
El top 5 de architectural characteristics prioriza Explainability y Auditability como drivers estructurales (`docs/architecture-characteristics.md`).
RNF-1 exige que toda decisión sea explicable localmente (SHAP) y trazable a política.
El dataset Home Credit presenta desbalanceo de clases (8.07% de default) y requiere integrar 8 tablas relacionales vía feature engineering agregado (PRD §2.2, §7.2).
La literatura de referencia (Japinye & Adedugbe, 2025, citada en PRD §2.4) reporta AUC 0.892-0.923 con XGBoost + SHAP manteniendo estabilidad de explicación (Kendall τ=0.94±0.03).
SHAP TreeExplainer opera de forma exacta y eficiente sobre modelos basados en árboles, a diferencia de modelos de caja negra (redes neuronales, ensembles heterogéneos) que requieren aproximaciones (KernelSHAP, más costosas y menos estables).
El PRD (§3.3) excluye explícitamente el uso de un LLM como scorer end-to-end.

## Decision
Se adopta **XGBoost** como modelo primario de predicción de default, con **Regresión Logística** como baseline interpretable por diseño.

## Consequences
- SHAP TreeExplainer puede aplicarse de forma exacta y eficiente, sin las aproximaciones ni el costo computacional de KernelSHAP sobre modelos de caja negra.
- Se cumple el requisito de estabilidad de explicación (Kendall τ) reportado en la literatura de referencia, necesario para que los reason codes sean confiables de cara al cliente.
- Se renuncia al techo de performance potencialmente mayor de ensembles heterogéneos o deep learning sobre datos tabulares, a cambio de explicabilidad nativa.
- La Regresión Logística como baseline permite contrastar el aporte real de XGBoost y detectar si la ganancia de performance justifica la complejidad adicional.
- Todo el pipeline de calibración (Brier score), fairness (Fairlearn/AIF360) y contrafácticos (DiCE) queda condicionado a operar sobre un modelo basado en árboles.

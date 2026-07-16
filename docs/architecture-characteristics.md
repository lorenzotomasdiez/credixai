# Architecture Characteristics — CrediXAI

**Fuente de requisitos:** `docs/informe-final.md`.
**Estado:** Borrador para discusión.

---

## 1. Characteristics extraídas

| # | Origen (PRD) | Requisito | Characteristic | Categoría |
|---|---|---|---|---|
| 1 | RNF-1 | Toda decisión debe ser explicable localmente y trazable a política | Explainability | Cross-cutting |
| 2 | §2.3, RNF-1 | Marco regulatorio EU AI Act, ECOA, BCRA sobre credit scoring | Auditability / Legal compliance | Cross-cutting |
| 3 | RNF-2 | Statistical parity, equal opportunity, disparate impact documentados | Fairness | Cross-cutting |
| 4 | RNF-3 | Seeds fijos, datos versionados, entorno containerizado | Reproducibility | Estructural |
| 5 | RNF-4 | Trazas de LLM, logs de predicción, monitoreo de drift | Observability | Operacional |
| 6 | RNF-5 | Scoring + SHAP < 2s por solicitud | Performance | Operacional |
| 7 | RNF-6 | Costo de API en rango de decenas de dólares | Cost-effectiveness (restricción, no characteristic) | — |
| 8 | RNF-7 | Guardrails contra prompt injection, validación con Pydantic | Security | Cross-cutting |
| 9 | RNF-8 | RAGAS faithfulness ≥ 0.90 | Sub-caso de Explainability/Auditability | Cross-cutting |
| 10 | §7.2 (tamaño de datos, ~688 MB, decenas de millones de filas) | Procesamiento por chunks, downcasting | Scalability (de datos, no de tráfico) | Operacional |
| 11 | §9, roadmap académico con fecha de entrega fija | Restricción de tiempo | — (restricción de proyecto) | — |
| 12 | Naturaleza académica + portfolio (doble propósito, §1) | El código lo leen un docente y un hiring manager | Learnability / Maintainability | Estructural |
| 13 | §7.5, arquitectura basada en tools/agentes | Cada componente (modelo, SHAP, RAG) se consume como tool independiente | Modularity | Estructural |
| 14 | §10, riesgo "inestabilidad de agentes/costo de tokens" | El sistema debe poder escalar de simple a complejo sin reescritura | Evolvability / Extensibility | Estructural |

---

## 2. Candidatas identificadas (sin priorizar)

| # | Characteristic | Categoría |
|---|---|---|
| 1 | Explainability | Cross-cutting |
| 2 | Auditability / Legal compliance | Cross-cutting |
| 3 | Fairness | Cross-cutting |
| 4 | Security | Cross-cutting |
| 5 | Reproducibility | Estructural |
| 6 | Observability | Operacional |
| 7 | Performance (latencia < 2s) | Operacional |
| 8 | Scalability (volumen de datos) | Operacional |
| 9 | Modularity | Estructural |
| 10 | Evolvability / Extensibility | Estructural |
| 11 | Maintainability / Learnability | Estructural |

---

## 3. Priorización — Top 5

1. **Explainability** — gobierna la elección de modelo (XGBoost interpretable con SHAP, no un ensemble opaco) y el diseño del pipeline RAG.
2. **Auditability** — MLflow, versionado de datos y trazas de Langfuse existen únicamente para esto.
3. **Fairness** — condiciona el umbral de decisión (thresholding con restricción de fairness, §7.3) y es un entregable explícito (model card).
4. **Modularity** — habilita que el núcleo académico (tareas 1-7) se entregue solo, y las capas RAG/agentes se sumen después sin reescribir nada (principio de gestión de scope, §9).
5. **Reproducibility** — condición necesaria para verificar el AUC ≥ 0.79 reportado.

**Quedan fuera del top 5** (se atienden, pero no gobiernan trade-offs de diseño): Performance, Scalability de tráfico, Security.

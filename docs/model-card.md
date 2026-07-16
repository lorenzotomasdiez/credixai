# Model card - CrediXAI (XGBoost, riesgo de default)

Formato basado en Mitchell et al. (2019), "Model Cards for Model Reporting".
Fuente de los números: `docs/informe-final.md` (secciones 4 y 5), reproducibles con `scripts/04_modeling.py` y `scripts/05_explainability.py`.

## 1. Detalles del modelo

- **Desarrollador:** Lorenzo Tomás Diez, como Práctica Profesionalizante (Tecnicatura Superior en Data Science, Teclab).
- **Fecha:** 2026.
- **Tipo de modelo:** XGBoost (gradient boosted trees), clasificación binaria.
- **Hiperparámetros clave:** `max_depth=5`, sin `scale_pos_weight` (ver justificación en sección 4.4 del informe final), 1086 árboles fijos en el modelo final (promedio de las mejores iteraciones observadas en holdout y validación cruzada).
- **Versión:** modelo único, sin historial de versiones todavía.
- **Licencia del código:** ver `README.md`.
- **Referencia:** `docs/informe-final.md` sección 4 (modelado) y sección 5 (explicabilidad y fairness).

## 2. Uso previsto

- **Uso primario:** estimar la probabilidad de que un solicitante de crédito incumpla sus pagos (default), como insumo de apoyo a la decisión de un analista humano, no como decisión automática.
- **Usuarios previstos:** analistas de crédito, risk managers y auditores/reguladores de una fintech simulada (personas definidas en `docs/informe-final.md`).
- **Fuera de alcance:** el modelo no está validado para producción real, no procesa datos personales reales, y no reemplaza la supervisión humana en la decisión final de crédito.

## 3. Datos de entrenamiento

- **Fuente:** Home Credit Default Risk (Kaggle), 8 tablas relacionales.
- **Tamaño:** 307,511 solicitudes con `TARGET` conocido, usadas para entrenamiento y validación; 48,744 adicionales sin etiqueta (partición de test de Kaggle), no usadas para medir performance.
- **Tasa de default real:** 8.07%.
- **Features:** variables de la solicitud, agregaciones de las 6 tablas relacionales (bureau, previous applications, installments, credit card balance, POS cash balance) y ratios de negocio construidos en la Tarea 2 (`docs/informe-final.md` sección 2).
- **Partición:** holdout estratificado 80/20 para decidir hiperparámetros; validación cruzada estratificada de 5 folds para confirmar el resultado; el modelo final se reentrena sobre el 100% de la partición con etiqueta.

## 4. Métricas de performance

| Métrica | Baseline (Regresión Logística) | XGBoost (holdout) | XGBoost (CV 5-fold) |
|---|---|---|---|
| ROC-AUC | 0.7637 | 0.7815 | 0.7802 ± 0.0032 |
| PR-AUC | 0.2488 | ver sección 4 del informe | - |
| KS | 0.3982 | ver sección 4 del informe | - |
| Brier | 0.1963 | 0.0660 | - |

El ROC-AUC del modelo final se ubica de forma estable en torno a 0.78 (rango entre folds: 0.7758-0.7848), una mejora de +1.6 puntos sobre el baseline, por debajo del objetivo de 0.79 planteado inicialmente para el proyecto.
Detalle completo del proceso de tuneo en `docs/informe-final.md` sección 4.4-4.5.

## 5. Explicabilidad

- **SHAP global:** `EXT_SOURCE_2`, `EXT_SOURCE_3` y `EXT_SOURCE_1` son las variables más predictivas; el ranking es estable (Kendall tau = 0.9917 ± 0.0010 sobre 30 remuestreos bootstrap).
- **SHAP local:** cada predicción individual tiene una explicación waterfall disponible en el dashboard (`app/dashboard.py`, pestaña "Detalle por solicitud").
- **Reason codes:** hasta 4 razones en lenguaje de negocio por solicitud de alto riesgo, excluyendo explícitamente atributos protegidos/proxy (`CODE_GENDER_M`, `CODE_GENDER_F`, `DAYS_BIRTH`) de la comunicación al solicitante, aunque el modelo los use internamente.
- **Contrafácticos (DiCE):** evaluados y descartados para uso en producción; incompatibilidad estructural con el manejo nativo de NaN de XGBoost para la población con historial incompleto (detalle en `docs/informe-final.md` sección 5.5).

## 6. Fairness

Auditado con tres métricas (statistical parity difference, disparate impact, equal opportunity difference), rango de referencia [-0.1, 0.1] para las diferencias (convención AIF360), regla del 80% para disparate impact.
Umbral de decisión: percentil que marca "alto riesgo" a la misma proporción de solicitudes (8.07%) que la tasa real de default.

| Grupo | Statistical parity diff | Disparate impact (ratio) | Equal opportunity diff |
|---|---|---|---|
| Género (hombre vs. mujer) | 0.0608 (dentro de rango) | 0.496 (falla regla del 80%) | 0.1315 (fuera de rango) |
| Edad (<30 vs. 60+) | 0.1282 (fuera de rango) | 0.134 (falla regla del 80%) | 0.3257 (fuera de rango) |

**Hallazgo principal:** el modelo no solo refleja la disparidad real de default entre estos grupos, la amplifica.
En género, la brecha real es de 3.14 puntos porcentuales (ratio 0.690); el modelo la traduce en una brecha predicha de 6.08 puntos (ratio 0.496), aproximadamente el doble.
En edad, la brecha real es de 6.52 puntos (ratio 0.430) contra una brecha predicha de 12.82 puntos (ratio 0.134), otra vez más del doble.
Detalle completo, incluida la lectura de por qué una sola métrica de fairness no alcanza para el caso de género, en `docs/informe-final.md` sección 5.4.

**Mitigaciones identificadas para trabajo futuro, no implementadas en esta versión:**

1. Thresholding con restricción de fairness (por ejemplo, `ThresholdOptimizer` de Fairlearn).
2. Remover o neutralizar `CODE_GENDER` y las variables más correlacionadas con edad antes de entrenar, y volver a medir si la amplificación persiste por la vía de otras features proxy.

## 7. Limitaciones conocidas

1. **Amplificación de disparidad (sección 6):** el modelo, en su forma actual, no debería usarse para una decisión de crédito real sin aplicar una de las mitigaciones de fairness identificadas.
2. **Contrafácticos no disponibles para historial incompleto:** la explicación "qué cambiar para revertir la decisión" no es fiable para la población con datos faltantes en las tablas relacionales, que es además la población de mayor riesgo (sección 5).
3. **ROC-AUC por debajo del objetivo inicial (0.78 vs. 0.79):** el tuneo manual de hiperparámetros no cerró esa brecha; probablemente requiera trabajo adicional de feature engineering o ensembles.
4. **Dataset sintético/histórico, no productivo:** Home Credit Default Risk es un dataset público de Kaggle, no datos reales de una fintech en producción; el modelo no fue validado contra drift de datos ni contra un flujo de scoring en vivo.
5. **Sin monitoreo de drift en esta versión:** la capa de observabilidad de producción (Evidently, logging de predicciones) es una extensión de portfolio no implementada en el núcleo académico.

## 8. Consideraciones éticas y regulatorias

El scoring crediticio de personas físicas es considerado de alto riesgo bajo el EU AI Act (Annex III 5(b)), con obligaciones de gestión de riesgo, examen de sesgos, documentación técnica, logging, transparencia y supervisión humana con capacidad de override.
Este proyecto usa ese marco como referencia de buena práctica, no porque opere en la Unión Europea.
Los reason codes están diseñados siguiendo el criterio de notificación de acción adversa de ECOA/Regulation B (EE.UU.) como referencia de buena práctica de la industria: nunca incluyen género ni edad como motivo comunicado, aunque el modelo los use internamente.

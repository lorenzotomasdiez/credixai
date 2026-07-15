# Informe final — CrediXAI

**Propósito:** registro acumulativo de hallazgos durante el desarrollo, insumo directo para el informe ejecutivo (Tarea 6, PRD §5) y la documentación técnica (Tarea 7).

---

## 1. EDA (`01_eda.ipynb`)

- `application_train.csv`: 307,511 filas × 122 columnas, consistente con lo documentado en `prd.md` §7.2.
- Distribución de `TARGET`: 282,686 no-default (0) vs. 24,825 default (1) — tasa de default 8.0729%, confirma el desbalanceo de clases citado en `prd.md` §2.2. Implica que la métrica primaria debe ser ROC-AUC (robusta a desbalanceo), no accuracy.
- Valores faltantes en `application_train`: 67 de 122 columnas tienen al menos un nulo. El top está dominado por variables del inmueble (`COMMONAREA_*`, `NONLIVINGAPARTMENTS_*`, `LIVINGAPARTMENTS_*`, `FLOORSMIN_*`, `YEARS_BUILD_*`, 65-70% nulos) y `OWN_CAR_AGE` (65.99%, nulo cuando el solicitante no posee auto — nulo estructural, no faltante real). Implica que la imputación no debe ser uniforme: para variables del inmueble evaluar si conviene un indicador de "dato no disponible" en vez de imputar un valor; para `OWN_CAR_AGE` el nulo es informativo por sí mismo.
- Correlación de `EXT_SOURCE_1/2/3` con `TARGET` (Pearson): EXT_SOURCE_3 = -0.1789, EXT_SOURCE_2 = -0.1605, EXT_SOURCE_1 = -0.1553. Confirma que son las tres variables más predictivas de forma individual, consistente con `prd.md` §7.2, aunque el orden difiere de lo citado en el PRD: en los datos reales, EXT_SOURCE_3 correlaciona más fuerte que EXT_SOURCE_2, no al revés. Los KDE por clase muestran que la distribución de solicitantes en default está corrida hacia valores más bajos en las tres variables.
- Anomalía en `DAYS_EMPLOYED`: el valor `365243` (~1000 años) aparece en 55,374 filas, un placeholder de "no aplica" para solicitantes sin relación de empleo activa (`NAME_INCOME_TYPE` = Pensioner: 55,352, Unemployed: 22 → suma exacta). Los valores reales de la columna, en cambio, aparecen unas pocas centenas de veces cada uno. No es un faltante al azar (MCAR) ni un outlier genuino: es un código de sistema. Implica reemplazar 365243 por NaN y crear un indicador booleano `DAYS_EMPLOYED_ANOM` en el feature engineering (Tarea 2), para no contaminar el modelo con un valor imposible y a la vez preservar la señal de "sin empleo activo".
- Edad (`DAYS_BIRTH` convertida a años, media 43, máximo 69) vs. `TARGET`: correlación de -0.0782. Existe relación real (a menor edad, mayor probabilidad de default) pero mucho más débil que la de `EXT_SOURCE_*`. Relevante para la auditoría de fairness posterior (RNF-2), dado que edad es un atributo protegido típico.
- Tasa de default por variable categórica (grupos con n suficiente, se excluyen categorías con n < 200 por no ser estadísticamente confiables): `CODE_GENDER` M=10.14% vs. F=7.00% (además, 4 filas con `XNA`, un problema de calidad de datos a resolver en Tarea 2); `NAME_EDUCATION_TYPE` desciende monótonamente con el nivel educativo (Lower secondary 10.93% → Higher education 5.36%); `NAME_INCOME_TYPE` con mayor default en Working (9.59%) y menor en Pensioner (5.39%) y State servant (5.76%). Género y educación son candidatos directos para la auditoría de fairness (statistical parity, equal opportunity) de la Tarea 5.
- Pasada liviana por las 7 tablas restantes: shapes coinciden exactamente con `prd.md` §7.2. Confirma la naturaleza relacional del dataset — `bureau.csv` (1,716,428 filas / 305,811 `SK_ID_CURR`, ~5.6 registros por solicitante), `installments_payments.csv` (13.6M filas / 339,587 `SK_ID_CURR`, ~40 cuotas por solicitante). `bureau_balance.csv` se relaciona con `application_train` de forma indirecta, vía `SK_ID_BUREAU` → `bureau.csv` → `SK_ID_CURR` (tabla de segundo grado). Implica que el feature engineering (Tarea 2) necesita una estrategia de agregación por tabla (min/max/mean/sum, conteos) antes de poder unir todo a nivel `SK_ID_CURR`.

---

## 2. Feature engineering (`02_features.py`)

_Pendiente._

---

## 3. Clustering / segmentación (`03_clustering.ipynb`)

_Pendiente._

---

## 4. Modelado supervisado

_Pendiente._

---

## 5. Explicabilidad y fairness

_Pendiente._

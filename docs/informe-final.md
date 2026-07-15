# Informe final - CrediXAI

**Propósito:** registro acumulativo de hallazgos durante el desarrollo, insumo directo para el informe ejecutivo (Tarea 6) y la documentación técnica (Tarea 7).

---

## 1. EDA (`01_eda.ipynb`)

Esta sección documenta el relevamiento exploratorio de las 8 tablas del dataset Home Credit Default Risk (Tarea 1).
Se organiza por tabla, en el mismo orden en que se exploraron: primero `application_train` (la tabla principal), luego las 6 tablas relacionales restantes en el orden de prioridad justificado más abajo (`bureau` antes que las demás por ser la única fuente de terceros), y cierra con una síntesis comparativa.
Cada hallazgo indica su implicancia directa para la Tarea 2 (feature engineering).

### 1.1 `application_train.csv`: estructura y desbalanceo de clases

`application_train.csv` tiene 307,511 filas × 122 columnas.
La distribución de `TARGET` es 282,686 no-default (0) vs. 24,825 default (1), una tasa de default de 8.0729% que confirma la hipótesis de partida de un dataset fuertemente desbalanceado.
Implica que la métrica primaria debe ser ROC-AUC (robusta a desbalanceo), no accuracy.

### 1.2 Valores faltantes en `application_train`

67 de las 122 columnas tienen al menos un nulo.
El top está dominado por variables del inmueble (`COMMONAREA_*`, `NONLIVINGAPARTMENTS_*`, `LIVINGAPARTMENTS_*`, `FLOORSMIN_*`, `YEARS_BUILD_*`, 65-70% nulos) y por `OWN_CAR_AGE` (65.99%), nulo cuando el solicitante no posee auto: un nulo estructural, no un faltante real.
Implica que la imputación no debe ser uniforme.
Para las variables del inmueble conviene evaluar si es mejor un indicador de "dato no disponible" en vez de imputar un valor; para `OWN_CAR_AGE` el nulo es informativo por sí mismo.

### 1.3 `EXT_SOURCE_1/2/3`: las variables individuales más predictivas

La correlación de Pearson de `EXT_SOURCE_1/2/3` con `TARGET` es EXT_SOURCE_3 = -0.1789, EXT_SOURCE_2 = -0.1605, EXT_SOURCE_1 = -0.1553.
La hipótesis de trabajo era que estas tres variables serían las más predictivas de forma individual, y los datos la confirman, aunque el orden relativo entre ellas resultó distinto al esperado: en los datos reales, EXT_SOURCE_3 correlaciona más fuerte que EXT_SOURCE_2, no al revés.
Los KDE por clase muestran que la distribución de solicitantes en default está corrida hacia valores más bajos en las tres variables.

### 1.4 Anomalía en `DAYS_EMPLOYED`

El valor `365243` (~1000 años) aparece en 55,374 filas, un placeholder de "no aplica" para solicitantes sin relación de empleo activa (`NAME_INCOME_TYPE` = Pensioner: 55,352, Unemployed: 22, suma exacta).
Los valores reales de la columna, en cambio, aparecen unas pocas centenas de veces cada uno.
No es un faltante al azar (MCAR) ni un outlier genuino: es un código de sistema.
Implica reemplazar 365243 por NaN y crear un indicador booleano `DAYS_EMPLOYED_ANOM` en el feature engineering (Tarea 2), para no contaminar el modelo con un valor imposible y a la vez preservar la señal de "sin empleo activo".

### 1.5 Edad y fairness

`DAYS_BIRTH` convertida a años (media 43, máximo 69) correlaciona -0.0782 con `TARGET`.
Existe relación real (a menor edad, mayor probabilidad de default) pero mucho más débil que la de `EXT_SOURCE_*`.
Es relevante para la auditoría de fairness posterior (RNF-2), dado que edad es un atributo protegido típico.

### 1.6 Tasa de default por variable categórica

Se excluyen categorías con n < 200 por no ser estadísticamente confiables.
`CODE_GENDER`: M=10.14% vs. F=7.00% (además, 4 filas con `XNA`, un problema de calidad de datos a resolver en Tarea 2).
`NAME_EDUCATION_TYPE` desciende monótonamente con el nivel educativo (Lower secondary 10.93% hasta Higher education 5.36%).
`NAME_INCOME_TYPE` tiene mayor default en Working (9.59%) y menor en Pensioner (5.39%) y State servant (5.76%).
Género y educación son candidatos directos para la auditoría de fairness (statistical parity, equal opportunity) de la Tarea 5.

### 1.7 Panorama de las 7 tablas relacionales

Una pasada liviana por las 7 tablas restantes confirma sus shapes documentadas por Kaggle para el dataset, lo que valida la naturaleza relacional del dataset: `bureau.csv` (1,716,428 filas / 305,811 `SK_ID_CURR`, ~5.6 registros por solicitante), `installments_payments.csv` (13.6M filas / 339,587 `SK_ID_CURR`, ~40 cuotas por solicitante).
`bureau_balance.csv` se relaciona con `application_train` de forma indirecta, vía `SK_ID_BUREAU` a `bureau.csv` y de ahí a `SK_ID_CURR` (tabla de segundo grado).
Implica que el feature engineering (Tarea 2) necesita una estrategia de agregación por tabla (min/max/mean/sum, conteos) antes de poder unir todo a nivel `SK_ID_CURR`.

### 1.8 `bureau.csv`

**Por qué explorarla primero.** Es la única fuente del dataset con información de crédito de **terceros** (otras entidades financieras), no autodeclarada ni generada por Home Credit, a diferencia de `application_train` (datos del solicitante) y de `previous_application`/`POS_CASH_balance`/`credit_card_balance`/`installments_payments` (todas historial interno de Home Credit).
Es también la tabla relacional con la conexión más directa a `application_train`: join de un solo salto por `SK_ID_CURR`, sin tabla intermedia como sí requiere `bureau_balance`.
Su valor como fuente de señal independiente se verifica empíricamente más abajo: `bureau_active_count` correlaciona con `TARGET` (0.0671) pese a no haber aplicado ninguna transformación.

**Estructura.** 1,716,428 filas × 17 columnas; 7 de las 17 columnas tienen al menos un nulo.
Los más altos son `AMT_ANNUITY` (71.47%) y `AMT_CREDIT_MAX_OVERDUE` (65.51%), donde el nulo probablemente refleja ausencia del dato en el buró externo más que un faltante al azar.
`DAYS_ENDDATE_FACT` (36.92% nulo) es esperable: solo se completa cuando el crédito ya cerró (`CREDIT_ACTIVE` = Closed), un nulo estructural, no un faltante real.
`AMT_CREDIT_SUM` (monto total del crédito) tiene solo 13 nulos, prácticamente completa.
Implica que la imputación de `bureau.csv` tampoco debe ser uniforme, igual que en `application_train`.

**Categorías.** `CREDIT_ACTIVE`: Closed 1,079,273 (62.9%), Active 630,607 (36.7%), Sold 6,527 (0.4%), Bad debt 21 (~0%).
La mayoría de los créditos históricos ya están cerrados, consistente con `DAYS_ENDDATE_FACT` nulo en ~37% de los casos (los activos).
`CREDIT_TYPE` está dominado por `Consumer credit` (1,251,615, 72.9%) y `Credit card` (402,195, 23.4%); el resto de los 13 tipos restantes suma menos del 4% combinado, candidatos a agrupar en una categoría "otros" en el feature engineering (Tarea 2) por bajo n.

**Agregación a nivel `SK_ID_CURR`.** Con cantidad de créditos, cantidad de créditos activos y deuda total, cruzada con `TARGET`: 44,020 de 307,511 solicitantes (14.3%) no tienen ningún registro en `bureau.csv`, un dato de cobertura relevante para la Tarea 2 (esos casos quedan en NaN tras el merge y requieren una decisión explícita, por ejemplo imputar 0 para conteos/sumas ya que "sin registro" es distinto de "cero créditos").
De las tres variables agregadas, `bureau_active_count` (cantidad de créditos activos) muestra la correlación más alta con `TARGET` (0.0671, positiva): a más créditos activos simultáneos, levemente mayor probabilidad de default.
`bureau_count` (0.0041) y `bureau_debt_sum` (0.0071) prácticamente no correlacionan en su forma cruda, lo que sugiere que estas variables necesitan transformación (ratios, normalización por ingreso) en el feature engineering para aportar señal; no alcanza con la suma/conteo directo.

### 1.9 `bureau_balance.csv`

**Hipótesis.** Si `bureau.csv` ya mostró señal con "cantidad de créditos activos" (§1.8), la hipótesis es que el detalle mes a mes de morosidad frente a otras entidades (no solo si el crédito está activo, sino si tuvo atrasos) debería ser una señal de riesgo más granular y potencialmente más fuerte, en la misma lógica que las agencias de buró de crédito usan el historial de pagos como insumo central del scoring.

**Estructura.** 27,299,925 filas × 3 columnas, sin nulos: registra el saldo mensual de cada crédito de `bureau.csv`, uno por `SK_ID_BUREAU` y mes.
`STATUS` (mora del mes, DPD) está dominado por `C` (cerrado, 50.0%), `0` (sin mora, 27.5%) y `X` (desconocido, 21.3%); los niveles de mora real (`1`=1-30 días, `2`=31-60, `3`=61-90, `4`=91-120, `5`=120+/dado de baja) suman apenas 1.2% combinado, señal de mora rara pero presente.
`MONTHS_BALANCE` va de 0 (mes más reciente) a -96 (8 años atrás), confirmando que es una serie temporal por crédito.
Implica que, para agregar esta tabla a nivel `SK_ID_CURR`, hace falta un join en dos saltos (`bureau_balance` a `bureau` por `SK_ID_BUREAU`, y de ahí a `application` por `SK_ID_CURR`) y una estrategia de agregación temporal (por ejemplo peor `STATUS` histórico, cantidad de meses en mora, tendencia reciente) antes de poder cruzar con `TARGET`.

**Resultado.** Con el join de dos saltos, agregando "algún mes en mora real" (`STATUS` 1-5) por crédito y sumando por `SK_ID_CURR`, cruzado con `TARGET`: 215,280 de 307,511 solicitantes (70.0%) no tienen ningún registro en `bureau_balance`, una cobertura mucho más baja que la de `bureau.csv` (14.3% sin registro), porque no todos los créditos reportados al buró tienen historial de saldo mensual.
La variable resultante (`bureau_credits_ever_dpd`) correlaciona con `TARGET` en 0.0406, positiva pero más débil que `bureau_active_count` (0.0671): señal real, aunque diluida por la baja cobertura.
Implica que en la Tarea 2 el porcentaje de faltantes debe tratarse como información en sí (menor cobertura de buró puede ser indicador de perfil crediticio más joven o menos formal), no solo como un valor a imputar.

### 1.10 `previous_application.csv`

**Hipótesis.** A diferencia de `bureau.csv` (terceros), esta es la primera tabla de historial **interno** de Home Credit.
La hipótesis es que el propio historial del solicitante con el mismo prestamista (cuántas veces pidió crédito antes, cuántas veces fue rechazado, tasa de aprobación) es un predictor directo de riesgo, ya que refleja decisiones de riesgo que Home Credit mismo tomó en el pasado sobre esa persona.

**Estructura.** 1,670,214 filas × 37 columnas, una fila por cada solicitud previa que la persona hizo a la propia empresa.
`RATE_INTEREST_PRIMARY` y `RATE_INTEREST_PRIVILEGED` están nulas en 99.64% de los casos, prácticamente inutilizables y candidatas a descartar directamente en la Tarea 2.
`AMT_DOWN_PAYMENT`/`RATE_DOWN_PAYMENT` (53.64% nulo) y el grupo `DAYS_FIRST_DRAWING`/`DAYS_FIRST_DUE`/`DAYS_LAST_DUE_1ST_VERSION`/`DAYS_LAST_DUE`/`DAYS_TERMINATION` (40.30% nulo, mismo porcentaje exacto en las 5 columnas) probablemente son nulas cuando la solicitud previa no se aprobó, ya que sin aprobación no hay fechas de desembolso o vencimiento; hipótesis a confirmar cruzando con `NAME_CONTRACT_STATUS` en la Tarea 2.
`NAME_CONTRACT_STATUS`: Approved 1,036,781 (62.1%), Canceled 316,319 (18.9%), Refused 290,678 (17.4%), Unused offer 26,436 (1.6%).

**Resultado.** Agregando cantidad de solicitudes previas, si alguna vez fue rechazada y tasa de aprobación histórica, cruzado con `TARGET`: solo 16,454 de 307,511 solicitantes (5.4%) no tienen ninguna solicitud previa, una cobertura mucho mejor que `bureau.csv` (14.3% sin registro) y que `bureau_balance.csv` (70.0%), consistente con ser historial de la propia empresa (todo cliente activo tiende a tener al menos una solicitud previa).
`prev_approval_rate` (tasa de aprobación histórica) es la variable con mayor correlación de las tres (-0.0635, negativa: a mayor tasa de aprobación pasada, menor probabilidad de default actual); `prev_ever_refused` (0.0563, positiva) confirma la misma dirección desde el ángulo opuesto.
`prev_count` (0.0198) aporta poca señal por sí solo.
Esta es, hasta ahora, la señal individual más fuerte entre las tablas relacionales exploradas, y confirma la lógica intuitiva de que el comportamiento previo del solicitante frente al mismo prestamista predice el riesgo actual.

### 1.11 `POS_CASH_balance.csv`

**Hipótesis.** A diferencia de `previous_application` (una fila por solicitud, foto estática del resultado), esta tabla registra el saldo **mes a mes** de cada crédito de punto de venta/efectivo ya otorgado por Home Credit.
La hipótesis es que la morosidad mensual histórica (`SK_DPD`, días de mora) es un predictor de comportamiento de pago más directo que el resultado de la solicitud: alguien que pagó en término todos los meses de un crédito previo debería tener menor riesgo actual que alguien con historial de atrasos, incluso si ambas solicitudes previas terminaron "Approved".
Se relaciona con `application_train` en dos saltos posibles (directo por `SK_ID_CURR`, o vía `SK_ID_PREV` a `previous_application`); acá se usa el directo por ser de un solo salto.

**Estructura.** 10,001,358 filas × 8 columnas, prácticamente completa: solo `CNT_INSTALMENT` y `CNT_INSTALMENT_FUTURE` tienen nulos, y apenas 0.26% cada una.
`NAME_CONTRACT_STATUS` está dominado por `Active` (91.5%) y `Completed` (7.4%); el resto de los 7 estados restantes suma menos del 1.1% combinado.
`SK_DPD` (días de mora del mes) tiene mediana 0 y media 11.6, fuertemente asimétrica a la derecha (máximo 4,231 días): la gran mayoría de los meses no registra mora, con una cola larga de casos de mora severa.

**Resultado.** Agregando mora máxima histórica y si alguna vez tuvo mora, cruzado con `TARGET`: 18,067 de 307,511 solicitantes (5.9%) sin registro, cobertura similar a `previous_application.csv`.
Contra la hipótesis planteada, la señal resultó más débil de lo esperado: `pos_ever_dpd` correlaciona apenas 0.0334 y `pos_dpd_max` 0.0048 (casi nula), más débil que `bureau_active_count` (0.0671) y bastante más débil que `prev_approval_rate` (-0.0635).
Implica que la morosidad mensual granular de POS/efectivo, al menos en su forma cruda (máximo/alguna vez), no es tan predictiva como se esperaba, probablemente porque la enorme mayoría de meses están en `Active` sin mora y la señal está diluida por el volumen de meses sin evento.
Puede necesitar una transformación más fina en la Tarea 2 (por ejemplo proporción de meses en mora, no solo el máximo).

### 1.12 `credit_card_balance.csv`

**Estructura.** 3,840,312 filas × 23 columnas.
Los nulos se concentran en las columnas de "drawings" (`AMT_DRAWINGS_ATM/OTHER/POS_CURRENT` y sus conteos, ~19.52% cada una) y en `AMT_PAYMENT_CURRENT` (20.00%), probablemente nulas en meses sin actividad de consumo o pago, a confirmar en la Tarea 2.
`AMT_BALANCE` tiene un mínimo negativo (-420,250), inusual para un saldo de tarjeta: podría representar sobrepago o crédito a favor del cliente, y requiere validación antes de usarse en un ratio.
`AMT_CREDIT_LIMIT_ACTUAL` tiene un mínimo de 0, lo que impide calcular directamente el ratio de utilización (`AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL`) sin manejar división por cero en la Tarea 2.

**Resultado.** Agregando el ratio de utilización promedio (`AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL`) cruzado con `TARGET`: 221,475 de 307,511 solicitantes (72.0%) no tienen ningún registro (no tuvieron tarjeta de crédito con Home Credit), la cobertura más baja de todas las tablas relacionales exploradas.
Pese a esto, `cc_utilization_mean` correlaciona 0.1356 con `TARGET`, la señal individual más fuerte encontrada entre las tablas relacionales (por encima de `prev_approval_rate` -0.0635 y `bureau_active_count` 0.0671), y se acerca al orden de magnitud de `EXT_SOURCE_2/3` (-0.16/-0.18).
Confirma la hipótesis planteada: el uso relativo del límite de crédito es más predictivo que la mora cruda.
Implica que, en la Tarea 2, esta variable es candidata fuerte a incluirse en el modelo pese a su baja cobertura, y que la ausencia de tarjeta de crédito en sí misma (72% de los casos) debería probarse también como feature booleana, no solo imputarse.

### 1.13 `installments_payments.csv`

**Estructura.** 13,605,401 filas × 8 columnas, prácticamente completa: solo `DAYS_ENTRY_PAYMENT` y `AMT_PAYMENT` tienen nulos, y apenas 0.02% cada una (probablemente cuotas aún no pagadas al momento del corte de datos).
Es la tabla más granular del dataset relacional: una fila por cuota programada, con el monto/fecha esperado (`AMT_INSTALMENT`, `DAYS_INSTALMENT`) y lo efectivamente pagado (`AMT_PAYMENT`, `DAYS_ENTRY_PAYMENT`).

**Resultado.** Agregando atraso promedio en días y déficit de pago promedio, cruzado con `TARGET`: 15,876 de 307,511 solicitantes (5.2%) sin registro, buena cobertura.
Contra la hipótesis de que esta sería la señal más fuerte por ser la más granular, el resultado fue el más débil de todas las tablas relacionales: `inst_shortfall_mean` correlaciona 0.0293 e `inst_delay_mean` 0.0209.
La hipótesis no se confirmó en su forma cruda: promediar atraso/déficit sobre todas las cuotas históricas (algunas de hace años) diluye la señal reciente, y la mayoría de las cuotas se pagan a tiempo y por el monto correcto (mediana de atraso y déficit esperable cerca de 0), por lo que el promedio no distingue bien a quienes tuvieron algunos atrasos puntuales.
Implica que en la Tarea 2 conviene probar agregaciones alternativas (máximo atraso, cantidad de cuotas con atraso mayor a X días, ventana temporal reciente) en lugar del promedio simple.

### 1.14 Síntesis de la Tarea 1

Ranking de correlación absoluta con `TARGET` de la variable más fuerte de cada tabla, cruda y sin transformar:

| Tabla | Variable | Correlación |
|---|---|---|
| `application_train` | `EXT_SOURCE_3` | -0.1789 |
| `application_train` | `EXT_SOURCE_2` | -0.1605 |
| `application_train` | `EXT_SOURCE_1` | -0.1553 |
| `credit_card_balance` | `cc_utilization_mean` | +0.1356 |
| `bureau` | `bureau_active_count` | +0.0671 |
| `previous_application` | `prev_approval_rate` | -0.0635 |
| `previous_application` | `prev_ever_refused` | +0.0563 |
| `bureau_balance` | `bureau_credits_ever_dpd` | +0.0406 |
| `POS_CASH_balance` | `pos_ever_dpd` | +0.0334 |
| `installments_payments` | `inst_shortfall_mean` | +0.0293 |
| `installments_payments` | `inst_delay_mean` | +0.0209 |

Ninguna variable relacional individual se acerca a `EXT_SOURCE_*`, pero `cc_utilization_mean` es la más cercana por un margen considerable.
Esto no descarta a las tablas con menor señal individual: la Tarea 2 combinará múltiples variables agregadas por tabla (no solo una por tabla como en este relevamiento) y el modelo (XGBoost) puede capturar interacciones no lineales que la correlación de Pearson no detecta.

---

## 2. Feature engineering (`02_features.ipynb` → a formalizar en `02_features.py`)

Esta sección documenta la construcción de la tabla de features a nivel `SK_ID_CURR` (Tarea 2), apoyada directamente en los hallazgos de la Tarea 1.
No repite la justificación de por qué se mira cada tabla relacional, ya registrada en la sección 1; acá el foco es agregación y construcción de features.

### 2.1 Unificación de `application_train`/`application_test` y limpieza base

Concatenamos `application_train` (307,511 filas) y `application_test` (48,744 filas) antes de generar cualquier feature (356,255 filas totales, columna `IS_TRAIN` para separarlos después).
Todas las agregaciones y ratios dependen solo de `SK_ID_CURR` y de las tablas relacionales, nunca de `TARGET`; calcularlos por separado arriesgaría inconsistencias de encoding entre train y test (train/test skew).
Corrección de la anomalía de `DAYS_EMPLOYED` (365243 → NaN + flag `DAYS_EMPLOYED_ANOM`) aplicada sobre el dataset concatenado: 64,648 de 356,255 filas (18.1%) marcadas, proporción consistente con el 18.0% (55,374/307,511) ya visto en la Tarea 1 sobre train solo.

### 2.2 Ratios de negocio

La hipótesis de trabajo era que credit-to-income, annuity-to-income y credit-to-goods serían ratios de negocio relevantes en scoring crediticio.
Calculamos `credit_to_income` (AMT_CREDIT/AMT_INCOME_TOTAL), `annuity_to_income` (AMT_ANNUITY/AMT_INCOME_TOTAL) y `credit_to_goods` (AMT_CREDIT/AMT_GOODS_PRICE), con división por cero manejada (`replace(0, np.nan)`).
Cruzados con `TARGET` (solo en la porción train): `credit_to_goods` es el más fuerte de los tres (0.0694, positivo, pedir más crédito relativo al valor del bien financiado se asocia a mayor riesgo), `annuity_to_income` es débil (0.0143) y `credit_to_income` prácticamente no aporta señal individual (-0.0077, incluso de signo contraintuitivo).
Implica que, igual que con `bureau_count`/`bureau_debt_sum` en la Tarea 1, estos ratios individuales no van a ser features fuertes por sí solos; su valor probablemente esté en interacciones que el modelo (XGBoost) capture, no en la correlación lineal simple.

### 2.3 Agregación de `bureau.csv`

En la Tarea 1 se agregaron solo 3 variables puntuales (`bureau_count`, `bureau_active_count`, `bureau_debt_sum`) para verificar señal.
Acá generalizamos a 9 variables por `SK_ID_CURR`: conteos (`bureau_count`, `bureau_active_count`), estadísticas de monto (`bureau_credit_sum_mean/max`, `bureau_debt_sum/mean`, `bureau_overdue_sum/max`) y `bureau_annuity_mean`.
Resultado sobre 305,811 `SK_ID_CURR` con registro (consistente con la Tarea 1: 44,020 de 307,511 sin registro, 14.3%): `bureau_debt_sum` tiene un mínimo negativo (-6.98M), lo que indica que `AMT_CREDIT_SUM_DEBT` puede tener valores negativos a nivel fila (deuda "a favor" del solicitante o error de reporte del buró), a validar antes de usarlo como feature directa.
`bureau_annuity_mean` solo tiene 118,224 valores no nulos de 305,811 (38.7%), la cobertura más baja de las 9 variables, consistente con el 71.47% de nulos en `AMT_ANNUITY` ya visto en la Tarea 1 a nivel fila.
El resto de las variables tiene distribuciones muy asimétricas (medias muy por encima de las medianas en `bureau_credit_sum_mean`, `bureau_debt_sum`, `bureau_overdue_sum`), esperable en variables de monto agregadas por suma/máximo; candidatas a transformación logarítmica en el preprocesamiento antes de modelar.

Cruzando las 9 variables con `TARGET` (solo train): `bureau_active_count` se mantiene como la más fuerte (0.0671, igual que en la Tarea 1), seguida de `bureau_overdue_sum` (0.0133) y `bureau_overdue_max` (0.0106), ambas débiles pero de signo esperado (más sobregiro histórico, más riesgo).
Notablemente, `bureau_credit_sum_mean` (-0.0200) y `bureau_credit_sum_max` (-0.0197) correlacionan negativo: montos de crédito históricos más altos se asocian a *menor* probabilidad de default, contraintuitivo a primera vista pero consistente con una lectura de que acceder a montos mayores en el buró externo es en sí una señal de mejor perfil crediticio previo.
`bureau_annuity_mean` y `bureau_debt_mean` no aportan señal individual (< 0.002 en valor absoluto).
Ninguna de las 9 variables individuales supera a `bureau_active_count`, lo que confirma que la señal de esta tabla está concentrada en el conteo de créditos activos, no en los montos agregados en su forma cruda.

### 2.4 Agregación de `bureau_balance.csv`

Igual que en la Tarea 1, requiere join de dos saltos (`bureau_balance` → `bureau` por `SK_ID_BUREAU` → `SK_ID_CURR`).
En vez de un único flag "algún mes en mora" (Tarea 1), generalizamos a 4 variables por `SK_ID_CURR`: `bb_credits_with_history` (cantidad de créditos con historial de saldo), `bb_months_dpd_sum` (meses en mora acumulados), `bb_worst_dpd_level_max` (peor nivel de mora alcanzado, 0-5) y `bb_months_count_mean` (duración promedio del historial mensual por crédito, proxy de antigüedad).
134,542 solicitantes tienen historial en `bureau_balance`, consistente con el ~30% de cobertura ya visto en la Tarea 1 (215,280 de 307,511 sin registro, 70.0%).

Cruzando con `TARGET` (solo train): `bb_months_count_mean` es, inesperadamente, la variable más fuerte de toda la tabla (-0.0802, negativa), más fuerte incluso que `bureau_active_count` (0.0671) de `bureau.csv`.
La dirección tiene sentido: a mayor duración promedio del historial mensual de un crédito, menor probabilidad de default, consistente con la lógica de que un historial crediticio más largo y sostenido es señal de perfil más establecido/confiable, en línea con por qué las fintechs de crédito al consumo en Argentina enfrentan mayor incertidumbre con solicitantes "thin file" (sin o con poco historial crediticio).
`bb_worst_dpd_level_max` (0.0360) y `bb_months_dpd_sum` (0.0248) confirman la dirección esperada (más mora, más riesgo) pero son más débiles.
`bb_credits_with_history` (0.0061) aporta poca señal individual.
Implica que la Tarea 2 debería priorizar `bb_months_count_mean` (o una variante, como antigüedad del crédito más antiguo) como feature candidata fuerte, algo que no se había explorado en la Tarea 1.

### 2.5 Agregación de `previous_application.csv`

Generalizamos las 3 variables de la Tarea 1 (`prev_count`, `prev_ever_refused`, `prev_approval_rate`) sumando estadísticas de monto (`prev_credit_mean`, `prev_annuity_mean`) y `prev_days_decision_mean` (promedio de `DAYS_DECISION`, días negativos desde la solicitud actual, proxy de qué tan reciente fue la actividad previa).
338,857 `SK_ID_CURR` con al menos una solicitud previa, consistente con la cobertura ya vista en la Tarea 1.

Cruzando con `TARGET` (solo train): `prev_approval_rate` se mantiene como la más fuerte (-0.0635, igual que en la Tarea 1).
`prev_days_decision_mean` (0.0469) es un hallazgo nuevo: como `DAYS_DECISION` es negativo (más cerca de 0 = más reciente), la correlación positiva implica que tener actividad de solicitud más reciente con Home Credit se asocia a mayor riesgo actual, consistente con la lectura de que múltiples solicitudes recientes pueden reflejar necesidad de crédito más urgente o inestabilidad financiera.
`prev_ever_refused` (0.0563) y `prev_annuity_mean` (-0.0349) confirman direcciones ya esperadas.
`prev_credit_mean` (-0.0161) y `prev_count` (0.0198) aportan poca señal individual.
Ningún agregado nuevo supera a `prev_approval_rate`, que sigue siendo la variable más fuerte de esta tabla.

### 2.6 Agregación de `POS_CASH_balance.csv`

En la Tarea 1, `pos_ever_dpd`/`pos_dpd_max` (máximo/alguna vez en mora) dieron señal débil (0.0334/0.0048), contra la hipótesis planteada.
Probamos la alternativa propuesta ahí mismo: `pos_dpd_rate` (proporción de meses en mora, en vez de máximo/alguna vez), más `pos_credits_count` (créditos POS/efectivo distintos) y `pos_months_count` (meses de historial totales).
337,252 `SK_ID_CURR` con registro, consistente con la Tarea 1.

Cruzando con `TARGET` (solo train): `pos_dpd_rate` (0.0306) no mejora sustancialmente a `pos_ever_dpd` (0.0334) de la Tarea 1, lo que confirma que la mora mensual de POS/efectivo, en cualquiera de sus formas simples, aporta señal débil y consistente.
El hallazgo nuevo es `pos_credits_count` (-0.0405), más fuerte que cualquier variable de mora de esta tabla: a más créditos POS/efectivo distintos con Home Credit, menor probabilidad de default, en la misma línea que `bb_months_count_mean` (§2.4) y `prev_approval_rate` (§2.5): historial de relación más extenso y sostenido con el prestamista se asocia a menor riesgo.
`pos_months_count` (-0.0356) va en la misma dirección, esperable por estar correlacionado con `pos_credits_count`.
Confirma un patrón que se repite en varias tablas internas de Home Credit: variables de "cantidad/antigüedad de relación" superan a las variables de mora cruda en señal individual.

### 2.7 Agregación de `credit_card_balance.csv`

En la Tarea 1, `cc_utilization_mean` fue la señal relacional más fuerte de todo el dataset (0.1356). Generalizamos sumando `cc_utilization_max`, `cc_dpd_rate` (proporción de meses en mora) y `cc_cards_count` (tarjetas distintas con Home Credit).

**Hipótesis para agregar el máximo de utilización, no solo el promedio.** El promedio mide uso sostenido del límite a lo largo de toda la relación con la tarjeta, pero puede diluir un pico puntual de estrés severo (alguien que usó el 95% del límite un par de meses por una emergencia y después volvió a un uso bajo tendría un promedio moderado).
El máximo captura ese pico específico, independientemente de cuánto duró.
Son señales de naturaleza distinta: uso sostenido alto (mean) sugiere dependencia estructural del crédito disponible, mientras que un pico aislado (max) sugiere un evento puntual de estrés.
No había forma de saber a priori cuál de las dos correlaciona más fuerte con `TARGET` sin probar ambas.

Cruzando con `TARGET` (solo train): `cc_utilization_mean` se confirma exactamente en 0.1356, igual que en la Tarea 1.
`cc_utilization_max` (0.0970) es más débil que el promedio, lo que indica que la señal de riesgo está más en el uso *sostenido* del límite (utilización promedio alta y constante) que en picos puntuales de utilización.
A diferencia de `bureau_balance`, `previous_application` y `POS_CASH_balance` (§2.4-2.6), acá el patrón de "cantidad/antigüedad de relación" se rompe: `cc_cards_count` (0.0044) y `cc_dpd_rate` (0.0018) no aportan señal individual prácticamente ninguna.
Implica que en `credit_card_balance` la señal es específica del comportamiento de uso (utilización), no de la cantidad de tarjetas ni de la mora, y confirma a `cc_utilization_mean` como la feature relacional individual más importante para la Tarea 4 (modelado).

### 2.8 Agregación de `installments_payments.csv`

En la Tarea 1, promediar atraso/déficit sobre todas las cuotas históricas dio la señal más débil del dataset relacional (0.0293/0.0209), contra la hipótesis de que sería la más fuerte por ser la más granular.
La hipótesis de por qué el promedio fallaba era que diluye la señal reciente y que la mayoría de las cuotas se pagan a tiempo, por lo que un promedio no distingue bien a quienes tuvieron algunos atrasos puntuales.
Probamos dos alternativas propuestas ahí mismo: `inst_delay_max` (atraso máximo, en vez de promedio) e `inst_late_rate` (proporción de cuotas con atraso mayor a 5 días, que sí captura "algunos atrasos puntuales" sin diluirlos en el promedio de cuotas pagadas a tiempo).
339,587 `SK_ID_CURR` con registro, consistente con la Tarea 1.

Cruzando con `TARGET` (solo train): `inst_late_rate` (0.0625) confirma la hipótesis y es, por lejos, la variable más fuerte de esta tabla, tres veces más que `inst_delay_mean` (0.0209) de la Tarea 1 y superando también a `inst_shortfall_mean` (0.0293).
`inst_delay_max` (0.0047), en cambio, resultó casi nula: el atraso máximo histórico de una sola cuota no es predictivo, probablemente porque un atraso aislado (incluso severo) es menos informativo que un patrón repetido de atrasos moderados.
Confirma que la forma de agregar importa más que la variable subyacente: la misma columna base (`payment_delay`) pasó de ser la señal más débil del dataset relacional (Tarea 1, como promedio) a una de las más fuertes (como tasa de eventos), superando incluso a `bureau_active_count` (0.0671) y acercándose a `prev_approval_rate` (-0.0635).

### 2.9 Merge final de la tabla de features

Unimos `application_train`+`application_test` (con ratios de negocio) con los 6 agregados relacionales (31 columnas nuevas en total) mediante `merge(..., how="left")` sobre `SK_ID_CURR`, para no perder solicitantes sin registro en alguna tabla.
Verificamos con asserts que `SK_ID_CURR` sigue siendo único y que la cantidad de filas no cambió (356,255), lo que descarta duplicación por el merge (riesgo típico cuando alguna tabla de la izquierda del join tiene múltiplas filas por clave).
Resultado: 356,255 filas × 158 columnas (127 de `application_*` + ratios + 31 relacionales).

El `%` de nulos de las 31 columnas nuevas es consistente con la cobertura ya documentada por tabla en la Tarea 1: `credit_card_balance` sigue siendo la de menor cobertura (~71% nulo, la falta de tarjeta de crédito con Home Credit), seguida de `bureau_balance` (~62% nulo, coherente con el 70% sin registro visto sobre train en la Tarea 1), `bureau` (~14% nulo), `previous_application`/`POS_CASH_balance`/`installments_payments` (~5% nulo cada una, las de mejor cobertura por ser historial directo con Home Credit).
Confirma que la estrategia de imputación de la Tarea 2 va a necesitar tratar el nulo como información (especialmente en `credit_card_balance` y `bureau_balance`, con más de 60% de solicitantes sin registro) y no solo rellenar con un valor neutro.

### 2.10 Chequeo intermedio antes de imputación/encoding

Antes de resolver imputación y encoding (§2.11-2.12), verificamos con un assert que `SK_ID_CURR` seguía siendo único tras el merge de las 6 tablas relacionales (356,255 filas × 158 columnas: 127 de `application_*` + ratios + 31 relacionales).
El chequeo de fuga de datos queda cubierto por construcción: todas las variables agregadas provienen de columnas que ya existían en las tablas fuente antes de la fecha de la solicitud actual (las columnas `DAYS_*` del dataset son siempre negativas o relativas a la solicitud), y ninguna agregación usó `TARGET` como insumo.
Se agregó `pyarrow` como dependencia del proyecto para poder escribir Parquet.

### 2.11 Estrategia de imputación

**Decisión de diseño.** El modelo primario del proyecto es XGBoost, que maneja `NaN` nativamente: en cada split, aprende hacia qué rama mandar los valores faltantes durante el entrenamiento, sin requerir un valor sustituto.
Imputar a ciegas antes de XGBoost destruiría la señal de "sin registro" que la Tarea 1 identificó como informativa (por ejemplo, el 72% de solicitantes sin `credit_card_balance` no es un faltante al azar: es que esa persona no tuvo tarjeta de crédito con Home Credit).
Por eso: para XGBoost se dejan los `NaN` tal cual y se agregan flags booleanos explícitos de "sin registro" por tabla relacional (una feature en sí misma, más allá de lo que el árbol infiera del NaN); la imputación numérica que sí necesita el baseline de Regresión Logística se resuelve aparte, con un `Pipeline`/`ColumnTransformer` de Scikit-learn específico para ese modelo, sin tocar la tabla de features cruda que consume XGBoost.

Se agregaron 6 flags (`bureau_no_record`, `bb_no_record`, `prev_no_record`, `pos_no_record`, `cc_no_record`, `inst_no_record`), cuyas proporciones confirman exactamente la cobertura por tabla ya documentada: `cc_no_record` 70.9%, `bb_no_record` 62.2%, `bureau_no_record` 14.2%, `pos_no_record` 5.3%, `prev_no_record` 4.9%, `inst_no_record` 4.7%.

### 2.12 Encoding de variables categóricas

`application_train`/`application_test` tienen 17 columnas de tipo `object`, 16 categóricas reales más `prev_ever_refused` (un flag booleano de 3 estados, True/False/sin dato, que quedó tipado como `object` por el `NaN` introducido en el `left merge`; se trató aparte, mapeado a 0/1).
La cardinalidad de las 16 categóricas reales es manejable: entre 2 y 58 valores únicos (`ORGANIZATION_TYPE` es la de mayor cardinalidad).
Se aplicó one-hot (`pd.get_dummies`, con `dummy_na=True` para preservar el faltante como categoría explícita en vez de perderlo) a las 16, sin agrupar categorías raras a priori: para un modelo de árboles (XGBoost) el volumen de columnas no penaliza de la misma forma que en un modelo lineal, y se prioriza no perder información sobre simplicidad.
El one-hot agregó 140 columnas nuevas (de 164 a 304 columnas totales).

### 2.13 Persistencia final

La tabla de features completa (limpieza de `application_*` + ratios de negocio + agregaciones de las 6 tablas relacionales + flags de "sin registro" + one-hot de categóricas) se guardó en `data/processed/features.parquet`: 356,255 filas × 304 columnas.

### 2.14 Formalización: `src/credixai/features.py` + `scripts/02_features.py` + DVC

La lógica validada interactivamente en `notebooks/02_features.ipynb` se transcribió a funciones reutilizables en `src/credixai/features.py` (una por paso: `load_application`, `clean_days_employed`, `add_business_ratios`, `aggregate_*` por tabla relacional, `add_no_record_flags`, `encode_categoricals`, orquestadas por `build_feature_table`), y a un script ejecutable `scripts/02_features.py` (`uv run python scripts/02_features.py`) que corre el pipeline completo y persiste el resultado, cumpliendo el deliverable de `prd.md` §5.
Se corrió el script de punta a punta y reprodujo exactamente el mismo resultado que el notebook: 356,255 filas × 304 columnas.

`data/raw` y `data/processed` se versionan con DVC (`dvc init`, `dvc add`): el repo trackea solo `data/raw.dvc`/`data/processed.dvc` (metadatos con hash), no los datos en sí, que siguen sin subirse a git.
Por ahora el cache de DVC es local, sin remote configurado (decisión explícita: alcance de este proyecto no incluye infraestructura de almacenamiento compartido); se documentó en `README.md` cómo re-trackear datos con `dvc add` para quien clone el repo.
Con esto, la Tarea 2 queda formalmente completa: imputación (§2.11), encoding (§2.12), script reproducible y DVC (esta sección).

---

## 3. Clustering / segmentación (`03_clustering.ipynb`)

_Pendiente._

---

## 4. Modelado supervisado

_Pendiente._

---

## 5. Explicabilidad y fairness

_Pendiente._

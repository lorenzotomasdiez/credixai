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

La lógica validada interactivamente en `notebooks/02_features.ipynb` se transcribió a funciones reutilizables en `src/credixai/features.py` (una por paso: `load_application`, `clean_days_employed`, `add_business_ratios`, `aggregate_*` por tabla relacional, `add_no_record_flags`, `encode_categoricals`, orquestadas por `build_feature_table`), y a un script ejecutable `scripts/02_features.py` (`uv run python scripts/02_features.py`) que corre el pipeline completo y persiste el resultado, cumpliendo el entregable esperado para la Tarea 2.
Se corrió el script de punta a punta y reprodujo exactamente el mismo resultado que el notebook: 356,255 filas × 304 columnas.

`data/raw` y `data/processed` se versionan con DVC (`dvc init`, `dvc add`): el repo trackea solo `data/raw.dvc`/`data/processed.dvc` (metadatos con hash), no los datos en sí, que siguen sin subirse a git.
Por ahora el cache de DVC es local, sin remote configurado (decisión explícita: alcance de este proyecto no incluye infraestructura de almacenamiento compartido); se documentó en `README.md` cómo re-trackear datos con `dvc add` para quien clone el repo.
Con esto, la Tarea 2 queda formalmente completa: imputación (§2.11), encoding (§2.12), script reproducible y DVC (esta sección).

---

## 3. Clustering / segmentación (`03_clustering.ipynb`)

### 3.1 Selección de features para clustering

K-Means mide distancia euclídea entre puntos.
Usar las 304 columnas de la tabla de features (Tarea 2) haría que las 140 columnas de one-hot, dispersas y en su mayoría en 0, dominaran la distancia por pura cantidad, aplastando el aporte de las variables continuas de negocio que ya mostraron señal real en las Tareas 1-2.
Un perfil de riesgo con 304 dimensiones tampoco es utilizable por un risk manager: el objetivo es un puñado de segmentos describibles en una frase.

Se seleccionaron 20 columnas en tres grupos: demográficas/capacidad de pago (edad, antigüedad laboral, ingreso, hijos, miembros de familia), ratios de negocio (`credit_to_income`, `annuity_to_income`, `credit_to_goods`) y un resumen de historial crediticio (la variable de mayor señal encontrada por tabla relacional en la Tarea 2: `bureau_active_count`, `bb_months_count_mean`, `prev_approval_rate`, `pos_dpd_rate`, `cc_utilization_mean`, `inst_late_rate`), más los 6 flags de "sin registro" ya construidos en la Tarea 2.

### 3.2 Imputación y escalado

Las variables de historial relacional conservan NaN por la misma razón documentada en la Tarea 2 (sin registro en esa tabla), con `bb_months_count_mean` y `cc_utilization_mean` como las de mayor faltante.
Como el flag `*_no_record` correspondiente ya viaja como columna aparte, se imputó con la mediana (por robustez a outliers, dado que varias columnas tienen colas largas) sin perder esa señal.
Se escaló con `StandardScaler`, requisito de K-Means: sin estandarizar, `AMT_INCOME_TOTAL` (rango de cientos de miles) dominaría por completo sobre variables de rango unitario como `bureau_active_count`.

### 3.3 Elección de `k`

Se probó `k` entre 2 y 8, con inertia sobre el dataset completo (307,511 filas) y silhouette estimado sobre una muestra de 10,000 puntos por costo computacional.
El silhouette cayó fuerte de `k=2` (0.4515) a `k=3+` (0.134-0.148), y la inertia decreció de forma suave sin codo marcado.
La hipótesis de trabajo era que ese salto en `k=2` podía ser un split trivial dominado por alguna variable binaria dispersa en vez de un perfil de riesgo real: se confirmó perfilando ese `k=2`, que separó clientes sin historial previo con Home Credit (cluster 0, 96-99% en los flags `prev_no_record`/`pos_no_record`/`inst_no_record`) de clientes con historial previo (cluster 1), un split real pero demasiado grueso para el objetivo de negocio.
Entre `k=3..8` las diferencias de silhouette son chicas y probablemente ruido de la muestra; se eligió `k=5` por ser un número de segmentos manejable para uso de negocio, con silhouette (0.138) prácticamente igual al máximo observado (`k=6`, 0.148).

### 3.4 Perfil de los 5 segmentos

La tasa de default global sobre train es 8.07%.
Los 5 segmentos muestran dispersión real de riesgo alrededor de ese valor, no solo diferencias de tamaño:

- **Cluster 0 - mora histórica** (13,862 clientes, 4.5%): `pos_dpd_rate` 27.6% e `inst_late_rate` 24.8% (vs. ~1-3% en el resto de los segmentos), default 12.86% (+59% vs. la tasa global, el segmento de mayor riesgo).
- **Cluster 4 - sin bureau externo, mayor carga relativa** (36,622, 11.9%): 100% `bureau_no_record`, ingreso más bajo (149,634), `annuity_to_income` más alto (0.195), default 9.75%.
- **Cluster 1 - familias jóvenes** (71,956, 23.4%): edad más baja (36.8 años), 1.46 hijos en promedio (el más alto), comportamiento de pago normal, default 8.64% (cerca del promedio).
- **Cluster 3 - núcleo estable, historial extenso** (168,980, 55.0%, el segmento más grande): historial extenso en todas las tablas relacionales (menor `*_no_record`), edad más alta (47.1 años), pocos hijos, default 7.29% (debajo del promedio).
- **Cluster 2 - nuevos en Home Credit** (16,091, 5.2%): 96-99% sin historial previo (`prev`/`pos`/`inst`), ingreso más alto (207,618), default 5.82% (el segmento de menor riesgo).

El hallazgo más contraintuitivo es que "sin historial previo" (cluster 2) resulta el segmento de *menor* riesgo, no mayor: son solicitantes de ingreso más alto que aún no tuvieron oportunidad de generar mora, a diferencia del cluster 0, que sí tiene historial y ese historial es de mora.
Esto valida la Tarea 2 desde otro ángulo: la ausencia de registro no es un dato faltante neutro, es información de negocio distinta según la variable.

### 3.5 Visualización PCA 2D

Se proyectaron las 20 dimensiones a 2 componentes principales solo para graficar (el clustering en sí corrió sobre el espacio completo de 20 dimensiones).
La varianza explicada por los 2 componentes es baja (24.7%), esperable al comprimir 20 dimensiones a 2, por lo que el gráfico subestima la separación real entre segmentos.
Aun así, el cluster 2 ("nuevos en Home Credit") queda visualmente aislado del resto en el plano PC1-PC2, consistente con su perfil de `*_no_record` extremo, mientras que los otros 4 segmentos forman sub-nubes distinguibles pero más solapadas entre sí.

### 3.6 Persistencia y formalización

El clustering se ajustó (`fit`) solo sobre la porción train, para no ajustar transformaciones con datos de test.
Para que el segmento esté disponible como feature de negocio sobre toda la población, se aplicó (`transform`/`predict`, no `fit`) el mismo `imputer`/`scaler`/`kmeans` ya ajustados a los 356,255 solicitantes (train + test).
La distribución de segmentos en test resultó razonablemente similar a la de train (sin discrepancias grandes en ningún cluster), lo que es evidencia de que el pipeline generaliza y no quedó sobreajustado a particularidades de train.

La lógica se transcribió a `src/credixai/clustering.py` (`fit_clustering`, `assign_clusters`, `build_segments`) y a un script ejecutable `scripts/03_clustering.py` (`uv run python scripts/03_clustering.py`), verificado de punta a punta: reprodujo el mismo shape (356,255 × 3) y los mismos tamaños de cluster exactos que el notebook.
El resultado se persistió en `data/processed/segments.parquet` (`SK_ID_CURR`, `IS_TRAIN`, `cluster`) y se versionó con DVC junto al resto de `data/processed`.

---

## 4. Modelado supervisado

### 4.1 Separación de datos y protocolo de validación

El objetivo es predecir `TARGET` (probabilidad de default) sobre la tabla de features de la Tarea 2.
Del total de 356,255 solicitantes, 307,511 tienen `TARGET` conocido (partición de entrenamiento) y 48,744 no (partición de test de Kaggle, reservada exclusivamente para un eventual submission al leaderboard).
Como el test de Kaggle no trae etiquetas, no puede usarse para medir performance del modelo.
Por eso se separó, dentro de la partición con etiqueta, un holdout de validación estratificado (80/20, manteniendo la proporción de 8.07% de default en ambos splits) para poder evaluar antes de tocar el test de Kaggle.

### 4.2 Métricas de evaluación

Se definió una función de evaluación única, reutilizada en todos los modelos, con cuatro métricas: ROC-AUC (primaria, robusta al desbalanceo), PR-AUC (relevante dado que solo el 8.07% de los casos son positivos), KS statistic (separación máxima entre las distribuciones acumuladas de buenos y malos pagadores, estándar en scoring crediticio) y Brier score (calibración: qué tan confiables son las probabilidades predichas, no solo su capacidad de ranking).

### 4.3 Baseline: Regresión Logística

Como piso de referencia interpretable por diseño se entrenó una Regresión Logística, con imputación por mediana y escalado estándar (requisitos de un modelo lineal basado en distancia/magnitud) y `class_weight="balanced"` para compensar el desbalanceo.
Resultado sobre el holdout: ROC-AUC 0.7637, PR-AUC 0.2488, KS 0.3982, Brier 0.1963.

### 4.4 Modelo primario: XGBoost

Se entrenó XGBoost como modelo primario, que maneja valores faltantes nativamente y es invariante a la escala de las features, por lo que no requiere el preprocesamiento del baseline.
El ajuste de hiperparámetros se hizo de forma incremental y documentada, contrastando siempre contra el mismo holdout:

- **Corrida inicial** (`scale_pos_weight` = razón negativos/positivos ≈ 11.4, para compensar el desbalanceo): ROC-AUC 0.7802, pero Brier 0.1751, muy por encima del objetivo de calibración.
- **Sin `scale_pos_weight`**: la hipótesis era que la reponderación distorsionaba las probabilidades sin aportar al ranking. Se confirmó: Brier bajó a 0.0660 y el ROC-AUC mejoró levemente a 0.7813.
- **Más árboles hasta converger**: el entrenamiento anterior no había activado el early stopping (seguía mejorando al cortar). Con más rondas convergió de verdad (mejor iteración 1343 de 3000) en ROC-AUC 0.7819, con una ganancia marginal ya mínima.
- **Árboles más profundos** (`max_depth` 7 en vez de 5): no mejoró (ROC-AUC 0.7815) y empezó a sobreajustar antes (mejor iteración 834 de 4000), señal de que la profundidad no era el cuello de botella.

Se optó por la configuración más simple (`max_depth=5`, sin `scale_pos_weight`) al no encontrar mejoras del tuneo manual adicional.

### 4.5 Validación cruzada y lectura honesta del resultado

Para descartar que el resultado del holdout fuera un artefacto de ese split particular, se corrió una validación cruzada estratificada de 5 folds sobre toda la partición de entrenamiento: ROC-AUC 0.7802 ± 0.0032 (rango entre folds: 0.7758-0.7848).
Este resultado confirma que el ROC-AUC real del modelo se ubica de forma estable en torno a 0.78, una mejora clara sobre el baseline (+1.6 puntos) pero por debajo del objetivo de 0.79 planteado para el proyecto.
La brecha remanente no cedió ante el tuneo manual de hiperparámetros ya descripto; cerrarla probablemente requiera trabajo adicional de feature engineering o ensembles, que se señala como línea de trabajo futura en las conclusiones.
El modelo final se reentrenó sobre el 100% de la partición de entrenamiento (sin holdout, ya no necesario para decidir hiperparámetros), con una cantidad fija de árboles (1086, el promedio de las mejores iteraciones observadas en el holdout y en los 5 folds de la validación cruzada) en lugar de early stopping.

### 4.6 Importancia de features y hallazgo de fairness

Se calculó la importancia de features del modelo final por `gain` (contribución promedio a la reducción de la función de pérdida), como insumo directo para la explicabilidad de la Tarea 5.
`EXT_SOURCE_2` y `EXT_SOURCE_3` encabezan el ranking, consistente con la correlación más fuerte con `TARGET` ya identificada en el EDA de la Tarea 1.
El resto del top 20 combina variables demográficas, ratios de negocio y varias agregaciones relacionales construidas en la Tarea 2 (utilización de tarjeta, tasa de rechazos previos, tasa de atrasos en cuotas), lo que confirma que ese trabajo de feature engineering aporta señal real.

Un hallazgo a marcar para la Tarea 5: `CODE_GENDER_M` y `CODE_GENDER_F` aparecen en el puesto 4 y 9 del ranking de importancia, con una contribución comparable a la de variables de negocio fuertes.
El género no es aquí solamente un atributo proxy a auditar externamente: es una feature que el modelo usa de forma directa para predecir.
Esto hace que la auditoría de fairness de la próxima tarea sea necesaria, no solo recomendable, para determinar si el modelo está usando el género como proxy de otra señal correlacionada o si introduce una disparidad de trato que deba corregirse.

### 4.7 Persistencia y tracking

El baseline y el modelo final de XGBoost se registraron en MLflow (hiperparámetros, las 4 métricas y el modelo serializado como artefacto), usando `mlflow-skinny` con un backend SQLite local, dado que el paquete `mlflow` completo todavía no soporta la versión de pandas usada en el proyecto.
La lógica se transcribió a `src/credixai/modeling.py` (`build_baseline`, `build_xgboost`, `evaluate`) y a un script ejecutable `scripts/04_modeling.py` (`uv run python scripts/04_modeling.py`), verificado end-to-end: reprodujo las métricas del baseline de forma exacta y las del modelo XGBoost final con una diferencia mínima (ROC-AUC 0.7815 contra 0.7819), coherente con la variabilidad ya observada entre folds de la validación cruzada.

---

## 5. Explicabilidad y fairness

### 5.1 SHAP global: importancia y estabilidad del ranking

Se calculó SHAP (`TreeExplainer`, exacto para modelos de árboles) sobre una muestra aleatoria de 5.000 solicitudes del set de entrenamiento.
El ranking por magnitud promedio de SHAP confirma el top-3 ya visto en el EDA y en la importancia por `gain` de la Tarea 4: `EXT_SOURCE_2`, `EXT_SOURCE_3` y `EXT_SOURCE_1`, con diferencia las variables más predictivas del dataset.
Le siguen variables de agregación construidas en la Tarea 2 (`prev_annuity_mean`, `credit_to_goods`, `prev_credit_mean`, `bureau_debt_mean`), lo que vuelve a validar ese trabajo de feature engineering con una métrica de explicabilidad independiente del `gain`.
`CODE_GENDER_M` cae al puesto 16 por SHAP (mean_abs_shap=0.070), más abajo que en el ranking por `gain` de la Tarea 4 (puesto 4): la diferencia se explica porque `gain` es sensible a splits puntuales de alto impacto aunque afecten pocas filas, mientras que SHAP mide el impacto promedio sobre toda la población.
Que la variable siga en el top-20 con dos métricas distintas refuerza, no debilita, la necesidad de la auditoría de fairness de este capítulo.

Para poder usar SHAP con confianza en explicaciones individuales, se midió la estabilidad del ranking mediante bootstrap: 30 remuestreos con reposición de la muestra de 5.000 filas, comparando cada ranking resultante contra el de referencia con el coeficiente de Kendall tau.
Resultado: tau = 0.9917 ± 0.0010, muy por encima del piso de referencia de 0.90 tomado como objetivo del proyecto para explicaciones estables.
El orden de importancia de las features, por lo tanto, no depende del subconjunto de solicitudes que se mire.

### 5.2 SHAP local: explicación por solicitud

Se inspeccionaron tres solicitudes de la muestra: la de mayor probabilidad de default predicha (proba=0.8583), la de menor (proba=0.0020) y una cerca del umbral de decisión (proba=0.4992), usando gráficos de tipo waterfall.
En la de mayor riesgo, el motor principal es un `EXT_SOURCE_2` bajo (0.108), reforzado por una tasa de atraso en cuotas previas alta y una relación crédito/bien elevada; ningún otro feature individual se acerca a esa magnitud, aunque el conjunto de las ~292 features restantes también aporta una parte significativa del empuje total.
En la de menor riesgo, `EXT_SOURCE_2` y `EXT_SOURCE_3` altos dominan la explicación, en el sentido esperado.
El caso cercano al umbral es el más ilustrativo: `EXT_SOURCE_2` muy bajo empuja fuerte hacia el riesgo, pero `EXT_SOURCE_1` relativamente alto empuja en la dirección contraria, y el resto de las señales terminan casi cancelándose, dejando la probabilidad casi exactamente en el borde de decisión.
En ninguno de los tres casos `CODE_GENDER_M` aparece entre las diez contribuciones individuales más grandes, algo esperable dada su magnitud promedio moderada; su efecto se audita de forma agregada en la sección 5.4, no caso a caso.

### 5.3 Reason codes y exclusión de atributos protegidos

Se implementó una función de reason codes que, para una solicitud dada, toma las features con mayor contribución positiva a SHAP (las que empujan hacia mayor riesgo) y las traduce a un texto legible de negocio, no al nombre crudo de la columna, siguiendo el criterio de máximo cuatro razones que suele considerarse útil para un solicitante.
La función excluye de forma explícita a los atributos protegidos/proxy (`CODE_GENDER_M`, `CODE_GENDER_F`, `DAYS_BIRTH`) de cualquier reason code, aun cuando el modelo los use internamente: que el modelo use una variable protegida de forma medible es un hecho a auditar (sección 5.4), pero comunicársela a un solicitante como motivo de rechazo es una práctica distinta, y no permitida.
Para la solicitud de mayor riesgo, las razones generadas fueron: score de riesgo externo bajo, tasa de atraso en cuotas previas, relación crédito/bien elevada e historial corto de créditos de consumo previos; todas coherentes con la explicación SHAP local de la sección 5.2.

### 5.4 Auditoría de fairness: el modelo amplifica la disparidad real

Se auditó el modelo con tres métricas de fairness sobre género (`CODE_GENDER_M`) y grupo etario (bucketizado a partir de `DAYS_BIRTH`): diferencia de paridad estadística (statistical parity difference), razón de impacto dispar (disparate impact) y diferencia de igualdad de oportunidad (equal opportunity difference), dentro de un rango de referencia de [-0.1, 0.1] para las diferencias (convención habitual en la literatura de fairness).
La decisión binaria "alto riesgo" se definió con un umbral reproducible: el percentil que marca como alto riesgo a la misma proporción de solicitudes (8.07%) que la tasa real de default, evitando introducir un umbral de negocio arbitrario no definido en el alcance del proyecto.

| Grupo | Statistical parity diff | Disparate impact (ratio) | Equal opportunity diff |
|---|---|---|---|
| Género | 0.0608 (dentro de rango) | 0.496 (falla la regla del 80%) | 0.1315 (fuera de rango) |
| Edad (<30 vs. 60+) | 0.1282 (fuera de rango) | 0.134 (falla la regla del 80%) | 0.3257 (fuera de rango) |

El hallazgo central de esta sección no es solo que existan disparidades, sino que el modelo las amplifica respecto de la disparidad real observada en los datos.
En género, la brecha real de default es de 3.14 puntos porcentuales (10.14% en hombres vs. 6.99% en mujeres, ratio 0.690); el modelo la traduce en una brecha de "alto riesgo" predicho de 6.08 puntos (ratio 0.496), prácticamente el doble tanto en puntos porcentuales como en el ratio.
En edad, el mismo patrón es más marcado: brecha real de 6.52 puntos entre menores de 30 y mayores de 60 (ratio 0.430) contra una brecha predicha de 12.82 puntos (ratio 0.134), otra vez más del doble.
El caso de género es el más sutil de comunicar con una sola métrica: la diferencia en puntos porcentuales queda dentro del rango de referencia por la baja tasa base general de default (8.07%), pero el ratio (disparate impact) y la diferencia de igualdad de oportunidad, que no dependen de esa escala, muestran una disparidad clara.
Mirar una sola métrica de fairness hubiera dado, en este caso, un falso resultado aprobatorio; el grupo etario, en cambio, falla las tres métricas sin ambigüedad.

Esta evidencia cuantitativa establece que tratar el género y la edad como simples atributos proxy subestima el problema: no es una correlación incidental con el riesgo real, es una amplificación medible que debe documentarse como limitación conocida del modelo.
Como líneas de mitigación para trabajo futuro, fuera del alcance de este capítulo, quedan: (1) thresholding con restricción de fairness (por ejemplo, `ThresholdOptimizer` de Fairlearn), o (2) remover o neutralizar `CODE_GENDER` y las variables más correlacionadas con edad antes de entrenar, y volver a medir si la amplificación persiste por la vía de otras features proxy.

### 5.5 Contrafácticos (DiCE): una limitación real, no un bug a resolver

Se intentó generar contrafácticos con DiCE (método `random`) para la solicitud de mayor riesgo, variando solo un subconjunto reducido y deliberado de features (monto de cuota, monto y precio del bien financiado), excluyendo explícitamente atributos protegidos y los `EXT_SOURCE_*` (scores externos no accionables por el solicitante).
DiCE no admite valores faltantes en la fila de consulta, a diferencia de XGBoost, que maneja los NaN de forma nativa; imputar la fila completa a la mediana para poder usar la herramienta cambió la predicción del modelo lo suficiente como para cruzar el umbral de decisión (de 0.8583 con NaN nativos a 0.4618 imputada), invalidando el contrafactico como explicación fiel de esa decisión real.
Se intentó acotar el problema restringiendo la búsqueda a solicitudes sin ningún valor faltante (21 de 5.000 en la muestra), pero ninguna de esas 21 filas supera el umbral de decisión: en este dataset, tener historial completo en las 6 tablas relacionales está asociado casi sin excepción a bajo riesgo, de modo que no existe un caso representativo de "solicitud rechazada con historial completo" sobre el cual demostrar el contrafactico.

La conclusión de esta sección es en sí misma un hallazgo de arquitectura, no un problema de configuración: DiCE, tal como está implementado en la librería utilizada, requiere filas completas, pero el manejo nativo de NaN es parte central de cómo XGBoost calcula el riesgo en este dataset, y esas dos cosas son incompatibles exactamente para la población que más necesitaría una explicación contrafactica accionable, los solicitantes con historial incompleto.
Como trabajo futuro, se identifican dos caminos: generar contrafácticos solo sobre el subconjunto de features que nunca tienen NaN (con pérdida de cobertura del historial crediticio), o reemplazar DiCE por un método de contrafácticos que perturbe features sin necesidad de imputar el resto de la fila.

### 5.6 Persistencia y formalización

El cálculo de SHAP, los reason codes y la auditoría de fairness se transcribieron a `src/credixai/explainability.py` (`compute_shap`, `mean_abs_shap`, `reason_codes`, `fairness_report`) y a un script ejecutable `scripts/05_explainability.py` (`uv run python scripts/05_explainability.py`), verificado end-to-end: reprodujo de forma exacta el ranking SHAP, los reason codes y las métricas de fairness obtenidas en el notebook.
Los contrafácticos con DiCE quedan únicamente en `notebooks/05_xai.ipynb` como prueba de concepto de la técnica, dada la limitación de compatibilidad con NaN documentada en la sección 5.5: no se formalizaron en `src/` ni en el script, para no ofrecer como funcionalidad de producción algo que no es fiable para la población que más lo necesitaría.

## 6. Visualización e informe ejecutivo (`app/dashboard.py`)

### 6.1 Alcance y arquitectura

El dashboard se construyó con Streamlit y cubre las cuatro vistas requeridas: métricas globales del modelo, segmentación, auditoría de fairness y detalle por solicitud (probabilidad + SHAP + reason codes).
La lógica de carga y preparación de datos se separó en `src/credixai/dashboard.py`, reutilizando sin duplicar las funciones ya formalizadas en `credixai.modeling`, `credixai.clustering` y `credixai.explainability` (Tareas 3, 4 y 5); `app/dashboard.py` queda como entrypoint delgado, siguiendo el mismo patrón de separación entre lógica reutilizable y script/entrypoint usado en las tareas anteriores.
El modelo final y la muestra para SHAP se calculan una sola vez por sesión mediante `st.cache_resource`, evitando reentrenar en cada interacción del usuario con la app.

### 6.2 Verificación

Antes de pedir al usuario que corriera la app, se verificaron por separado las funciones de datos de `src/credixai/dashboard.py` (fuera de Streamlit): las métricas de holdout, el umbral de decisión, el perfil de los 5 segmentos y las métricas de fairness por género y edad coincidieron de forma exacta con los valores ya validados en las Tareas 3, 4 y 5.
La app se probó también con el framework de testing de Streamlit (`AppTest`), confirmando que las cuatro pestañas cargan sin excepciones.
El usuario corrió la app en su navegador (`uv run streamlit run app/dashboard.py`) y confirmó visualmente las cuatro pestañas: resumen ejecutivo, segmentación, fairness y detalle por solicitud, incluida la navegación entre distintas solicitudes y la explicación SHAP de un caso de alto riesgo (proba=0.7102, por encima del umbral 0.2114).

### 6.3 Ajustes de usabilidad tras la revisión visual

Dos ajustes surgieron de la revisión visual junto con el usuario, no previstos en el diseño inicial:

1. El gráfico waterfall de SHAP se generaba con fondo blanco de matplotlib por defecto, lo que contrastaba mal con el tema oscuro del dashboard; se ajustó a un estilo oscuro (`dark_background`, fondo `#0e1117`) consistente con el resto de la interfaz.
2. El selector de solicitud (`SK_ID_CURR`) no permitía saber la probabilidad de una solicitud sin elegirla primero, dificultando encontrar un caso de alto riesgo a propósito para probar los reason codes; se resolvió ordenando el selector de mayor a menor probabilidad de default y mostrando la probabilidad junto a cada ID, de forma que el primer ítem de la lista es siempre el de mayor riesgo.

### 6.4 Informe ejecutivo

Se redactó `docs/informe-ejecutivo.md`, un resumen en lenguaje no técnico para un público de negocio, con los mismos hallazgos centrales de este informe (performance, segmentación, explicabilidad, el hallazgo de amplificación de fairness y la limitación de contrafácticos) sin la profundidad metodológica.

---

## 7. Documentación técnica

### 7.1 Alcance

La Tarea 7 formaliza como documentación de repositorio lo que hasta este punto vivía repartido entre notebooks, scripts y este informe: un README de nivel producción, una model card y la arquitectura de decisiones (ADRs y characteristics), ya iniciada desde el arranque del proyecto y mantenida al día en `docs/adr/`, `docs/architecture-characteristics.md` y `docs/architecture-style-selection.md`.
Por eso este capítulo no repite ese trabajo, documenta lo que se agregó específicamente para el cierre del núcleo académico: la model card y la reescritura del README.

### 7.2 Model card

Se creó `docs/model-card.md` siguiendo el formato de Mitchell et al. (2019), "Model Cards for Model Reporting": detalles del modelo, uso previsto, datos de entrenamiento, métricas de performance, explicabilidad, fairness (con la tabla completa de la sección 5.4 de este informe), limitaciones conocidas y consideraciones éticas/regulatorias.
Todos los números citados en la model card son una referencia directa a valores ya reportados y verificados en las secciones 4 y 5 de este informe, sin recalcular nada nuevo; el objetivo del documento es ser el punto de entrada único para un auditor o regulador que necesite evaluar performance, fairness y limitaciones del modelo, sin tener que leer el informe completo.

### 7.3 README

Se reescribió `README.md` a un formato de nivel producción: un hook de una frase basado en el hallazgo de fairness (la amplificación de disparidad, no un resumen genérico del proyecto), tres resultados técnicos cuantificados en la portada, un diagrama de arquitectura (Mermaid) que distingue explícitamente qué está implementado (núcleo académico: Data & Feature Store, ML Scoring, XAI, Segmentación, Serving vía Streamlit) de qué es una extensión de portfolio no implementada todavía (API REST, RAG normativo, copiloto agéntico), una tabla de navegación a toda la documentación del repo, y los comandos exactos para reproducir el pipeline completo y levantar el dashboard.
Se optó deliberadamente por no incluir un GIF de demo, que requeriría grabar y editar video, fuera del alcance de esta sesión de trabajo; queda señalado como pendiente de portfolio, no se simuló ni se dejó un placeholder engañoso.

### 7.4 API docs (FastAPI/OpenAPI)

La documentación de API vía FastAPI/OpenAPI se definió desde el inicio del proyecto como una extensión de portfolio, no como parte del núcleo académico obligatorio.
No se implementó en esta sesión: el sistema expone sus resultados únicamente a través del dashboard Streamlit de la Tarea 6, no de una API REST.
Queda como trabajo futuro, junto con el RAG normativo y el copiloto agéntico, consistente con el principio de gestión de scope adoptado para priorizar primero el núcleo académico completo.

---

## 8. Extensiones de portfolio

Con el núcleo académico (tareas 1-7) cerrado, lo que sigue son las tres capas avanzadas del proyecto (RAG, agentes, MLOps/LLMOps), ejecutadas en un orden definido por dependencia técnica real entre componentes, no por el orden cronológico del roadmap original.
Cada paso de esta secuencia se documenta como una subsección propia a medida que se completa.

### 8.1 Tests automatizados (`pytest`)

Primer paso de la secuencia: sin tests, un pipeline de CI/CD solo podría lintear código, no validar que se comporta como se espera.
Se agregaron 21 tests unitarios sobre las cuatro funciones reutilizables del proyecto (`src/credixai/features.py`, `clustering.py`, `modeling.py`, `explainability.py`), todos con datos sintéticos generados dentro del propio test, sin depender del dataset real de Kaggle ni de archivos versionados con DVC, para que la suite corra en cualquier entorno sin necesidad de descargar datos.

La cobertura por módulo:

- **`features.py`:** limpieza del centinela de `DAYS_EMPLOYED`, ratios de negocio (incluida la división por ingreso cero, que debe dar `NaN` y no `inf` ni una excepción), flags de "sin historial", one-hot de categóricas, cada función de agregación relacional (`aggregate_bureau`, `aggregate_bureau_balance`) contra CSVs mínimos escritos en un directorio temporal, y un test de integración de punta a punta de `build_feature_table` con una versión minúscula (2-3 solicitantes) de las 8 tablas del dataset.
- **`clustering.py`:** derivación de `AGE_YEARS`/`EMPLOYED_YEARS`, que `fit_clustering` entrena solo sobre `IS_TRAIN == 1`, que `assign_clusters` cubre a toda la población (train y test), y que `build_segments` separa correctamente dos grupos sintéticos con ingresos claramente distintos en clusters distintos.
- **`modeling.py`:** que `feature_columns` excluye exactamente `SK_ID_CURR`/`TARGET`/`IS_TRAIN`, que `evaluate` reproduce de forma exacta las métricas de referencia de scikit-learn (ROC-AUC, PR-AUC, Brier), y que tanto el baseline como XGBoost entrenan y predicen probabilidades válidas sobre datos sintéticos, incluido un caso con valores faltantes para validar el manejo nativo de NaN de XGBoost.
- **`explainability.py`:** que `mean_abs_shap` ordena correctamente por magnitud promedio, que `reason_codes` nunca incluye un atributo protegido aunque tenga la mayor contribución positiva (el caso más importante de todo el proyecto para esta suite), que respeta el orden descendente y el límite `top_n`, y que `fairness_report` reproduce de forma exacta las métricas de referencia de Fairlearn sobre un caso sintético que simula el mismo patrón de amplificación de disparidad documentado en la sección 5.4.

Toda la suite corre en menos de 3 segundos (`uv run pytest`), sin advertencias propias del código (las únicas advertencias observadas son `PendingDeprecationWarning` internas de la librería `shap`, no relacionadas con este proyecto).

### 8.2 API REST (FastAPI) - RF-8

Segundo paso de la secuencia: formaliza RF-8 (scoring y explicación vía API REST) reutilizando, sin duplicar, el modelo de la Tarea 4 y el SHAP/reason codes de la Tarea 5.
Se desarrolló con TDD a partir de este paso: los archivos `tests/test_api_service.py` y `tests/test_api_http.py` se escribieron primero (contra un módulo `credixai.api` que todavía no existía, confirmando que fallaban por `ModuleNotFoundError`), y luego se implementó `src/credixai/api.py` y `app/api.py` hasta que la suite completa pasó en verde.

**Diseño:** `ScoringService` (`src/credixai/api.py`) envuelve el mismo bundle que ya produce `credixai.dashboard.train_full_model` (modelo, features, probabilidades, umbral) y responde consultas por `SK_ID_CURR`.
A diferencia del dashboard, que precalcula SHAP solo sobre una muestra de 2.000 solicitudes por motivos de UI, `ScoringService.explain` calcula SHAP bajo demanda para la fila puntual solicitada con `shap.TreeExplainer`, lo que permite explicar cualquier solicitud de la población, no solo las de la muestra.
`app/api.py` es el entrypoint delgado (patrón ya usado en `app/dashboard.py`): dos endpoints, `GET /score/{sk_id_curr}` y `GET /explain/{sk_id_curr}`, con esquemas de respuesta Pydantic y documentación OpenAPI automática en `/docs`.
Los reason codes solo se devuelven cuando la decisión es "alto_riesgo", igual que en el dashboard, y nunca incluyen atributos protegidos, verificado explícitamente por test.

**Verificación:** la suite de tests (`tests/test_api_service.py`, `tests/test_api_http.py`) usa un bundle sintético inyectado por `dependency_overrides` de FastAPI, sin tocar `data/processed` ni reentrenar el modelo real, para que corra en segundos.
Por separado, se levantó la API real (`uv run uvicorn app.api:app`) contra `data/processed/features.parquet` y se confirmó manualmente: `/health` responde `{"status": "ok"}`, `/score/100002` devuelve probabilidad 0.4728 y decisión "alto_riesgo" (umbral 0.2114, el mismo ya validado en el dashboard de la Tarea 6), `/explain/100002` devuelve un valor SHAP por feature más los reason codes, y un `SK_ID_CURR` inexistente devuelve `404`.

**Fuera de alcance de este paso:** el endpoint `/copilot` mencionado en el diseño original de la capa de serving depende del copiloto agéntico (pasos 5-6 de la secuencia de extensiones), que todavía no existe; no se agregó un stub vacío para no ofrecer una funcionalidad que no está implementada.

El informe ejecutivo en lenguaje no técnico, dirigido a un público de negocio, se redactó en `docs/informe-ejecutivo.md`: resume los resultados del modelo, la segmentación, la explicabilidad, el hallazgo de amplificación de disparidad en fairness y la limitación de los contrafácticos, sin el detalle metodológico de este informe técnico.

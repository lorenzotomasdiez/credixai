# Informe ejecutivo - CrediXAI

## Que es CrediXAI

CrediXAI es un sistema de scoring crediticio para una fintech simulada.
Predice la probabilidad de que un solicitante de credito incumpla sus pagos.
A diferencia de un scoring tradicional de caja negra, cada decision viene acompanada de una explicacion trazable: que factores la motivaron y si el proceso es equitativo entre grupos.
Este documento resume los resultados para un publico no tecnico.
El detalle metodologico completo esta en `docs/informe-final.md`.

## Resultados del modelo

El modelo (XGBoost) predice el riesgo de default con un ROC-AUC de 0.7815 sobre datos no vistos, es decir, distingue correctamente entre un solicitante que efectivamente incumple y uno que no en el 78% de los pares comparados al azar.
La tasa de incumplimiento real en la base de solicitantes es del 8.07%.
El umbral de decision se fija para marcar como "alto riesgo" a la misma proporcion de solicitudes que el default real observado, evitando un corte arbitrario.

## Segmentacion de clientes

Se identificaron 5 segmentos de perfiles de riesgo mediante clustering no supervisado, cada uno con una tasa de default y un perfil socioeconomico propio (ingreso, credito solicitado, historial crediticio previo).
Esta segmentacion permite a un analista de negocio priorizar politicas comerciales o de cobranza por grupo, en lugar de tratar a toda la cartera de manera uniforme.

## Explicabilidad

Cada decision individual viene acompanada de hasta 4 razones concretas y en lenguaje llano (por ejemplo, "score de riesgo externo por debajo del promedio"), calculadas con SHAP y compatibles con los requisitos legales de notificacion de accion adversa (ECOA/Regulation B en EE.UU., como referencia de buena practica de la industria).
Estas razones nunca incluyen genero ni edad, aunque el modelo pueda usar esas variables internamente: comunicarlas como motivo de rechazo no es legalmente admisible.

## Hallazgo principal: equidad del modelo

Se audito el modelo para verificar si trata de forma equitativa a distintos grupos de genero y edad.
El hallazgo central es que el modelo no solo refleja la disparidad real de incumplimiento entre estos grupos, sino que la amplifica: la brecha en la tasa de "alto riesgo" predicha es aproximadamente el doble de la brecha real observada en los datos, tanto para genero como para grupo etario.
Esto significa que, sin intervencion, el modelo en su forma actual generaria un impacto desigual mayor al que justifica el riesgo real, algo relevante desde el punto de vista regulatorio y reputacional.
Se documentaron vias de mitigacion (ajuste de umbral por grupo, remocion o neutralizacion de variables proxy) como trabajo futuro antes de cualquier uso en produccion.

## Limitacion conocida: explicaciones contrafacticas

Se evaluo generar explicaciones del tipo "si el solicitante presentara tal cambio, la decision se revertiria".
Se encontro una incompatibilidad estructural entre esta tecnica y la forma en que el modelo usa la ausencia de historial crediticio como senal de riesgo: imputar los datos faltantes para poder generar el contrafactico cambia la decision que se esta intentando explicar.
Este hallazgo, mas que un defecto a resolver, es una caracteristica del dominio que condiciona que tecnicas de explicabilidad son aplicables, y queda documentado para la version de produccion.

## Como ver el sistema en accion

Un dashboard interactivo (Streamlit) permite explorar los resultados: metricas globales del modelo, perfiles de segmento, resultados de la auditoria de equidad, y el detalle de una solicitud individual con su explicacion y sus razones de rechazo.

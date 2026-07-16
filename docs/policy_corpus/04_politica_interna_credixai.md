# Política interna de originación y equidad de CrediXAI

> Nota: este documento describe la política interna ficticia de la fintech simulada "CrediXAI", escrita para el corpus RAG de este proyecto.
> No corresponde a ninguna entidad real; formaliza las decisiones de diseño ya tomadas en el modelo, la API y el dashboard del proyecto, para que el copiloto agéntico y el RAG puedan citarlas como fuente de política.

## Umbral de decisión

CrediXAI clasifica cada solicitud como "riesgo_aceptable" o "alto_riesgo" según un umbral de probabilidad de default calibrado sobre la tasa de default histórica de la cartera (percentil correspondiente a la proporción real de defaults observada en el conjunto de entrenamiento).
Una solicitud con probabilidad de default por encima del umbral se clasifica como alto riesgo y activa el flujo de comunicación de reason codes.

## Reason codes

Siguiendo el Comentario CFPB a §1002.9 (ver documento de adverse action en este mismo corpus), CrediXAI comunica como máximo 4 reason codes por solicitud de alto riesgo, derivados directamente de los valores SHAP de esa predicción individual.
Los reason codes nunca incluyen atributos protegidos (género, edad) ni sus proxies conocidos, aun cuando esos atributos participen en el modelo interno y en la auditoría de equidad.

## Auditoría de equidad (fairness)

CrediXAI audita periódicamente su modelo con las siguientes métricas de equidad, calculadas por grupo protegido (género, tramo etario):

- **Statistical parity difference**
- **Equal opportunity difference**
- **Disparate impact**

El rango de referencia aceptable para statistical parity difference y equal opportunity difference es [-0.1, 0.1], siguiendo la convención de la librería AIF360.
Cuando una métrica excede ese rango, la política interna exige documentar la disparidad en el model card del modelo vigente, evaluar si corresponde ajustar el umbral de decisión por grupo (thresholding con restricción de fairness), y en caso de no poder mitigarla, comunicar explícitamente la limitación a las áreas de negocio y cumplimiento antes de mantener el modelo en producción.

## Revisión humana

Todo caso de alto riesgo cercano al umbral de decisión (dentro de una banda de incertidumbre a definir por el comité de riesgos) debe pasar por revisión humana antes de la comunicación final al solicitante.
El copiloto agéntico de CrediXAI puede redactar un borrador de memo crediticio, pero ese borrador siempre requiere revisión y aprobación humana antes de convertirse en una decisión comunicada.

## Observabilidad y monitoreo

CrediXAI monitorea drift en la distribución de las variables de entrada de producción respecto de los datos de entrenamiento (PSI, population stability index).
Un PSI mayor a 0.2 en cualquier variable relevante del modelo dispara una alerta y una revisión del modelo, dado que indica un cambio sustancial en la población de solicitantes que puede degradar la calidad de las predicciones y de las auditorías de equidad asociadas.

# Guía de usuario - CrediXAI

Esta guía explica cómo usar CrediXAI paso a paso, sin dar por sabido conocimiento técnico previo.
Está pensada para un analista de riesgo, un product owner, o cualquier persona que quiera explorar el sistema sin escribir código.
Para el detalle metodológico completo, ver `docs/informe-final.md`; para un resumen de resultados, ver `docs/informe-ejecutivo.md`.

## Qué partes de CrediXAI tienen pantalla y cuáles no

Todo lo que un usuario de negocio necesita (scoring, segmentación, fairness, preguntas de política y memos del copiloto) tiene pantalla propia en el dashboard de Streamlit, cubierto en la sección "Dashboard paso a paso" de esta guía.
El monitoreo de drift es la única excepción: es un reporte generado bajo demanda, no una pantalla interactiva (sección "Dónde ver el reporte de monitoreo de drift").
Quien prefiera usar la API directamente (por ejemplo, para integrarla con otro sistema) puede hacerlo por Swagger, explicado en la sección "Usar la API directamente (opcional)".

## Antes de empezar: levantar el sistema

Instrucciones completas de instalación en `README.md`.
En resumen, con las dependencias ya instaladas (`uv sync`) y los datos ya disponibles:

```
uv run python app/dashboard_launcher.py
```

Esto abre el dashboard en el navegador, normalmente en `http://localhost:8501`.
La primera vez tarda uno o dos minutos porque entrena el modelo; las siguientes veces es instantáneo.

## Dashboard paso a paso

El dashboard tiene seis pestañas.
Las dos últimas ("Consulta normativa" y "Copiloto") además de los datos ya cargados necesitan la API corriendo (ver requisitos en `README.md`, secciones "RAG normativo" y "Copiloto agéntico": Qdrant y una clave de OpenRouter); si la API no está corriendo, esas dos pestañas muestran un mensaje de error claro en vez de fallar en silencio.

### Pestaña "Resumen ejecutivo"

Muestra qué tan bien predice el modelo, medido sobre solicitudes que el modelo no vio durante el entrenamiento (para que el número sea honesto).

- **ROC-AUC:** qué tan bien distingue el modelo entre un solicitante que efectivamente incumple y uno que no. 0.78 significa que, comparando dos solicitudes al azar (una que incumplió y otra que no), el modelo las ordena correctamente en el 78% de los casos.
- **PR-AUC, KS, Brier:** métricas técnicas complementarias, reportadas por completitud; el detalle de cada una está en `docs/informe-final.md`.
- **Tasa de default real:** de cada 100 solicitudes en la base, cuántas efectivamente incumplieron.
- **Umbral de decisión:** a partir de qué probabilidad una solicitud se marca como "alto riesgo". Se fija para que la proporción de solicitudes marcadas como alto riesgo coincida con la tasa de incumplimiento real observada, en vez de un corte arbitrario.

### Pestaña "Segmentación"

Agrupa a los solicitantes en 5 perfiles de riesgo distintos, encontrados automáticamente a partir de sus datos (ingreso, crédito solicitado, historial previo), sin que nadie les haya asignado una categoría a mano.

- La tabla muestra, por segmento: qué proporción de la cartera representa, su tasa de default, y su perfil socioeconómico promedio.
- El gráfico de barras compara la tasa de default entre segmentos, de un vistazo.
- Uso de negocio: priorizar políticas comerciales o de cobranza por segmento, en vez de tratar a toda la cartera de la misma forma.

### Pestaña "Fairness"

Audita si el modelo trata de forma equitativa a distintos grupos de género y edad.

- **Statistical parity difference, disparate impact, equal opportunity difference:** tres formas distintas de medir si un grupo recibe más rechazos que otro. El rango de referencia aceptable (convención de la industria) está marcado en pantalla: [-0.1, 0.1].
- La tabla compara, por grupo, la tasa de default real contra la tasa de "alto riesgo" que predice el modelo.
- El cuadro informativo al final resume el hallazgo principal: el modelo no solo refleja la disparidad real de default por género y edad, sino que **la amplifica al doble**. Esto es relevante desde el punto de vista regulatorio y reputacional, y está documentado como limitación abierta (no resuelta todavía) en `docs/informe-final.md` sección 5.4 y en `docs/model-card.md`.

### Pestaña "Detalle por solicitud"

Permite elegir una solicitud puntual (por `SK_ID_CURR`, el identificador de la solicitud, ordenadas de mayor a menor riesgo en el selector) y ver:

- Su probabilidad de default y la decisión resultante (alto riesgo / riesgo aceptable).
- Un gráfico "waterfall" que muestra qué factores empujaron la probabilidad hacia arriba o hacia abajo, y cuánto pesó cada uno.
- Si la solicitud es de alto riesgo, hasta 4 **reason codes**: razones concretas en lenguaje llano (por ejemplo, "score de riesgo externo por debajo del promedio"), pensadas para poder comunicárselas al solicitante como motivo de rechazo. Estas razones nunca mencionan género ni edad, aun cuando el modelo usa esas variables internamente, porque comunicarlas como motivo de rechazo no es admisible.

### Pestaña "Consulta normativa"

Responde preguntas de política/normativa citando siempre documento y fragmento fuente.

1. Escribir una pregunta en el campo de texto, por ejemplo: `¿Cuántos reason codes como máximo se comunican al solicitante?`.
2. Hacer clic en "Consultar".
3. La respuesta aparece debajo: un texto en lenguaje natural, más las citas (documento y fragmento exacto) de donde salió esa respuesta. El corpus de política que responde estas preguntas (`docs/policy_corpus/`) son resúmenes sintetizados con fines educativos, no el texto normativo oficial.

### Pestaña "Copiloto"

Investiga una solicitud puntual y redacta un borrador de memo crediticio con reason codes y citas de política.

1. Elegir un `SK_ID_CURR` en el selector (los mismos números que aparecen en "Detalle por solicitud", ordenados de mayor a menor riesgo).
2. Hacer clic en "Redactar memo". Una solicitud de alto riesgo tarda más (hasta un minuto) porque el copiloto hace varios pasos: consulta el score, pide la explicación SHAP, busca la política aplicable, y redacta el memo; una de riesgo aceptable es rápida porque salta esos pasos intermedios.
3. El resultado muestra la decisión, un `status` (`approved` si el memo pasó la revisión automática de calidad, o `needs_human_review` si no la pasó dos veces y necesita que una persona lo revise), el memo, y sus citas de política.

## Usar la API directamente (opcional)

Todo lo anterior también está disponible por API REST, útil para integrar CrediXAI con otro sistema en vez de usarlo por el dashboard.
Con la API corriendo (`uv run uvicorn app.api:app --reload`), Swagger (una interfaz de prueba que se genera sola a partir de la API, sin escribir código) queda disponible en:

```
http://localhost:8000/docs
```

Ahí se puede desplegar cualquier endpoint (`/score/{sk_id_curr}`, `/explain/{sk_id_curr}`, `/rag/query`, `/copilot/memo/{sk_id_curr}`), hacer clic en "Try it out", completar los parámetros, y "Execute".

## Dónde ver el reporte de monitoreo de drift

No es una pantalla interactiva sino un archivo de reporte, generado bajo demanda.
Instrucciones para generarlo en `README.md`, sección "Monitoreo de drift".
Una vez generado, se abre directamente en el navegador desde `models/monitoring/drift_report.html`.

## Preguntas frecuentes

**¿Por qué las pestañas "Consulta normativa" y "Copiloto" necesitan la API corriendo aparte del dashboard?**
El dashboard consume esas dos funciones por HTTP contra la API en vez de reimplementarlas, para no duplicar el setup de Qdrant/OpenRouter/LangGraph dentro de Streamlit. Sin la API corriendo, esas dos pestañas muestran un mensaje de error explicando qué falta, en vez de fallar en silencio.

**¿Puedo confiar en el memo que redacta el copiloto sin revisarlo?**
No. El sistema está diseñado para que un memo con `status: needs_human_review` no se use sin que una persona lo revise, y aun con `status: approved`, el memo es un borrador de apoyo al analista, no una decisión final automatizada.

**¿Por qué algunas solicitudes no tienen razones (reason codes)?**
Solo se generan para solicitudes marcadas como "alto riesgo". Una solicitud de "riesgo aceptable" no necesita justificar un rechazo porque no fue rechazada.

**¿El modelo está listo para producción?**
No sin antes mitigar el hallazgo de fairness (sección "Fairness" de esta guía) y las limitaciones documentadas en `docs/model-card.md`.

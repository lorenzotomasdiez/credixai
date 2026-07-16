# Gestión integral del riesgo de crédito (síntesis BCRA)

> Nota: este documento es un resumen sintetizado con fines educativos, escrito para el corpus RAG de CrediXAI.
> No es el texto normativo oficial ni un sustituto de las Comunicaciones "A" del BCRA vigentes; simplifica y reformula conceptos generales de gestión de riesgo crediticio para entidades financieras argentinas.

## Políticas de originación

Toda entidad financiera debe contar con políticas de originación de crédito aprobadas por su directorio, que definan como mínimo:

- Los criterios de elegibilidad del solicitante (capacidad de pago, historial crediticio, relación cuota-ingreso).
- Los límites de exposición por deudor y por segmento de cartera.
- Los mecanismos de verificación de la información declarada por el solicitante.
- La segregación de funciones entre quien origina, quien aprueba y quien audita la cartera.

Estas políticas deben revisarse periódicamente y actualizarse cuando cambien las condiciones macroeconómicas o el perfil de riesgo de la cartera.

## Evaluación de capacidad de pago

La evaluación de capacidad de pago no puede basarse exclusivamente en el patrimonio o las garantías ofrecidas: debe priorizar el flujo de ingresos genuino y sostenible del solicitante.
Se recomienda el uso de ratios como cuota-ingreso y deuda-ingreso, con umbrales prudenciales que la entidad debe justificar y documentar.

## Clasificación de deudores y previsionamiento

Los deudores se clasifican según su situación de pago (situación normal, riesgo bajo, riesgo medio, riesgo alto, irrecuperable), y cada categoría exige un nivel mínimo de previsión por incobrabilidad.
El scoring automatizado que asiste esta clasificación debe:

- Ser auditable: la entidad debe poder reconstruir por qué un caso recibió una clasificación determinada.
- Evitar el uso directo de atributos protegidos (género, edad, u otros proxies de discriminación) como criterio de rechazo, aun cuando esos atributos puedan usarse internamente con fines de auditoría de equidad.
- Contar con un proceso de revisión humana para los casos límite o de alto impacto.

## Gestión integral de riesgos

La gestión de riesgo crediticio se enmarca dentro de un esquema más amplio de gestión integral de riesgos (crédito, mercado, liquidez, operacional), que exige:

- Un comité de riesgos con reporte directo al directorio.
- Backtesting periódico de los modelos de scoring contra el desempeño real de la cartera.
- Planes de contingencia ante deterioro súbito de la calidad de cartera (stress testing).

## Transparencia frente al solicitante

Cuando una solicitud de crédito es rechazada o recibe condiciones desfavorables por un modelo automatizado, la entidad debe poder comunicar al solicitante las razones principales de esa decisión, en lenguaje claro y sin necesidad de exponer el funcionamiento interno del modelo ni atributos protegidos.

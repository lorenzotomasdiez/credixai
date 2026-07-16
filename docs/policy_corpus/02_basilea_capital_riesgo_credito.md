# Riesgo de crédito y gestión de modelos (síntesis marco de Basilea)

> Nota: este documento es un resumen sintetizado con fines educativos, escrito para el corpus RAG de CrediXAI.
> No es el texto oficial del Comité de Supervisión Bancaria de Basilea (Basilea III/IV); simplifica y reformula conceptos generales sobre capital, riesgo de crédito y gestión de modelos.

## Capital y riesgo de crédito (Pilar 1)

El marco de Basilea exige que las entidades financieras mantengan capital regulatorio proporcional al riesgo de sus activos, medido mediante activos ponderados por riesgo (RWA).
Para el riesgo de crédito, las entidades pueden usar:

- **Enfoque estandarizado**: ponderaciones fijas según categoría de contraparte y calificación externa.
- **Enfoque basado en calificaciones internas (IRB)**: la propia entidad estima la probabilidad de default (PD), la pérdida en caso de default (LGD) y la exposición al default (EAD), sujeto a validación del supervisor.

Un modelo de scoring como el de CrediXAI, que estima P(default), es conceptualmente análogo al componente PD de un enfoque IRB, aunque este proyecto no está certificado para uso regulatorio de capital.

## Gestión de riesgo de modelo (Model Risk Management)

Independientemente del enfoque de capital, el Comité de Basilea y los supervisores nacionales esperan que las entidades gestionen el riesgo de modelo con:

- **Validación independiente**: el equipo que valida un modelo de crédito no debe ser el mismo que lo desarrolló.
- **Documentación de supuestos y limitaciones**: incluyendo el rango de datos de entrenamiento, la población para la que el modelo es válido, y las condiciones bajo las cuales su desempeño se degrada.
- **Monitoreo continuo de desempeño**: comparación periódica entre las predicciones del modelo y los resultados observados (backtesting), y detección de drift en la distribución de los datos de entrada.
- **Planes de contingencia**: procesos manuales o modelos alternativos disponibles si el modelo principal deja de ser confiable.

## Pilar 3: divulgación

El Pilar 3 del marco de Basilea exige divulgación pública sobre la exposición al riesgo y las metodologías usadas para medirlo, con el objetivo de que el mercado pueda evaluar la solidez de una entidad.
Trasladado a un sistema de scoring individual, este principio se traduce en la exigencia de explicabilidad: cada decisión de crédito debe poder explicarse en términos comprensibles, no solo justificarse con una puntuación numérica opaca.

## Riesgo de discriminación algorítmica

Aunque el marco de Basilea no regula específicamente la equidad algorítmica, la práctica supervisora internacional (y las guías de gestión de riesgo de modelo de varios bancos centrales) considera que un modelo que amplifica disparidades preexistentes por atributos protegidos constituye un riesgo de modelo no gestionado, con impacto reputacional, legal y de riesgo de conducta.
Se espera que la entidad audite y documente estas disparidades, y las mitigue o las justifique explícitamente antes de desplegar el modelo en producción.

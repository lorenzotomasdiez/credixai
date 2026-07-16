# Adverse action y reason codes (síntesis Comentario CFPB a §1002.9)

> Nota: este documento es un resumen sintetizado con fines educativos, escrito para el corpus RAG de CrediXAI.
> No es el texto oficial del Regulation B (Equal Credit Opportunity Act) ni del Comentario Oficial del CFPB a §1002.9; simplifica y reformula el estándar de "adverse action notice" aplicado a scoring automatizado.

## Qué es una "adverse action"

Se considera adverse action a cualquier decisión que deniega una solicitud de crédito, la aprueba en condiciones menos favorables a las solicitadas, o cancela una línea de crédito existente.
Cuando la decisión se toma total o parcialmente en base a un modelo automatizado (incluyendo modelos de machine learning), la entidad sigue obligada a poder explicar las razones principales de esa decisión al solicitante.

## Reason codes: principios

Los reason codes (razones de la decisión adversa) que se comunican al solicitante deben cumplir:

- **Especificidad**: razones genéricas como "no cumple con nuestros criterios internos" no son aceptables; deben referirse a factores concretos (por ejemplo, "relación cuota-ingreso elevada", "historial de pagos reciente").
- **Cantidad acotada**: la práctica de mercado y el Comentario Oficial recomiendan comunicar como máximo entre 3 y 4 razones principales, para que la comunicación sea comprensible en vez de exhaustiva.
- **Exclusión de atributos protegidos**: las razones comunicadas nunca deben mencionar género, edad, estado civil, origen u otros atributos protegidos, aun cuando el modelo los haya usado o correlacionado internamente. La comunicación debe basarse en factores accionables y legítimos (ingresos, historial de pago, nivel de endeudamiento).
- **Trazabilidad al modelo**: las razones comunicadas deben derivarse de un método de atribución de importancia validado (por ejemplo, SHAP u otro método de explicabilidad local), no de una heurística separada del modelo que efectivamente decidió.

## Modelos de machine learning y explicabilidad

El uso de modelos complejos (ensambles de árboles, redes neuronales) no exime a la entidad de cumplir con la obligación de reason codes.
El Comentario Oficial reconoce que estos modelos pueden generar explicaciones válidas siempre que:

1. El método de atribución (SHAP, LIME u otro) refleje efectivamente la contribución de cada variable a la predicción del modelo, y no una aproximación no verificada.
2. La entidad pueda demostrar, ante una auditoría o reclamo, que las razones comunicadas corresponden a la predicción real del modelo para ese caso, y no a un resumen genérico o promedio.

## Relación con contrafácticos

Una explicación contrafáctica ("si su relación cuota-ingreso bajara a X, la decisión cambiaría") complementa, pero no reemplaza, a los reason codes.
Es una buena práctica de transparencia adicional porque le da al solicitante una vía accionable para mejorar su situación crediticia futura, aunque no es un requisito explícito del estándar de adverse action.

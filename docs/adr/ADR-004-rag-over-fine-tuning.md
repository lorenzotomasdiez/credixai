# ADR-004: RAG sobre fine-tuning para las explicaciones en lenguaje natural

## Status
Accepted

## Context
El top 5 de architectural characteristics prioriza Explainability y Auditability como drivers estructurales (`docs/architecture-characteristics.md`).
RNF-1 exige que toda decisión sea trazable a política (normativa BCRA, Basilea, políticas de crédito sintéticas).
RNF-8 exige groundedness medible (RAGAS faithfulness ≥ 0.90, answer relevancy ≥ 0.85).
La normativa y las políticas de crédito cambian con el tiempo (freshness), y un modelo fine-tuneado quedaría desactualizado sin reentrenamiento.
El riesgo de alucinaciones en explicaciones de cara al cliente/regulador está identificado como riesgo de impacto alto (PRD §10).
El presupuesto del proyecto está acotado a decenas de dólares en APIs de LLM (RNF-6).

## Decision
Se adopta **RAG (Retrieval-Augmented Generation)** sobre el corpus normativo/de políticas, en lugar de fine-tuning de un LLM, para generar explicaciones en lenguaje natural.

## Consequences
- Cada afirmación generada puede citarse a un documento y fragmento concreto (RF-5), lo que hace posible medir faithfulness con RAGAS y satisface la Auditability priorizada.
- Actualizar el corpus (nueva normativa, nuevas políticas) no requiere reentrenar ni re-desplegar un modelo, solo re-indexar documentos.
- El costo se mantiene acotado a inferencia (con prompt caching y batch), evitando el costo de entrenamiento y de curar un dataset de fine-tuning.
- Se renuncia a que el modelo "internalice" el conocimiento normativo de forma implícita; toda respuesta depende de la calidad del retrieval (hybrid search + reranking), lo que traslada el riesgo de fallo de la generación al retrieval.
- Queda pendiente de diseño un mecanismo de evaluación continua (RAGAS en CI) para detectar degradación de faithfulness cuando el corpus o los prompts cambien.

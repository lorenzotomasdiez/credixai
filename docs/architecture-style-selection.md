# Architecture Style Selection — CrediXAI

**Fuente de requisitos:** `docs/architecture-characteristics.md` (top 5), `prd.md` §7.1 (diagrama de componentes).
**Estado:** Borrador para discusión.

---

## 1. Estilos evaluados

| Estilo | Modularity | Deployability | Testability | Simplicity | Scalability | Cost |
|---|---|---|---|---|---|---|
| Layered (n-tier) | Baja | Baja | Baja | Alta | Baja | Bajo |
| Pipeline (pipes-and-filters) | Media | Media | Media | Alta | Baja | Bajo |
| Microkernel (plug-in) | Media-Alta | Media | Media | Media | Baja | Medio |
| Service-based | Alta | Media-Alta | Media-Alta | Media-Alta | Media | Medio |
| Event-driven | Alta | Alta | Media | Baja | Alta | Alto |
| Space-based | Media | Alta | Baja | Baja | Muy alta | Alto |
| Microservices | Muy alta | Alta | Alta | Baja | Alta | Alto |

---

## 2. Evaluación contra el top 5 de characteristics

| Characteristic | Estilos que mejor la satisfacen | Estilos que la penalizan |
|---|---|---|
| Explainability | Todos por igual (no es una dimensión estructural, depende del componente XAI, no del estilo) | — |
| Auditability | Service-based, Microservices (límites claros por servicio → logging/tracing por dominio) | Layered (todo atraviesa las mismas capas, difícil aislar auditoría por dominio) |
| Fairness | Todos por igual (depende de la lógica de negocio, no del estilo) | — |
| Modularity | Microservices, Service-based, Microkernel | Layered, Event-driven (topología broker acopla implícitamente vía eventos) |
| Reproducibility | Pipeline, Service-based (unidades de despliegue acotadas y testeables aisladamente) | Space-based, Event-driven (estado distribuido, más difícil de reproducir determinísticamente) |

---

## 3. Restricciones del proyecto que afectan la elección

| Restricción | Origen | Implicancia |
|---|---|---|
| Un solo desarrollador | Contexto del proyecto | Descarta estilos de alto costo operativo (Microservices, Space-based, Event-driven) |
| Sin tráfico productivo real | PRD §3.3 (no-objetivo) | Elasticidad/escalabilidad de tráfico no son drivers |
| Entrega por fases con núcleo obligatorio + extensiones opcionales | PRD §9 | Requiere límites de módulo claros para que una fase no bloquee a otra |
| Volumen de datos alto pero procesamiento batch, no streaming | PRD §7.2 | El sub-flujo de feature engineering es naturalmente secuencial |

---

## 4. Estilo seleccionado

**Macro-estilo: Service-based architecture.**
Componentes del sistema agrupados en servicios de dominio de grano grueso, cada uno con responsabilidad única, expuestos a través de FastAPI:

| Servicio de dominio | Componente(s) del PRD (§7.1) |
|---|---|
| Data & Feature Store | Feature engineering, feature store versionado (DVC) |
| ML Scoring | XGBoost/LogReg, calibración, MLflow registry |
| XAI | SHAP, contrafácticos (DiCE), fairness (Fairlearn/AIF360) |
| Knowledge (RAG) | Chunking, embeddings, vector DB, hybrid search, reranking |
| Agent Orchestrator | LangGraph copiloto, tools, evaluator-optimizer |
| Serving | FastAPI (`/score`, `/explain`, `/copilot`), Streamlit |

**Infraestructura transversal (no es un servicio de dominio):** MLflow, Evidently, Langfuse, CI/CD, Docker. Soportan a todos los servicios de dominio por igual; no responden a una pregunta del negocio crediticio.

**Micro-estilo anidado: Pipeline (pipes-and-filters)** dentro del servicio Data & Feature Store, para el flujo secuencial ingesta → limpieza → agregación relacional → features.

---

## 5. Justificación resumida

| Criterio | Resultado |
|---|---|
| Modularity (característica más priorizada) | Service-based la satisface sin el costo operativo de Microservices |
| Auditability | Límites de servicio permiten logging/tracing por dominio |
| Reproducibility | Cada servicio es una unidad de build/test aislable |
| Restricción de equipo (1 dev) y timeline académico | Descarta estilos distribuidos de alto costo (Event-driven, Space-based, Microservices) |
| Alineación con PRD §7.1 | El diagrama de subgraphs del PRD ya refleja esta separación por dominio |

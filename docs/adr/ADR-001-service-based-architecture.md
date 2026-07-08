# ADR-001: Adopción de Service-based Architecture como macro-estilo

## Status
Accepted

## Context
El top 5 de architectural characteristics (`docs/architecture-characteristics.md`) prioriza Modularity, Auditability y Reproducibility como drivers estructurales.
El proyecto se desarrolla con un único desarrollador, sin tráfico productivo real, y con un cronograma académico fijo (PRD §9).
El PRD (§7.1) ya describe el sistema como subgraphs separados por dominio: Data, ML, RAG, Agent, Serving, Ops.
Los estilos candidatos evaluados fueron Layered, Pipeline, Microkernel, Service-based, Event-driven, Space-based y Microservices (`docs/architecture-style-selection.md`).

## Decision
Se adopta **Service-based architecture** como macro-estilo del sistema, con seis servicios de dominio de grano grueso (Data & Feature Store, ML Scoring, XAI, Knowledge/RAG, Agent Orchestrator, Serving) expuestos a través de FastAPI, más una capa de infraestructura transversal (MLflow, Evidently, Langfuse, CI/CD, Docker) que no constituye un servicio de dominio.
Dentro del servicio Data & Feature Store se anida un estilo Pipeline (pipes-and-filters) para el flujo secuencial de ingesta y feature engineering.

## Consequences
- Cada servicio de dominio puede desarrollarse, testearse y entregarse de forma independiente, alineado con el principio de gestión de scope del PRD (núcleo académico obligatorio vs. extensiones opcionales).
- Se descartan explícitamente Microservices, Event-driven y Space-based por su costo operativo, no justificado para un solo desarrollador sin necesidad de escalar tráfico.
- Se descarta Layered puro por su baja modularidad y testability, que dificultarían la Auditability priorizada.
- Al no haber despliegue independiente real de cada servicio (todo corre en los mismos contenedores/entorno del proyecto), los límites entre servicios son lógicos (paquetes/módulos), no físicos (procesos o repos separados).

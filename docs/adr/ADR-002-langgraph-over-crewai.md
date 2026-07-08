# ADR-002: LangGraph sobre CrewAI para el copiloto agéntico

## Status
Accepted

## Context
El copiloto de analista de crédito (RF-6) necesita orquestar múltiples tools (scoring, SHAP, RAG) en un patrón orchestrator-workers, con un loop evaluator-optimizer que revise el memo antes de entregarlo.
El AI Act (Art. 14, citado en PRD §2.3) exige supervisión humana con capacidad de override antes de que una decisión de alto riesgo se emita.
El top 5 de architectural characteristics prioriza Auditability y Modularity como drivers estructurales (`docs/architecture-characteristics.md`).
Los frameworks evaluados para esta capa fueron LangGraph y CrewAI.

## Decision
Se adopta **LangGraph** como framework de orquestación del copiloto agéntico, en lugar de CrewAI.

## Consequences
- Se obtiene control de estado explícito y checkpointing nativo, lo que permite implementar el punto de revisión humana (human-in-the-loop) antes de "emitir" el memo, requerido por Art. 14 del AI Act.
- La integración nativa de observabilidad (trazas por nodo del grafo) favorece la Auditability priorizada, al quedar cada paso del razonamiento del agente trazado individualmente.
- LangGraph corre como librería standalone, sin dependencia del framework LangChain completo, lo que reduce superficie de dependencias del proyecto.
- Se renuncia a la simplicidad de la abstracción de "roles" de CrewAI (más rápida de prototipar para equipos que priorizan velocidad de desarrollo por sobre control granular).
- El patrón orchestrator-workers y el loop evaluator-optimizer deben modelarse explícitamente como grafo de estados, lo que implica una curva de aprendizaje mayor que la de CrewAI.

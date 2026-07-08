# CrediXAI

Sistema de scoring crediticio explicable, con RAG sobre normativa BCRA/Basilea y copiloto agéntico, desarrollado como Práctica Profesionalizante (Tecnicatura Superior en Data Science, Teclab).

- Producto/alcance: `prd.md`
- Decisiones de arquitectura: `docs/`

## Estructura del repo

```
data/           # datos crudos y procesados (versionados con DVC, no con git)
notebooks/      # notebooks por tarea (EDA, features, clustering, XAI...)
src/credixai/   # paquete Python reutilizable
models/         # artefactos de modelos entrenados
docs/           # architecture characteristics, ADRs, diagramas
tests/          # tests automatizados
```

## Setup

```
uv sync
```

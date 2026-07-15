# CrediXAI

Sistema de scoring crediticio explicable, con RAG sobre normativa BCRA/Basilea y copiloto agéntico, desarrollado como Práctica Profesionalizante (Tecnicatura Superior en Data Science, Teclab).

- Producto/alcance: `prd.md`
- Decisiones de arquitectura: `docs/`

## Estructura del repo

```
data/           # datos crudos y procesados (versionados con DVC, no con git)
notebooks/      # notebooks por tarea (EDA, features, clustering, XAI...)
scripts/        # entrypoints reproducibles por tarea (ej. 02_features.py)
src/credixai/   # paquete Python reutilizable
models/         # artefactos de modelos entrenados
docs/           # architecture characteristics, ADRs, diagramas
tests/          # tests automatizados
```

## Setup

```
uv sync
```

## Datos

Los datos (`data/raw`, `data/processed`) se versionan con DVC, no con git; el repo solo trackea `data/raw.dvc` y `data/processed.dvc` (metadatos con hash).
Por ahora el cache de DVC es local (sin remote configurado): quien clone el repo necesita colocar el dataset original de Kaggle en `data/raw/` y correr `dvc add data/raw data/processed` para que los hashes coincidan, o `dvc checkout` si ya tiene acceso al mismo cache local.

```
dvc status   # ver si data/raw o data/processed cambiaron respecto al .dvc trackeado
dvc add data/raw data/processed   # re-trackear después de un cambio
```

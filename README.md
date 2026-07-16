# CrediXAI

Un modelo de scoring crediticio que, además de predecir riesgo de default, audita si esa predicción es equitativa entre grupos, y lo demuestra: encontró que el modelo amplifica al doble la disparidad real de default por género y edad, un hallazgo que un scoring de caja negra nunca hubiera expuesto.

Desarrollado como Práctica Profesionalizante (Tecnicatura Superior en Data Science, Teclab), sobre el dataset Home Credit Default Risk (Kaggle).

## Resultado técnico

- **ROC-AUC 0.7815** en holdout (0.7802 ± 0.0032 en validación cruzada de 5 folds), +1.6 puntos sobre el baseline de Regresión Logística.
- **Explicaciones SHAP estables:** Kendall tau = 0.9917 ± 0.0010 sobre 30 remuestreos bootstrap del ranking de importancia.
- **Auditoría de fairness cuantificada:** el modelo amplifica la brecha real de default por género y edad en aproximadamente 2x (detalle en `docs/model-card.md` y `docs/informe-final.md` sección 5.4).
- **Reason codes de adverse action** que excluyen atributos protegidos de la comunicación al solicitante, aun cuando el modelo los usa internamente.

## Arquitectura

```mermaid
flowchart LR
    subgraph nucleo["Núcleo académico (implementado)"]
        data["Data & Feature Store\nPandas + DVC"] --> ml["ML Scoring\nXGBoost"]
        ml --> xai["XAI\nSHAP + Fairlearn"]
        ml --> serving["Serving\nStreamlit"]
        xai --> serving
        data --> cluster["Segmentación\nK-Means"]
        cluster --> serving
    end
    subgraph impl_ext["Extensiones de portfolio (implementadas)"]
        api["API REST\nFastAPI"]
        docker["Contenedores\nDocker"]
        cicd["CI/CD\nGitHub Actions"]
        rag["RAG normativo\nQdrant + OpenRouter"]
        agent["Copiloto\nLangGraph"]
    end
    ml --> api
    xai --> api
    api --> docker
    serving --> docker
    docker --> cicd
    rag --> api
    agent --> api
    agent --> rag
```

Diagrama de componentes completo (incluidas las extensiones planificadas) y decisiones de arquitectura (ADRs) en `docs/`.
El estilo elegido es service-based, justificado en `docs/architecture-style-selection.md`.

## Documentación

| Documento | Contenido |
|---|---|
| `docs/informe-final.md` | Metodología y resultados completos, tarea por tarea (EDA, features, clustering, modelado, XAI/fairness, dashboard) |
| `docs/informe-ejecutivo.md` | Resumen para público no técnico |
| `docs/model-card.md` | Performance, fairness, limitaciones y uso previsto del modelo |
| `docs/adr/` | Decisiones de arquitectura (ADRs) |
| `docs/architecture-characteristics.md`, `docs/architecture-style-selection.md` | Trade-offs de diseño |
| `prd.md` | Producto y alcance completo del proyecto |

## Estructura del repo

```
data/           # datos crudos y procesados (versionados con DVC, no con git)
notebooks/      # notebooks por tarea (EDA, features, clustering, XAI...)
scripts/        # entrypoints reproducibles por tarea (ej. 02_features.py)
src/credixai/   # paquete Python reutilizable
app/            # dashboard Streamlit (entrypoint delgado sobre src/credixai)
models/         # artefactos de modelos entrenados
docs/           # informes, model card, architecture characteristics, ADRs, diagramas
tests/          # tests automatizados
```

## Setup

```
uv sync
```

## Tests

Tests unitarios sobre `src/credixai` (features, clustering, modeling, explainability), con datos sintéticos generados en el propio test: no requieren el dataset de Kaggle.

```
uv run ruff check .
uv run pytest
uv run pytest --cov=credixai --cov=app --cov-report=term-missing   # con cobertura
```

Ambos pasos corren en CI en cada push/PR a `main` (`.github/workflows/ci.yml`), junto con el build de las dos imágenes Docker.

`src/credixai` (la lógica reutilizable) está al 100% de cobertura, salvo `dashboard.py` al 98% (la única línea sin cubrir lee el parquet real, un límite de I/O verificado manualmente).
`app/dashboard.py` y la inicialización de la API (`get_service()`) no están cubiertos por la suite rápida a propósito: requieren el dataset real y un runtime completo (Streamlit/Uvicorn); se verifican manualmente contra datos reales, documentado en `docs/informe-final.md`.

Los tests bajo `tests/rag/` que llaman a OpenRouter real están marcados `integration` y excluidos por default (`addopts = "-m 'not integration'"` en `pyproject.toml`); se corren manualmente con `uv run pytest -m integration` cuando hay `OPENROUTER_API_KEY` disponible.

## Cómo correr

Con los datos ya versionados (ver sección "Datos" abajo), reproducir el pipeline completo en orden:

```
uv run python scripts/02_features.py
uv run python scripts/03_clustering.py
uv run python scripts/04_modeling.py
uv run python scripts/05_explainability.py
```

Para explorar los resultados de forma interactiva (dashboard con métricas, segmentación, fairness y explicación por solicitud):

```
uv run streamlit run app/dashboard.py
```

Para levantar la API REST (`/score`, `/explain`, docs interactivas en `/docs`):

```
uv run uvicorn app.api:app --reload
```

## Docker

API y dashboard corren en contenedores separados (un proceso por imagen, alineado con el estilo service-based del proyecto).
`data/processed` no se hornea en la imagen porque se versiona con DVC, no con git: se monta como volumen de solo lectura en runtime.

```
docker compose up --build
```

Deja la API en `http://localhost:8000` (`/health`, `/score/{id}`, `/explain/{id}`, docs en `/docs`) y el dashboard en `http://localhost:8501`.

El contrato de las imágenes (build exitoso, healthcheck en verde con datos reales montados) se valida con:

```
bash tests/smoke/docker_smoke.sh
```

## RAG normativo

`POST /rag/query` responde preguntas de política/normativa (BCRA, Basilea, adverse action, política interna) citando siempre documento y fragmento fuente.
El corpus (`docs/policy_corpus/`) son resúmenes sintetizados con fines educativos, no el texto normativo oficial.
Retrieval híbrido (Qdrant + BM25, fusionados con Reciprocal Rank Fusion) y reranking listwise, ambos con un único provider LLM (OpenRouter).

Requiere `OPENROUTER_API_KEY` (copiar `.env.example` a `.env`) y Qdrant corriendo:

```
docker compose up -d qdrant
uv run python scripts/06_rag_ingest.py     # ingesta el corpus (una vez, o tras cambiar docs/policy_corpus/)
uv run uvicorn app.api:app --reload
```

Evaluación con RAGAS (faithfulness, answer relevancy), resultado y limitaciones documentadas en `docs/informe-final.md` sección 8.5:

```
uv run python scripts/07_rag_eval.py
```

## Copiloto agentico

`POST /copilot/memo/{sk_id_curr}` investiga una solicitud y redacta un borrador de memo crediticio con reason codes y citas de politica.
Orquestador LangGraph con tool-calling real (patron orchestrator-workers): decide dinamicamente que tools llamar segun el caso (`score_application`, y si es alto riesgo tambien `explain_shap` y `retrieve_policy`), todas via HTTP contra esta misma API en vez de imports directos.
Un loop evaluator-optimizer (precheck deterministico + juez LLM) revisa el memo antes de entregarlo: si no pasa, se redacta una vez mas con el feedback; si vuelve a fallar, la respuesta queda en `status: needs_human_review` en vez de reintentar indefinidamente.

Mismos requisitos que RAG normativo (Qdrant + `OPENROUTER_API_KEY`), mas el modelo ya entrenado:

```
docker compose up -d qdrant
uv run python scripts/06_rag_ingest.py
uv run uvicorn app.api:app --reload
curl -X POST http://localhost:8000/copilot/memo/100002
```

## Datos

Los datos (`data/raw`, `data/processed`) se versionan con DVC, no con git; el repo solo trackea `data/raw.dvc` y `data/processed.dvc` (metadatos con hash).
Por ahora el cache de DVC es local (sin remote configurado): quien clone el repo necesita colocar el dataset original de Kaggle en `data/raw/` y correr `dvc add data/raw data/processed` para que los hashes coincidan, o `dvc checkout` si ya tiene acceso al mismo cache local.

```
dvc status   # ver si data/raw o data/processed cambiaron respecto al .dvc trackeado
dvc add data/raw data/processed   # re-trackear después de un cambio
```

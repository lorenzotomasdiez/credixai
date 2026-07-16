.PHONY: setup up down

# Deja el proyecto listo de punta a punta: dependencias, dataset, features,
# clustering, modelo, explicabilidad e ingesta del RAG normativo.
# Requiere antes: credenciales de Kaggle (ver scripts/00_download_data.py)
# y un .env con OPENROUTER_API_KEY (ver .env.example).
setup:
	uv sync --all-extras --dev
	uv run python scripts/00_download_data.py
	uv run dvc add data/raw
	uv run python scripts/02_features.py
	uv run python scripts/03_clustering.py
	uv run python scripts/04_modeling.py
	uv run python scripts/05_explainability.py
	uv run dvc add data/processed
	docker compose up -d qdrant
	uv run python scripts/06_rag_ingest.py

# Levanta todo el stack containerizado: API, dashboard, Qdrant y Langfuse
# self-hosteado (Postgres, ClickHouse, Redis, MinIO) para observabilidad LLM.
up:
	docker compose up -d --build

down:
	docker compose down

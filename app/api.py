"""API REST (RF-8, paso 2 de prd.md 9.1) para CrediXAI.

Entrypoint delgado sobre credixai.api.ScoringService y
credixai.rag.pipeline.RagPipeline, mismo patron de separacion
logica/entrypoint que app/dashboard.py. Expone /score y /explain sobre el
modelo final de la Tarea 4, reutilizando SHAP y reason codes de la Tarea 5
sin reimplementar nada, y /rag/query (RF-5, paso 5 de prd.md 9.1) sobre el
corpus normativo.

Requiere haber corrido antes scripts/02_features.py y, para /rag/query,
scripts/06_rag_ingest.py con Qdrant corriendo y OPENROUTER_API_KEY seteada.

Uso:
    uv run fastapi run app/api.py
"""

import json
import os
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from credixai.api import ScoringService
from credixai.dashboard import load_features, train_full_model
from credixai.rag.chunking import Chunk
from credixai.rag.generation import PolicyAnswerer
from credixai.rag.openrouter_client import OpenRouterClient
from credixai.rag.pipeline import RagPipeline
from credixai.rag.reranker import LLMReranker
from credixai.rag.retrieval import PolicyRetriever
from credixai.rag.sparse_index import BM25Index
from credixai.rag.vector_store import QdrantStore

load_dotenv()

RAG_CHUNKS_PATH = "models/rag/chunks.json"
RAG_COLLECTION_NAME = "policy_chunks"
RAG_EMBEDDING_SIZE = 1536

app = FastAPI(
    title="CrediXAI API",
    description="Scoring crediticio explicable: probabilidad de default y explicacion SHAP por solicitud.",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_service() -> ScoringService:
    features = load_features()
    bundle = train_full_model(features)
    return ScoringService(bundle)


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RagPipeline:
    with open(RAG_CHUNKS_PATH) as f:
        chunks = [Chunk(**c) for c in json.load(f)]
    chunks_by_id = {c.chunk_id: c for c in chunks}

    client = OpenRouterClient()
    qdrant_client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    vector_store = QdrantStore(qdrant_client, collection_name=RAG_COLLECTION_NAME, vector_size=RAG_EMBEDDING_SIZE)
    bm25_index = BM25Index(chunks)
    reranker = LLMReranker(chat_fn=client.chat)

    retriever = PolicyRetriever(
        embed_fn=client.embed,
        vector_store=vector_store,
        bm25_index=bm25_index,
        chunks_by_id=chunks_by_id,
        reranker=reranker.rerank,
    )
    answerer = PolicyAnswerer(chat_fn=client.chat)
    return RagPipeline(retriever=retriever, answerer=answerer)


class ScoreResponse(BaseModel):
    sk_id_curr: int
    probability: float
    threshold: float
    decision: str


class ExplainResponse(BaseModel):
    sk_id_curr: int
    probability: float
    threshold: float
    decision: str
    base_value: float
    shap_values: list[float]
    feature_names: list[str]
    reason_codes: list[str]


class RagQueryRequest(BaseModel):
    question: str = Field(min_length=1)


class RagCitationResponse(BaseModel):
    doc_title: str
    chunk_id: str
    snippet: str


class RagQueryResponse(BaseModel):
    answer: str
    citations: list[RagCitationResponse]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/score/{sk_id_curr}", response_model=ScoreResponse)
def score(sk_id_curr: int, service: ScoringService = Depends(get_service)) -> ScoreResponse:
    try:
        result = service.score(sk_id_curr)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"SK_ID_CURR {sk_id_curr} no encontrado")
    return ScoreResponse(**result.__dict__)


@app.get("/explain/{sk_id_curr}", response_model=ExplainResponse)
def explain(
    sk_id_curr: int,
    top_n: int = Query(default=4, ge=1),
    service: ScoringService = Depends(get_service),
) -> ExplainResponse:
    try:
        result = service.explain(sk_id_curr, top_n=top_n)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"SK_ID_CURR {sk_id_curr} no encontrado")
    return ExplainResponse(**result.__dict__)


@app.post("/rag/query", response_model=RagQueryResponse)
def rag_query(request: RagQueryRequest, pipeline: RagPipeline = Depends(get_rag_pipeline)) -> RagQueryResponse:
    result = pipeline.query(request.question)
    return RagQueryResponse(
        answer=result.answer,
        citations=[RagCitationResponse(**c.__dict__) for c in result.citations],
    )

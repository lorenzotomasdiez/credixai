"""Paso 5 (prd.md 9.1): evaluacion RAGAS del RAG normativo (RNF-8).

Corre el pipeline real (retrieval + generacion) sobre un set de preguntas
escritas a mano, ancladas en el corpus (docs/policy_corpus), y mide con
RAGAS:
- faithfulness: la respuesta no dice nada que no este soportado por los
  chunks recuperados (umbral RNF-8: >= 0.90).
- answer relevancy: la respuesta efectivamente contesta la pregunta hecha
  (umbral RNF-8: >= 0.85).

Requiere OPENROUTER_API_KEY y Qdrant corriendo con el corpus ya ingestado
(scripts/06_rag_ingest.py).

Uso:
    uv run python scripts/07_rag_eval.py
"""

import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from credixai.rag.chunking import Chunk
from credixai.rag.generation import PolicyAnswerer
from credixai.rag.openrouter_client import OpenRouterClient
from credixai.rag.reranker import LLMReranker
from credixai.rag.retrieval import PolicyRetriever
from credixai.rag.sparse_index import BM25Index
from credixai.rag.vector_store import QdrantStore

CHUNKS_PATH = "models/rag/chunks.json"
COLLECTION_NAME = "policy_chunks"
EMBEDDING_SIZE = 1536
FAITHFULNESS_THRESHOLD = 0.90
ANSWER_RELEVANCY_THRESHOLD = 0.85

EVAL_SET = [
    "¿Cuántos reason codes como máximo se comunican al solicitante y por qué?",
    "¿Pueden los reason codes mencionar género o edad del solicitante?",
    "¿Qué es el PSI y qué umbral dispara una alerta de drift?",
    "¿Qué enfoques existen en el marco de Basilea para medir el riesgo de credito?",
    "¿Qué es el Pilar 3 de Basilea y como se relaciona con la explicabilidad?",
    "¿Qué debe documentar una entidad segun BCRA cuando clasifica deudores con un modelo automatizado?",
    "¿Cuál es el rango de referencia aceptable para statistical parity difference en CrediXAI?",
    "¿Qué pasa con una solicitud de alto riesgo cercana al umbral de decision?",
    "¿Qué diferencia hay entre el enfoque estandarizado y el enfoque IRB para riesgo de credito?",
    "¿Que exige la politica de originacion de credito segun BCRA?",
    "¿En que se debe basar principalmente la evaluacion de capacidad de pago segun BCRA?",
    "¿Que es la validacion independiente en la gestion de riesgo de modelo?",
    "¿Que metricas de fairness audita CrediXAI por grupo protegido?",
    "¿Un contrafactico reemplaza a los reason codes segun el estandar de adverse action?",
]


def _build_pipeline():
    with open(CHUNKS_PATH) as f:
        chunks = [Chunk(**c) for c in json.load(f)]
    chunks_by_id = {c.chunk_id: c for c in chunks}

    client = OpenRouterClient()
    qdrant_client = QdrantClient(url="http://localhost:6333")
    vector_store = QdrantStore(qdrant_client, collection_name=COLLECTION_NAME, vector_size=EMBEDDING_SIZE)
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
    return retriever, answerer


def _build_ragas_llm_and_embeddings(api_key: str):
    openai_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    llm = llm_factory("openai/gpt-4o-mini", client=openai_client)
    embeddings = embedding_factory("openai", "openai/text-embedding-3-small", client=openai_client)
    return llm, embeddings


async def _run_eval() -> None:
    load_dotenv()
    api_key = os.environ["OPENROUTER_API_KEY"]

    retriever, answerer = _build_pipeline()
    llm, embeddings = _build_ragas_llm_and_embeddings(api_key)
    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)

    faithfulness_scores = []
    relevancy_scores = []

    for question in EVAL_SET:
        retrieved = retriever.retrieve(question, top_k=4)
        result = answerer.answer(question, retrieved)
        contexts = [r.text for r in retrieved]

        f_score = await faithfulness.ascore(user_input=question, response=result.answer, retrieved_contexts=contexts)
        r_score = await answer_relevancy.ascore(user_input=question, response=result.answer)

        faithfulness_scores.append(f_score.value)
        relevancy_scores.append(r_score.value)

        print(f"- {question}")
        print(f"  faithfulness={f_score.value:.3f} answer_relevancy={r_score.value:.3f}")

    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
    avg_relevancy = sum(relevancy_scores) / len(relevancy_scores)

    print("\n=== RAGAS (RNF-8) ===")
    print(f"faithfulness promedio: {avg_faithfulness:.4f} (umbral >= {FAITHFULNESS_THRESHOLD})")
    print(f"answer relevancy promedio: {avg_relevancy:.4f} (umbral >= {ANSWER_RELEVANCY_THRESHOLD})")
    print(f"faithfulness OK: {avg_faithfulness >= FAITHFULNESS_THRESHOLD}")
    print(f"answer relevancy OK: {avg_relevancy >= ANSWER_RELEVANCY_THRESHOLD}")


def main() -> None:
    asyncio.run(_run_eval())


if __name__ == "__main__":
    main()

"""Orquestador de retrieval hibrido + rerank (paso 5, prd.md 9.1).

PolicyRetriever recibe todas sus dependencias inyectadas (embed_fn,
vector_store, bm25_index, chunks_by_id, reranker), mismo patron que
ScoringService recibe un bundle ya entrenado: la logica de orquestacion se
testea con stubs, y el wiring real (OpenRouter + Qdrant + LLMReranker) se
arma una sola vez en scripts/06_rag_ingest.py y app/api.py.
"""

from dataclasses import dataclass
from typing import Callable

from credixai.rag.chunking import Chunk
from credixai.rag.hybrid_search import reciprocal_rank_fusion


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    chunk_id: str
    title: str
    text: str


EmbedFn = Callable[[str], list[float]]
RerankFn = Callable[[str, list[Chunk], int], list[Chunk]]


class PolicyRetriever:
    def __init__(
        self,
        embed_fn: EmbedFn,
        vector_store,
        bm25_index,
        chunks_by_id: dict[str, Chunk],
        reranker: RerankFn,
    ):
        self._embed_fn = embed_fn
        self._vector_store = vector_store
        self._bm25_index = bm25_index
        self._chunks_by_id = chunks_by_id
        self._reranker = reranker

    def retrieve(self, query: str, top_k: int = 4, candidate_k: int = 10) -> list[RetrievedChunk]:
        query_vector = self._embed_fn(query)
        dense_hits = [chunk_id for chunk_id, _ in self._vector_store.search(query_vector, top_k=candidate_k)]
        sparse_hits = [chunk_id for chunk_id, _ in self._bm25_index.search(query, top_k=candidate_k)]

        fused = reciprocal_rank_fusion([dense_hits, sparse_hits])
        candidates = [self._chunks_by_id[chunk_id] for chunk_id, _ in fused]

        reranked = self._reranker(query, candidates, top_k)

        return [
            RetrievedChunk(doc_id=c.doc_id, chunk_id=c.chunk_id, title=c.title, text=c.text) for c in reranked
        ]

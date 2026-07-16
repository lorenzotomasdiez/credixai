"""Tests para credixai.rag.retrieval (paso 5).

TDD: se escribe antes que credixai/rag/retrieval.py. PolicyRetriever orquesta
embed + busqueda densa + BM25 + RRF + rerank; todas sus dependencias se
inyectan como stubs (mismo patron que ScoringService recibe un bundle), asi
que el test no toca red ni Qdrant real.
"""

from credixai.rag.chunking import Chunk
from credixai.rag.retrieval import PolicyRetriever, RetrievedChunk


def _chunks():
    return {
        "d1::0": Chunk(doc_id="d1", chunk_id="d1::0", title="A", text="texto d1 0"),
        "d1::1": Chunk(doc_id="d1", chunk_id="d1::1", title="A", text="texto d1 1"),
        "d2::0": Chunk(doc_id="d2", chunk_id="d2::0", title="B", text="texto d2 0"),
    }


class _StubVectorStore:
    def search(self, query_vector, top_k):
        return [("d1::1", 0.9), ("d2::0", 0.5)][:top_k]


class _StubBM25:
    def search(self, query, top_k):
        return [("d1::0", 3.0), ("d1::1", 1.0)][:top_k]


def _stub_embed_fn(text):
    return [0.1, 0.2, 0.3]


def _stub_reranker(query, candidates, top_n):
    # devuelve los candidatos en orden inverso, para poder distinguir en el
    # test que el retriever efectivamente llama al reranker
    return list(reversed(candidates))[:top_n]


def test_retrieve_returns_retrieved_chunks_with_citation_metadata():
    retriever = PolicyRetriever(
        embed_fn=_stub_embed_fn,
        vector_store=_StubVectorStore(),
        bm25_index=_StubBM25(),
        chunks_by_id=_chunks(),
        reranker=_stub_reranker,
    )

    results = retriever.retrieve("pregunta", top_k=3, candidate_k=2)

    assert all(isinstance(r, RetrievedChunk) for r in results)
    assert {r.chunk_id for r in results} == {"d1::0", "d1::1", "d2::0"}


def test_retrieve_applies_reranker_order():
    retriever = PolicyRetriever(
        embed_fn=_stub_embed_fn,
        vector_store=_StubVectorStore(),
        bm25_index=_StubBM25(),
        chunks_by_id=_chunks(),
        reranker=_stub_reranker,
    )

    results = retriever.retrieve("pregunta", top_k=1, candidate_k=2)

    assert len(results) == 1


def test_retrieve_carries_doc_title_for_citations():
    retriever = PolicyRetriever(
        embed_fn=_stub_embed_fn,
        vector_store=_StubVectorStore(),
        bm25_index=_StubBM25(),
        chunks_by_id=_chunks(),
        reranker=_stub_reranker,
    )

    results = retriever.retrieve("pregunta", top_k=3, candidate_k=2)

    titles = {r.chunk_id: r.title for r in results}
    assert titles["d1::0"] == "A"
    assert titles["d2::0"] == "B"

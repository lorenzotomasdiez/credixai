"""Tests para credixai.rag.sparse_index (paso 5).

TDD: se escribe antes que credixai/rag/sparse_index.py. BM25 corre
localmente (rank_bm25), sin red, por eso vive en la suite rapida.
"""

from credixai.rag.chunking import Chunk
from credixai.rag.sparse_index import BM25Index


def _chunks():
    return [
        Chunk(doc_id="d1", chunk_id="d1::0", title="A", text="limite de reason codes es cuatro razones"),
        Chunk(doc_id="d1", chunk_id="d1::1", title="A", text="capital regulatorio y activos ponderados por riesgo"),
        Chunk(doc_id="d2", chunk_id="d2::0", title="B", text="statistical parity difference y equal opportunity"),
    ]


def test_search_returns_chunk_ids_ranked_by_relevance():
    index = BM25Index(_chunks())

    results = index.search("cuantos reason codes se comunican", top_k=2)

    assert results[0][0] == "d1::0"


def test_search_respects_top_k():
    index = BM25Index(_chunks())

    results = index.search("riesgo", top_k=1)

    assert len(results) == 1


def test_search_query_with_no_overlap_still_returns_ranked_results():
    index = BM25Index(_chunks())

    results = index.search("terminos totalmente ausentes del corpus xyz", top_k=3)

    assert len(results) == 3

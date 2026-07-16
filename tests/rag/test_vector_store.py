"""Tests para credixai.rag.vector_store (paso 5).

TDD: se escribe antes que credixai/rag/vector_store.py. Corre contra un
QdrantClient real en modo ":memory:" (sin red, sin docker), asi que este
test valida el wiring real contra Qdrant, no un mock.
"""

from qdrant_client import QdrantClient

from credixai.rag.vector_store import QdrantStore


def _store():
    client = QdrantClient(":memory:")
    store = QdrantStore(client, collection_name="policy_chunks", vector_size=3)
    store.ensure_collection()
    return store


def test_search_returns_closest_vector_first():
    store = _store()
    store.upsert(
        ids=["a", "b", "c"],
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]],
    )

    results = store.search(query_vector=[1.0, 0.0, 0.0], top_k=2)

    assert results[0][0] == "a"
    assert {item for item, _ in results} == {"a", "c"}


def test_search_respects_top_k():
    store = _store()
    store.upsert(
        ids=["a", "b", "c"],
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]],
    )

    results = store.search(query_vector=[1.0, 0.0, 0.0], top_k=1)

    assert len(results) == 1

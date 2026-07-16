"""Wrapper delgado sobre QdrantClient para busqueda densa (paso 5).

El cliente se inyecta desde afuera: en tests apunta a Qdrant en modo
":memory:", en desarrollo/produccion apunta al servicio "qdrant" de
docker-compose.yml.
"""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class QdrantStore:
    def __init__(self, client: QdrantClient, collection_name: str, vector_size: int):
        self._client = client
        self._collection_name = collection_name
        self._vector_size = vector_size

    def ensure_collection(self) -> None:
        if self._client.collection_exists(self._collection_name):
            return
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
        )

    def upsert(self, ids: list[str], vectors: list[list[float]]) -> None:
        points = [
            PointStruct(id=_point_id(chunk_id), vector=vector, payload={"chunk_id": chunk_id})
            for chunk_id, vector in zip(ids, vectors)
        ]
        self._client.upsert(collection_name=self._collection_name, points=points)

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[str, float]]:
        hits = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=top_k,
        ).points
        return [(hit.payload["chunk_id"], hit.score) for hit in hits]


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

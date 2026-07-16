"""Paso 5: ingesta del corpus normativo al RAG.

Carga docs/policy_corpus, chunkea cada documento, embebe los chunks via
OpenRouter y los sube a Qdrant. Ademas persiste los chunks (texto +
metadata) en models/rag/chunks.json: BM25Index se reconstruye en memoria a
partir de ese archivo en cada arranque de la API, en vez de intentar
persistir el indice BM25 en si.

Requiere OPENROUTER_API_KEY (.env) y Qdrant corriendo (docker compose up qdrant,
o el default http://localhost:6333).

Uso:
    uv run python scripts/06_rag_ingest.py
"""

import json
import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from credixai.rag.chunking import chunk_document
from credixai.rag.corpus import load_corpus
from credixai.rag.openrouter_client import OpenRouterClient
from credixai.rag.vector_store import QdrantStore

CORPUS_DIR = "docs/policy_corpus"
OUT_DIR = "models/rag"
COLLECTION_NAME = "policy_chunks"
CHUNK_SIZE = 180
CHUNK_OVERLAP = 30
EMBEDDING_SIZE = 1536  # openai/text-embedding-3-small


def main() -> None:
    load_dotenv()

    documents = load_corpus(CORPUS_DIR)
    print(f"Corpus cargado: {len(documents)} documentos")

    chunks = [
        chunk
        for doc in documents
        for chunk in chunk_document(doc.doc_id, doc.title, doc.text, CHUNK_SIZE, CHUNK_OVERLAP)
    ]
    print(f"Chunking: {len(chunks)} chunks")

    client = OpenRouterClient()
    vectors = client.embed_batch([c.text for c in chunks])
    print(f"Embeddings generados: {len(vectors)} vectores de dimension {len(vectors[0])}")

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_client = QdrantClient(url=qdrant_url)
    store = QdrantStore(qdrant_client, collection_name=COLLECTION_NAME, vector_size=len(vectors[0]))
    store.ensure_collection()
    store.upsert(ids=[c.chunk_id for c in chunks], vectors=vectors)
    print(f"Subidos a Qdrant ({qdrant_url}, coleccion '{COLLECTION_NAME}')")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/chunks.json"
    with open(out_path, "w") as f:
        json.dump(
            [{"doc_id": c.doc_id, "chunk_id": c.chunk_id, "title": c.title, "text": c.text} for c in chunks],
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Chunks persistidos en {out_path}")


if __name__ == "__main__":
    main()

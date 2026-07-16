"""Chunking de documentos de politica para el corpus RAG (paso 5, prd.md 9.1).

Split por palabras con overlap, sin dependencias externas: los documentos del
corpus son texto plano corto, y un tokenizador de subwords no aporta nada
aqui que el conteo de palabras no resuelva mas simple.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str
    title: str
    text: str


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap debe ser menor que chunk_size")

    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_document(doc_id: str, title: str, text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    return [
        Chunk(doc_id=doc_id, chunk_id=f"{doc_id}::{i}", title=title, text=piece)
        for i, piece in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=overlap))
    ]

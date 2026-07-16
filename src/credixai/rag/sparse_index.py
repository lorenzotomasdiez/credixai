"""Indice BM25 (busqueda por keywords) sobre los chunks del corpus (paso 5)."""

import re

from rank_bm25 import BM25Okapi

from credixai.rag.chunking import Chunk


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self._chunk_ids = [c.chunk_id for c in chunks]
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._chunk_ids, scores), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k]

"""Reranking listwise via LLM (paso 5, prd.md 9.1).

Se usa un reranker basado en LLM (via OpenRouter, mismo provider que
embeddings y generacion) en vez de un cross-encoder local, para no sumar una
dependencia pesada (torch/sentence-transformers) al proyecto solo para
reranking, ya al usar el mismo pattern de chat_fn inyectado.
"""

import json
from typing import Callable

from credixai.llm_json import strip_markdown_fences
from credixai.rag.chunking import Chunk

ChatFn = Callable[[list[dict]], str]

_SYSTEM_PROMPT = (
    "Sos un sistema de reranking para un RAG de politicas de credito. "
    "Dada una pregunta y una lista de fragmentos candidatos (id y texto), "
    "devolves SOLO un JSON con la forma "
    '{"ranked_chunk_ids": ["id1", "id2", ...]}, '
    "ordenando los ids de mas a menos relevante para responder la pregunta. "
    "No incluyas texto fuera del JSON."
)


class LLMReranker:
    def __init__(self, chat_fn: ChatFn):
        self._chat_fn = chat_fn

    def rerank(self, query: str, candidates: list[Chunk], top_n: int) -> list[Chunk]:
        by_id = {c.chunk_id: c for c in candidates}
        candidates_block = "\n".join(f"- id={c.chunk_id}: {c.text}" for c in candidates)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Pregunta: {query}\n\nCandidatos:\n{candidates_block}"},
        ]

        raw_response = self._chat_fn(messages)
        try:
            parsed = json.loads(strip_markdown_fences(raw_response))
            ranked_ids = parsed["ranked_chunk_ids"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Respuesta de reranking invalida: {raw_response!r}") from exc

        ordered = [by_id[cid] for cid in ranked_ids if cid in by_id]
        seen = {c.chunk_id for c in ordered}
        ordered += [c for c in candidates if c.chunk_id not in seen]

        return ordered[:top_n]

"""Generacion de respuestas grounded con citas (RF-5, paso 5).

PolicyAnswerer arma un prompt grounded (RNF-1: toda explicacion trazable a
politica) a partir de los chunks recuperados, y deriva las citas (RF-5:
documento + fragmento) de esos mismos chunks en vez de parsear la respuesta
del LLM, para que la trazabilidad no dependa de que el LLM cite bien.
"""

from dataclasses import dataclass, field
from typing import Callable

from credixai.rag.retrieval import RetrievedChunk

ChatFn = Callable[[list[dict]], str]

_SYSTEM_PROMPT = (
    "Sos el asistente normativo de CrediXAI, un sistema de scoring crediticio explicable. "
    "Respondes preguntas sobre politica de credito y normativa (BCRA, Basilea, adverse action, "
    "politica interna) usando UNICAMENTE el contexto provisto a continuacion. "
    "Si el contexto no alcanza para responder, decilo explicitamente en vez de inventar. "
    "No reveles ni discutas atributos protegidos como criterio de decision."
)


@dataclass(frozen=True)
class Citation:
    doc_title: str
    chunk_id: str
    snippet: str


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)


def _snippet(text: str, max_chars: int = 200) -> str:
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "..."


class PolicyAnswerer:
    def __init__(self, chat_fn: ChatFn):
        self._chat_fn = chat_fn

    def answer(self, query: str, retrieved: list[RetrievedChunk]) -> AnswerResult:
        context_block = "\n\n".join(f"[{r.title}] (id={r.chunk_id})\n{r.text}" for r in retrieved)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Contexto:\n\n{context_block}\n\nPregunta: {query}"},
        ]

        answer_text = self._chat_fn(messages)

        citations = [
            Citation(doc_title=r.title, chunk_id=r.chunk_id, snippet=_snippet(r.text)) for r in retrieved
        ]

        return AnswerResult(answer=answer_text, citations=citations)

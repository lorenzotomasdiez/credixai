"""Fachada que junta PolicyRetriever + PolicyAnswerer (paso 5, prd.md 9.1).

Punto unico que app/api.py necesita conocer para RF-5, igual que
ScoringService es el unico punto que conoce para /score y /explain.
"""

from credixai.rag.generation import AnswerResult, PolicyAnswerer
from credixai.rag.retrieval import PolicyRetriever


class RagPipeline:
    def __init__(self, retriever: PolicyRetriever, answerer: PolicyAnswerer, top_k: int = 4):
        self._retriever = retriever
        self._answerer = answerer
        self._top_k = top_k

    def query(self, question: str) -> AnswerResult:
        retrieved = self._retriever.retrieve(question, top_k=self._top_k)
        return self._answerer.answer(question, retrieved)

"""Tests para credixai.rag.pipeline (paso 5)."""

from credixai.rag.generation import AnswerResult
from credixai.rag.pipeline import RagPipeline


class _StubRetriever:
    def retrieve(self, question, top_k):
        assert top_k == 4
        return ["chunk-stub"]


class _StubAnswerer:
    def answer(self, question, retrieved):
        return AnswerResult(answer=f"{question}:{retrieved}", citations=[])


def test_query_wires_retriever_output_into_answerer():
    pipeline = RagPipeline(retriever=_StubRetriever(), answerer=_StubAnswerer())

    result = pipeline.query("pregunta")

    assert result.answer == "pregunta:['chunk-stub']"

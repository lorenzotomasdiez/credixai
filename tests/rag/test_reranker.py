"""Tests para credixai.rag.reranker (paso 5, prd.md 9.1).

TDD: se escribe antes que credixai/rag/reranker.py. LLMReranker recibe
chat_fn inyectado (mismo patron que ScoringService recibe un bundle ya
entrenado), asi que el test no llama a OpenRouter: stubea chat_fn con una
respuesta JSON fija.
"""

import json

import pytest

from credixai.rag.chunking import Chunk
from credixai.rag.reranker import LLMReranker


def _candidates():
    return [
        Chunk(doc_id="d1", chunk_id="d1::0", title="A", text="texto poco relevante"),
        Chunk(doc_id="d1", chunk_id="d1::1", title="A", text="texto muy relevante a la pregunta"),
        Chunk(doc_id="d2", chunk_id="d2::0", title="B", text="texto medianamente relevante"),
    ]


def test_rerank_orders_candidates_by_llm_ranking():
    def fake_chat_fn(messages):
        return json.dumps({"ranked_chunk_ids": ["d1::1", "d2::0", "d1::0"]})

    reranker = LLMReranker(chat_fn=fake_chat_fn)

    result = reranker.rerank("pregunta de ejemplo", _candidates(), top_n=3)

    assert [c.chunk_id for c in result] == ["d1::1", "d2::0", "d1::0"]


def test_rerank_respects_top_n():
    def fake_chat_fn(messages):
        return json.dumps({"ranked_chunk_ids": ["d1::1", "d2::0", "d1::0"]})

    reranker = LLMReranker(chat_fn=fake_chat_fn)

    result = reranker.rerank("pregunta", _candidates(), top_n=2)

    assert len(result) == 2


def test_rerank_ignores_unknown_chunk_ids_from_llm_response():
    def fake_chat_fn(messages):
        return json.dumps({"ranked_chunk_ids": ["chunk-inventado", "d1::1", "d1::0", "d2::0"]})

    reranker = LLMReranker(chat_fn=fake_chat_fn)

    result = reranker.rerank("pregunta", _candidates(), top_n=3)

    assert [c.chunk_id for c in result] == ["d1::1", "d1::0", "d2::0"]


def test_rerank_appends_missing_candidates_after_ranked_ones():
    def fake_chat_fn(messages):
        return json.dumps({"ranked_chunk_ids": ["d1::1"]})

    reranker = LLMReranker(chat_fn=fake_chat_fn)

    result = reranker.rerank("pregunta", _candidates(), top_n=3)

    assert result[0].chunk_id == "d1::1"
    assert {c.chunk_id for c in result} == {"d1::0", "d1::1", "d2::0"}


def test_rerank_strips_markdown_code_fences_before_parsing():
    def fake_chat_fn(messages):
        return '```json\n{"ranked_chunk_ids": ["d1::1", "d2::0", "d1::0"]}\n```'

    reranker = LLMReranker(chat_fn=fake_chat_fn)

    result = reranker.rerank("pregunta", _candidates(), top_n=3)

    assert [c.chunk_id for c in result] == ["d1::1", "d2::0", "d1::0"]


def test_rerank_raises_on_malformed_llm_response():
    def fake_chat_fn(messages):
        return "esto no es json"

    reranker = LLMReranker(chat_fn=fake_chat_fn)

    with pytest.raises(ValueError):
        reranker.rerank("pregunta", _candidates(), top_n=2)

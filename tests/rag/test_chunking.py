"""Tests para credixai.rag.chunking (paso 5, prd.md 9.1).

TDD: este archivo se escribe antes que credixai/rag/chunking.py. Chunking es
logica pura (split de texto), sin llamadas a red, por eso vive en la suite
rapida sin marcar como integration.
"""

import pytest

from credixai.rag.chunking import Chunk, chunk_document, chunk_text


def test_chunk_text_respects_max_words_per_chunk():
    text = " ".join(f"palabra{i}" for i in range(250))

    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert all(len(c.split()) <= 100 for c in chunks)


def test_chunk_text_overlap_repeats_words_across_consecutive_chunks():
    text = " ".join(f"palabra{i}" for i in range(250))

    chunks = chunk_text(text, chunk_size=100, overlap=20)

    first_tail = chunks[0].split()[-20:]
    second_head = chunks[1].split()[:20]
    assert first_tail == second_head


def test_chunk_text_short_document_returns_single_chunk():
    text = "una politica corta de pocas palabras"

    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert chunks == [text]


def test_chunk_text_rejects_overlap_not_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("algun texto", chunk_size=50, overlap=50)


def test_chunk_document_assigns_stable_ids_and_carries_doc_metadata():
    text = " ".join(f"palabra{i}" for i in range(250))

    chunks = chunk_document(doc_id="doc-1", title="Politica de ejemplo", text=text, chunk_size=100, overlap=20)

    assert all(isinstance(c, Chunk) for c in chunks)
    assert [c.chunk_id for c in chunks] == ["doc-1::0", "doc-1::1", "doc-1::2"]
    assert all(c.doc_id == "doc-1" for c in chunks)
    assert all(c.title == "Politica de ejemplo" for c in chunks)


def test_chunk_document_single_chunk_document():
    chunks = chunk_document(doc_id="doc-2", title="Corta", text="texto breve", chunk_size=100, overlap=20)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "doc-2::0"

"""Tests para credixai.rag.corpus (paso 5).

TDD: se escribe antes que credixai/rag/corpus.py. Usa un directorio temporal
con markdown sintetico, no el corpus real de docs/policy_corpus, para no
acoplar el test al contenido real de las politicas.
"""

from credixai.rag.corpus import load_corpus


def test_load_corpus_reads_markdown_files_and_extracts_title(tmp_path):
    (tmp_path / "01_a.md").write_text("# Politica A\n\nContenido de A.\n")
    (tmp_path / "02_b.md").write_text("# Politica B\n\nContenido de B.\n")

    docs = load_corpus(tmp_path)

    assert {d.doc_id for d in docs} == {"01_a", "02_b"}
    titles = {d.doc_id: d.title for d in docs}
    assert titles["01_a"] == "Politica A"
    assert titles["02_b"] == "Politica B"


def test_load_corpus_strips_title_heading_from_body_text(tmp_path):
    (tmp_path / "01_a.md").write_text("# Politica A\n\nContenido de A.\n")

    docs = load_corpus(tmp_path)

    assert "Politica A" not in docs[0].text
    assert "Contenido de A." in docs[0].text


def test_load_corpus_ignores_non_markdown_files(tmp_path):
    (tmp_path / "01_a.md").write_text("# Politica A\n\nContenido.\n")
    (tmp_path / "notes.txt").write_text("no deberia cargarse")

    docs = load_corpus(tmp_path)

    assert len(docs) == 1


def test_load_corpus_sorted_by_doc_id(tmp_path):
    (tmp_path / "02_b.md").write_text("# B\n\ntexto b\n")
    (tmp_path / "01_a.md").write_text("# A\n\ntexto a\n")

    docs = load_corpus(tmp_path)

    assert [d.doc_id for d in docs] == ["01_a", "02_b"]

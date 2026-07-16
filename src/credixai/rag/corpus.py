"""Carga del corpus de politicas (paso 5).

Cada archivo markdown en el directorio de corpus es un documento: el nombre
de archivo (sin extension) es el doc_id, y el primer encabezado `# ...` es
el titulo, que se separa del cuerpo antes de chunkear.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


def _split_title(raw: str) -> tuple[str, str]:
    lines = raw.strip().splitlines()
    title = lines[0].lstrip("#").strip()
    body = "\n".join(lines[1:]).strip()
    return title, body


def load_corpus(corpus_dir: Path) -> list[Document]:
    documents = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        title, body = _split_title(path.read_text())
        documents.append(Document(doc_id=path.stem, title=title, text=body))
    return documents

"""Parseo de JSON devuelto por LLMs que a veces lo envuelven en fences markdown.

Compartido entre credixai.rag.reranker y credixai.copilot.evaluator: ambos
piden JSON puro a un LLM via prompt y necesitan tolerar que igual llegue
envuelto en ```json ... ``` pese a la instruccion explicita de no hacerlo
(bug real encontrado durante el desarrollo del RAG, ver docs/informe-final.md).
"""

import re


def strip_markdown_fences(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1) if match else text

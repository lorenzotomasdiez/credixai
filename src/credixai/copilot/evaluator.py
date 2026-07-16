"""Evaluator del loop evaluator-optimizer (RF-6, paso 6).

run_precheck es una capa determinista, sin LLM, que corre antes del juez LLM:
si algo aca falla no tiene sentido gastar una llamada de generacion para
juzgar un memo que ya sabemos que esta mal formado.
"""

import json
from typing import Callable

from credixai.llm_json import strip_markdown_fences

ChatFn = Callable[[list[dict]], str]

MAX_REASON_CODES = 4

_JUDGE_SYSTEM_PROMPT = (
    "Sos un auditor de memos crediticios en CrediXAI. Te doy el contexto (score, "
    "reason codes, citas de politica) y un memo redactado a partir de ese contexto. "
    "Decidi si el memo afirma UNICAMENTE cosas soportadas por el contexto, sin "
    "inventar razones ni citas. Devolves SOLO un JSON con la forma "
    '{"approved": true|false, "feedback": "..."}. No incluyas texto fuera del JSON.'
)

_PROTECTED_ATTRIBUTE_KEYWORDS = [
    "genero",
    "género",
    "sexo",
    "edad",
    "nacionalidad",
    "etnia",
    "raza",
    "religion",
    "religión",
    "estado civil",
]


def run_precheck(memo: str, decision: str, reason_codes: list[str], citations: list[str]) -> list[str]:
    issues = []

    if not memo.strip():
        issues.append("memo vacio")

    if len(reason_codes) > MAX_REASON_CODES:
        issues.append("mas de 4 reason codes")

    if decision == "alto_riesgo" and not citations:
        issues.append("decision de alto riesgo sin cita de politica")

    memo_lower = memo.lower()
    found_keywords = [kw for kw in _PROTECTED_ATTRIBUTE_KEYWORDS if kw in memo_lower]
    if found_keywords:
        issues.append(f"memo menciona atributo protegido: {', '.join(found_keywords)}")

    return issues


def llm_judge(memo: str, context: str, chat_fn: ChatFn) -> tuple[bool, str]:
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Contexto:\n{context}\n\nMemo:\n{memo}"},
    ]

    raw_response = chat_fn(messages)
    try:
        parsed = json.loads(strip_markdown_fences(raw_response))
        return bool(parsed["approved"]), str(parsed["feedback"])
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Respuesta de evaluacion invalida: {raw_response!r}") from exc

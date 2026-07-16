"""Redaccion del memo crediticio (RF-6, paso 6).

draft_memo redacta el memo a partir de datos ya calculados por las tools
(score, reason_codes, citas de politica): el LLM narra, no decide las
razones ni elige que citar, mismo principio anti-alucinacion que
credixai.rag.generation.PolicyAnswerer (citas derivadas de datos, no
parseadas de texto libre del LLM).
"""

from typing import Callable

ChatFn = Callable[[list[dict]], str]

_SYSTEM_PROMPT = (
    "Sos el copiloto de un analista de credito en CrediXAI. Redactas un memo "
    "crediticio breve y claro a partir del score, los reason codes y las citas "
    "de politica ya provistos. No inventes razones que no esten en los reason "
    "codes provistos, no cites politica que no este en las citas provistas, y "
    "no menciones ni discutas atributos protegidos (genero, edad, etc.) como "
    "criterio de decision."
)


def draft_memo(
    score: dict,
    reason_codes: list[str],
    policy_citations: list[str],
    chat_fn: ChatFn,
    revision_feedback: str | None = None,
) -> str:
    reason_codes_block = "\n".join(f"- {r}" for r in reason_codes) or "(sin reason codes: riesgo aceptable)"
    citations_block = "\n".join(f"- {c}" for c in policy_citations) or "(sin citas de politica)"

    user_content = (
        f"Score: sk_id_curr={score['sk_id_curr']}, probability={score['probability']}, "
        f"threshold={score['threshold']}, decision={score['decision']}\n\n"
        f"Reason codes:\n{reason_codes_block}\n\n"
        f"Citas de politica:\n{citations_block}"
    )
    if revision_feedback:
        user_content += f"\n\nEsta es una revision. Feedback del evaluador anterior:\n{revision_feedback}"

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return chat_fn(messages)

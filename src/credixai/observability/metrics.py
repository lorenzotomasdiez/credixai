"""Metricas puras para observabilidad Langfuse (RNF-4, paso 7).

Funciones sin efectos de red: traducen respuestas de OpenAI/OpenRouter y
estado del copiloto a los datos que se adjuntan como usage/score en
Langfuse. Se testean solas, sin Langfuse real.
"""

from typing import Any


def extract_usage(response: Any) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {
        "input": getattr(usage, "prompt_tokens", None),
        "output": getattr(usage, "completion_tokens", None),
        "total": getattr(usage, "total_tokens", None),
    }


def first_try_success(iteration: int, status: str) -> bool:
    return status == "approved" and iteration == 0

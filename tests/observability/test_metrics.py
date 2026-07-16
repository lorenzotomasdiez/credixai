"""Tests para credixai.observability.metrics (paso 7, prd.md 9.1).

TDD: se escribe antes que credixai/observability/metrics.py.
Funciones puras (sin red, sin Langfuse real) que traducen resultados del
copiloto y respuestas de OpenAI/OpenRouter a los datos que despues se
adjuntan como score/usage en Langfuse.
"""

from types import SimpleNamespace

from credixai.observability.metrics import extract_usage, first_try_success


def test_extract_usage_reads_prompt_completion_and_total_tokens():
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15))

    usage = extract_usage(response)

    assert usage == {"input": 10, "output": 5, "total": 15}


def test_extract_usage_handles_missing_completion_tokens_for_embeddings():
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=8, total_tokens=8))

    usage = extract_usage(response)

    assert usage == {"input": 8, "output": None, "total": 8}


def test_extract_usage_returns_empty_dict_when_response_has_no_usage():
    response = SimpleNamespace()

    usage = extract_usage(response)

    assert usage == {}


def test_first_try_success_true_when_approved_on_first_iteration():
    assert first_try_success(iteration=0, status="approved") is True


def test_first_try_success_false_when_approved_after_a_retry():
    assert first_try_success(iteration=1, status="approved") is False


def test_first_try_success_false_when_needs_human_review():
    assert first_try_success(iteration=0, status="needs_human_review") is False

"""Tests para credixai.rag.openrouter_client (paso 5).

TDD: se escribe antes que credixai/rag/openrouter_client.py.
La validacion de configuracion (falta de API key) es un limite del sistema
(RNF-7 dice validar en los bordes), asi que se testea sin red.
Las llamadas reales a OpenRouter se verifican por separado en
tests/rag/test_openrouter_client_integration.py (marcado integration).
"""

import pytest

from credixai.rag.openrouter_client import OpenRouterClient


def test_raises_when_no_api_key_available(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        OpenRouterClient(api_key=None)


def test_accepts_explicit_api_key_even_without_env_var(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    client = OpenRouterClient(api_key="sk-explicit")

    assert client is not None

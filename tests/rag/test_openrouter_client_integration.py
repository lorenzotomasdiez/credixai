"""Verificacion real contra OpenRouter (paso 5).

Marcado integration: no corre en CI (ver ci.yml, `pytest -m "not integration"`)
porque requiere OPENROUTER_API_KEY real y consume creditos, aunque minimos.
Se corre manualmente antes de dar por cerrado el paso 5, como el resto de la
verificacion real (mismo criterio que docker_smoke.sh o get_service()).
"""

import pytest

from credixai.rag.openrouter_client import OpenRouterClient

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    return OpenRouterClient()


def test_embed_returns_nonempty_float_vector(client):
    vector = client.embed("politica de riesgo crediticio")

    assert len(vector) > 0
    assert all(isinstance(v, float) for v in vector)


def test_embed_batch_returns_one_vector_per_text(client):
    vectors = client.embed_batch(["texto uno", "texto dos", "texto tres"])

    assert len(vectors) == 3
    assert len({len(v) for v in vectors}) == 1


def test_chat_returns_nonempty_text_response(client):
    response = client.chat([{"role": "user", "content": "Respondé solo con la palabra: listo"}])

    assert isinstance(response, str)
    assert len(response) > 0

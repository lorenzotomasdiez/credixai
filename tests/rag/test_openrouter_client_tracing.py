"""Tests para la instrumentacion Langfuse de OpenRouterClient (paso 7).

TDD: OpenRouterClient es el unico choke point por el que pasan todas las
llamadas a LLM del proyecto (RAG y copiloto), asi que se instrumenta una
sola vez aca en vez de en cada call-site. Se testea con un fake Langfuse
client inyectado (mismo patron de DI que ya usa la clase para api_key),
sin tocar la red ni el SDK real de Langfuse.
"""

from types import SimpleNamespace

from credixai.rag.openrouter_client import OpenRouterClient


class _FakeObservation:
    def __init__(self):
        self.updates = []

    def update(self, **kwargs):
        self.updates.append(kwargs)


class _FakeObservationContext:
    def __init__(self, observation):
        self._observation = observation

    def __enter__(self):
        return self._observation

    def __exit__(self, *exc_info):
        return False


class _FakeLangfuseClient:
    def __init__(self):
        self.calls = []
        self.observation = _FakeObservation()

    def start_as_current_observation(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeObservationContext(self.observation)


def _fake_chat_response(content, prompt_tokens=10, completion_tokens=5, total_tokens=15):
    message = SimpleNamespace(content=content, tool_calls=None)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def test_chat_wraps_call_in_a_langfuse_generation(monkeypatch):
    fake_langfuse = _FakeLangfuseClient()
    client = OpenRouterClient(api_key="sk-explicit", langfuse_client=fake_langfuse)
    monkeypatch.setattr(
        client._client.chat.completions, "create", lambda **kwargs: _fake_chat_response("hola")
    )

    result = client.chat(messages=[{"role": "user", "content": "hola"}])

    assert result == "hola"
    assert len(fake_langfuse.calls) == 1
    call = fake_langfuse.calls[0]
    assert call["as_type"] == "generation"
    assert call["model"] == client._chat_model
    update = fake_langfuse.observation.updates[0]
    assert update["output"] == "hola"
    assert update["usage_details"] == {"input": 10, "output": 5, "total": 15}


def test_embed_batch_wraps_call_in_a_langfuse_embedding_generation(monkeypatch):
    fake_langfuse = _FakeLangfuseClient()
    client = OpenRouterClient(api_key="sk-explicit", langfuse_client=fake_langfuse)
    embeddings_response = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2])],
        usage=SimpleNamespace(prompt_tokens=3, total_tokens=3),
    )
    monkeypatch.setattr(client._client.embeddings, "create", lambda **kwargs: embeddings_response)

    result = client.embed_batch(["texto"])

    assert result == [[0.1, 0.2]]
    call = fake_langfuse.calls[0]
    assert call["as_type"] == "generation"
    assert call["model"] == client._embedding_model
    update = fake_langfuse.observation.updates[0]
    assert update["usage_details"] == {"input": 3, "output": None, "total": 3}


def test_chat_with_tools_wraps_call_in_a_langfuse_generation(monkeypatch):
    fake_langfuse = _FakeLangfuseClient()
    client = OpenRouterClient(api_key="sk-explicit", langfuse_client=fake_langfuse)
    fake_tool_call = SimpleNamespace(
        id="call_1", function=SimpleNamespace(name="score_application", arguments="{}")
    )
    message = SimpleNamespace(content=None, tool_calls=[fake_tool_call])
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2, total_tokens=6),
    )
    monkeypatch.setattr(client._client.chat.completions, "create", lambda **kwargs: response)

    result = client.chat_with_tools(messages=[{"role": "user", "content": "hola"}], tools=[])

    assert len(result.tool_calls) == 1
    call = fake_langfuse.calls[0]
    assert call["as_type"] == "generation"
    update = fake_langfuse.observation.updates[0]
    assert update["usage_details"] == {"input": 4, "output": 2, "total": 6}


def test_default_langfuse_client_is_used_when_none_is_injected(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    client = OpenRouterClient(api_key="sk-explicit")

    assert client._langfuse is not None

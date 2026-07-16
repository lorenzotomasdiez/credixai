"""Tests para OpenRouterClient.chat_with_tools (paso 6).

TDD: chat_with_tools es lo que necesita el orchestrator del copiloto para
tool-calling real. Se testea sin red, monkeypatcheando el cliente OpenAI
interno con una respuesta con forma real de tool_calls.
"""

from types import SimpleNamespace

from credixai.rag.openrouter_client import OpenRouterClient


def _fake_response(content, tool_calls):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_chat_with_tools_returns_content_and_tool_calls_when_llm_calls_a_tool(monkeypatch):
    client = OpenRouterClient(api_key="sk-explicit")

    fake_tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="score_application", arguments='{"sk_id_curr": 100002}'),
    )

    def fake_create(model, messages, tools=None):
        return _fake_response(content=None, tool_calls=[fake_tool_call])

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)

    tools = [{"type": "function", "function": {"name": "score_application", "parameters": {}}}]
    result = client.chat_with_tools(messages=[{"role": "user", "content": "hola"}], tools=tools)

    assert result.content is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "score_application"
    assert result.tool_calls[0].arguments == {"sk_id_curr": 100002}
    assert result.tool_calls[0].id == "call_1"


def test_chat_with_tools_returns_empty_tool_calls_when_llm_answers_directly(monkeypatch):
    client = OpenRouterClient(api_key="sk-explicit")

    def fake_create(model, messages, tools=None):
        return _fake_response(content="ya tengo todo lo que necesito", tool_calls=None)

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)

    result = client.chat_with_tools(messages=[{"role": "user", "content": "hola"}], tools=[])

    assert result.content == "ya tengo todo lo que necesito"
    assert result.tool_calls == []

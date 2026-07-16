"""Cliente delgado sobre OpenRouter (embeddings + chat), paso 5, prd.md 9.1.

Un unico provider (OpenRouter) para embeddings, generacion grounded y
reranking, elegido para no fragmentar la configuracion de LLM en el
proyecto. Modelos elegidos por costo/calidad (ver docs/informe-final.md
seccion RAG): openai/text-embedding-3-small para embeddings, openai/gpt-4o-mini
para chat/rerank.
"""

import json
import os
from dataclasses import dataclass, field

from openai import OpenAI

DEFAULT_CHAT_MODEL = "openai/gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class ChatWithToolsResult:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class OpenRouterClient:
    def __init__(
        self,
        api_key: str | None = None,
        chat_model: str = DEFAULT_CHAT_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY no esta configurada (env var o .env). "
                "Ver .env.example."
            )
        self._client = OpenAI(base_url=_BASE_URL, api_key=api_key)
        self._chat_model = chat_model
        self._embedding_model = embedding_model

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._embedding_model, input=texts)
        return [item.embedding for item in response.data]

    def chat(self, messages: list[dict]) -> str:
        response = self._client.chat.completions.create(model=self._chat_model, messages=messages)
        return response.choices[0].message.content

    def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> ChatWithToolsResult:
        response = self._client.chat.completions.create(
            model=self._chat_model, messages=messages, tools=tools
        )
        message = response.choices[0].message
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments))
            for tc in (message.tool_calls or [])
        ]
        return ChatWithToolsResult(content=message.content, tool_calls=tool_calls)

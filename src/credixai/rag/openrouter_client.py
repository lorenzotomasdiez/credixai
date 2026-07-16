"""Cliente delgado sobre OpenRouter (embeddings + chat), paso 5, prd.md 9.1.

Un unico provider (OpenRouter) para embeddings, generacion grounded y
reranking, elegido para no fragmentar la configuracion de LLM en el
proyecto. Modelos elegidos por costo/calidad (ver docs/informe-final.md
seccion RAG): openai/text-embedding-3-small para embeddings, openai/gpt-4o-mini
para chat/rerank.

Choke point unico de observabilidad (paso 7, prd.md 9.1): todas las
llamadas a LLM del proyecto (RAG y copiloto) pasan por esta clase, asi que
se instrumentan con Langfuse aca en vez de en cada call-site.
"""

import json
import os
from dataclasses import dataclass, field

from openai import OpenAI

from credixai.observability.metrics import extract_usage

DEFAULT_CHAT_MODEL = "openai/gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
_BASE_URL = "https://openrouter.ai/api/v1"


def _default_langfuse_client():
    from langfuse import get_client

    return get_client()


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
        langfuse_client=None,
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
        self._langfuse = langfuse_client if langfuse_client is not None else _default_langfuse_client()

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        with self._langfuse.start_as_current_observation(
            name="embed", as_type="generation", model=self._embedding_model, input=texts
        ) as generation:
            response = self._client.embeddings.create(model=self._embedding_model, input=texts)
            embeddings = [item.embedding for item in response.data]
            generation.update(output=f"{len(embeddings)} embeddings", usage_details=extract_usage(response))
            return embeddings

    def chat(self, messages: list[dict]) -> str:
        with self._langfuse.start_as_current_observation(
            name="chat", as_type="generation", model=self._chat_model, input=messages
        ) as generation:
            response = self._client.chat.completions.create(model=self._chat_model, messages=messages)
            content = response.choices[0].message.content
            generation.update(output=content, usage_details=extract_usage(response))
            return content

    def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> ChatWithToolsResult:
        with self._langfuse.start_as_current_observation(
            name="chat_with_tools", as_type="generation", model=self._chat_model, input=messages
        ) as generation:
            response = self._client.chat.completions.create(
                model=self._chat_model, messages=messages, tools=tools
            )
            message = response.choices[0].message
            tool_calls = [
                ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments))
                for tc in (message.tool_calls or [])
            ]
            generation.update(
                output={"content": message.content, "tool_calls": [tc.name for tc in tool_calls]},
                usage_details=extract_usage(response),
            )
            return ChatWithToolsResult(content=message.content, tool_calls=tool_calls)

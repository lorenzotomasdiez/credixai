"""Cliente HTTP del dashboard hacia la API (paso 9, prd.md 9.1: UI de RAG/copiloto).

El dashboard consume /rag/query y /copilot/memo/{sk_id_curr} por HTTP en vez
de importar credixai.rag/credixai.copilot directo, mismo principio ya usado
en credixai/copilot/tools.py para las tools del copiloto: evita duplicar el
setup de Qdrant/OpenRouter/LangGraph (async) dentro de Streamlit.
"""

import httpx


def query_policy(client: httpx.Client, question: str) -> dict:
    response = client.post("/rag/query", json={"question": question})
    response.raise_for_status()
    return response.json()


def request_copilot_memo(client: httpx.Client, sk_id_curr: int) -> dict:
    response = client.post(f"/copilot/memo/{sk_id_curr}")
    response.raise_for_status()
    return response.json()
